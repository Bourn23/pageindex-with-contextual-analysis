# Evaluates the advanced pipeline
## Added metrics for evaluation
# Evaluates the advanced pipeline with Recall, Precision, and Numeric Accuracy
import argparse
import json
import os
import math
import pandas as pd
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# ==============================================================================
# 1. Helper: Unit Standardization
# ==============================================================================
def standardize_conductivity(value: float, unit_str: str) -> Optional[float]:
    """
    Standardizes ionic conductivity to S/cm based on unit string.
    Returns None if value is missing or invalid.
    """
    if value is None:
        return None
    
    # If unit is None, we assume the value is already in the target unit 
    # OR we return raw value if we trust the source. 
    # Here, let's be strict: if unit is totally missing, return raw value (assuming standard)
    if not unit_str:
        try: return float(value)
        except: return None

    try:
        val = float(value)
    except (ValueError, TypeError):
        return None

    u_clean = str(unit_str).lower().replace(" ", "").replace("·", "").replace(".", "")
    
    # Base Multiplier
    multiplier = 1.0
    
    # 1. Metric Prefix
    if "ms" in u_clean:          # Milli (10^-3)
        multiplier = 1e-3
    elif "us" in u_clean or "μs" in u_clean or "µs" in u_clean: # Micro (10^-6)
        multiplier = 1e-6
    elif "ns" in u_clean:        # Nano (10^-9)
        multiplier = 1e-9
    elif "ks" in u_clean:        # Kilo (10^3)
        multiplier = 1000.0
    
    # 2. Geometry (cm vs m)
    # Target is S/cm. 
    # If unit is S/m, we must divide by 100 (1 S/m = 0.01 S/cm)
    # We check for "m" explicitly without "cm" or "mm"
    if "m" in u_clean and "cm" not in u_clean and "mm" not in u_clean:
         # Check for inverse meters (m-1) or per meter (/m) or S/m
         if "m-1" in u_clean or "/m" in u_clean or "sm-1" in u_clean:
             multiplier *= 0.01

    return val * multiplier

# ==============================================================================
# 2. Pydantic Schema
# ==============================================================================
class MatchResult(BaseModel):
    ground_truth_id: str = Field(..., description="The ID from the Ground Truth input (e.g., 'GT-0').")
    extracted_id: Optional[str] = Field(None, description="The ID from the Extracted input that matches (e.g., 'EX-2'). If no match, return None.")
    reason: str = Field(..., description="Explanation of why this is a match or mismatch.")
    is_match: bool = Field(..., description="True if the material/composition and physical context match.")

class BenchmarkReport(BaseModel):
    matches: List[MatchResult]
    summary_thought: str = Field(..., description="Brief summary of extraction performance.")

# ==============================================================================
# 3. The "Judge" Prompt
# ==============================================================================
JUDGE_PROMPT = """
You are a Scientific Data Auditor. 
Compare "Ground Truth" (human-curated) vs "Extracted Data" (AI-pipeline).

GOAL: Link every Ground Truth (GT) row to a matching Extracted (EX) row.

RULES FOR MATCHING:
1. Chemical Formulas: Normalize stoichiometries (e.g., "Li7La3Zr2O12" == "LLZO").
2. Context: Ensure temperature conditions are roughly similar (e.g., don't match High-T measurement to Room Temp).
3. IDs: You MUST return the specific identifiers (e.g. GT-0, EX-5).

INPUT DATA:
Ground Truth:
{ground_truth}

Extracted Data:
{extracted_data}

TASK:
For every row in Ground Truth:
1. Find the corresponding row in Extracted Data representing the same physical entity.
2. If found, mark `is_match`=True and provide `extracted_id`.
3. If not found, mark `is_match`=False.
"""

def main():
    parser = argparse.ArgumentParser(description='Benchmark Validator: Recall + Log Accuracy')
    parser.add_argument('--ground-truth', '-gt', required=True, help='Path to Ground Truth CSV')
    parser.add_argument('--doi', '-i', required=True, help='DOI of the paper')
    parser.add_argument('--extracted', '-ex', required=True, help='Path to Extracted Results JSON')
    parser.add_argument('--model', default='gemini-2.5-flash', help='Model for judging')
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})

    # --- 1. Load & Process Ground Truth ---
    try:
        df = pd.read_csv(args.ground_truth)
        df = df[df['DOI'] == args.doi]
        if df.empty:
            print(f"No GT found for DOI: {args.doi}")
            return
    except Exception as e:
        print(f"Error loading GT: {e}")
        return

    # Map for easy lookup later: { "GT-0": 0.0012, "GT-1": 0.005 } (values in S/cm)
    gt_map = {}
    gt_rows_text = []
    
    for idx, row in df.iterrows():
        gt_id = f"GT-{idx}"
        
        comp = row.get('Composition', 'Unknown')
        raw_cond = row.get('Ionic conductivity (S cm-1)', None)
        
        # NOTE: Adjust unit string 'S/cm' if your CSV headers differ
        # Assuming header implies S/cm
        std_cond = standardize_conductivity(raw_cond, "S/cm")
        
        gt_map[gt_id] = std_cond
        gt_rows_text.append(f"{gt_id}: {comp} | {raw_cond} S/cm")

    gt_str = "\n".join(gt_rows_text)
    total_gt = len(gt_rows_text)

    # --- 2. Load & Process Extracted Data ---
    try:
        if args.extracted.endswith('.json'):
            with open(args.extracted, 'r') as f:
                data = json.load(f)
                ex_list = data.get('materials', []) if isinstance(data, dict) else data
        else:
            print("Only JSON supported.")
            return
    except Exception as e:
        print(f"Error loading Extracted: {e}")
        return

    # Map for extracted values
    ex_map = {}
    ex_rows_text = []

    for idx, row in enumerate(ex_list):
        ex_id = f"EX-{idx}"
        
        # Formula Display
        formula = row.get('canonical_formula') or row.get('original_name') or "Unknown"
        if isinstance(row.get('electrolyte_name'), dict):
            formula = row['electrolyte_name'].get('full_name', formula)

        # Conductivity Handling: Prefer _norm_cond, fall back to raw
        val = row.get('_norm_cond')
        if val is None:
            raw_val = row.get('ionic_conductivity_S_per_cm')
            # If unit is mixed in string or implicit S/cm
            if raw_val is not None:
                val = standardize_conductivity(raw_val, "S/cm")

        ex_map[ex_id] = val
        
        temp = row.get('_norm_temp') or row.get('measurement_temperature') or "N/A"
        ex_rows_text.append(f"{ex_id}: {formula} | {val} S/cm | {temp}")

    ex_str = "\n".join(ex_rows_text)
    total_ex = len(ex_rows_text)

    # --- 3. Run LLM Judge ---
    print(f"--- Validating {args.doi} ---")
    formatted_prompt = JUDGE_PROMPT.format(ground_truth=gt_str, extracted_data=ex_str)
    
    try:
        response = client.models.generate_content(
            model=args.model,
            contents=formatted_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=BenchmarkReport.model_json_schema(),
                temperature=0.0
            )
        )
        
        report = BenchmarkReport.model_validate_json(response.text)

        # --- 4. Calculate Metrics ---
        tp_gt_ids = set()
        tp_ex_ids = set()
        log_errors = [] 

        # Accuracy counter
        accurate_matches_count = 0
        LOG_TOLERANCE = 0.5
        
        print("\n" + "="*80)
        print(f"{'GT ID':<8} | {'EX ID':<8} | {'GT Val':<10} | {'EX Val':<10} | {'Log10 Err':<10} | {'Status'}")
        print("-" * 80)

        with open(args.extracted.replace('.json', '_advanced.txt'), 'w') as f:
            f.write("\n" + "="*80 + "\n")
            f.write(f"{'GT ID':<8} | {'EX ID':<8} | {'GT Val':<10} | {'EX Val':<10} | {'Log10 Err':<10} | {'Status'}\n")
            f.write("-" * 80 + "\n")
            for match in report.matches:
                status = "❌ MISSING"
                gt_val = gt_map.get(match.ground_truth_id)
                ex_val = None
                log_err = "N/A"

                if match.is_match and match.extracted_id:
                    # Sanitize ID (handle "EX-1 (approx)" cases)
                    clean_ex_id = match.extracted_id.split(":")[0].strip().split(" ")[0]
                    
                    if clean_ex_id in ex_map:
                        status = "✅ MATCH"
                        tp_gt_ids.add(match.ground_truth_id)
                        tp_ex_ids.add(clean_ex_id)
                        
                        ex_val = ex_map[clean_ex_id]
                        
                        # --- NUMERIC ACCURACY (LOG10 ERROR) ---
                        if gt_val and ex_val and gt_val > 0 and ex_val > 0:
                            try:
                                # Log10 Error Calculation
                                err = abs(math.log10(gt_val) - math.log10(ex_val))
                                log_errors.append(err)
                                log_err = f"{err:.4f}"

                                if err <= LOG_TOLERANCE:
                                    is_accurate = True
                                    accurate_matches_count += 1
                                    status = "✅ Accurate"
                                else:
                                    status = "❌ Inaccurate"
                            except:
                                log_err = "Math Err"
                        else:
                            log_err = "Null/Zero"
                            status = "❌ No Value"

                report_text = f"{match.ground_truth_id:<8} | {str(match.extracted_id)[:8]:<8} | {str(gt_val)[:10]:<10} | {str(ex_val)[:10]:<10} | {str(log_err):<10} | {status}"
                f.write(report_text + "\n")
                print(report_text)

        # Summary Calcs
        recall = (len(tp_gt_ids) / total_gt * 100) if total_gt > 0 else 0
        precision = (len(tp_ex_ids) / total_ex * 100) if total_ex > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        avg_log_error = sum(log_errors) / len(log_errors) if log_errors else 0.0

        num_numeric_matches = len(log_errors)
        numeric_accuracy_pct = (accurate_matches_count / num_numeric_matches * 100) if num_numeric_matches > 0 else 0.0
        
        # Save Results
        metrics = {
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "avg_log10_error": avg_log_error,
            "numeric_accuracy_pct": numeric_accuracy_pct,
            "num_valid_numeric_matches": len(log_errors),
            "total_gt": total_gt,
            "total_ex": total_ex
        }
        
        with open(args.extracted.replace('.json', '_advanced.txt'), 'a') as f:
            f.write("="*80 + "\n")
            f.write(f"RESULTS SUMMARY" + "\n")
            f.write(f"Recall:      {recall:.2f}%" + "\n")
            f.write(f"Precision:   {precision:.2f}%" + "\n")
            f.write(f"F1 Score:    {f1:.2f}" + "\n")
            f.write(f"Numeric Accuracy: {numeric_accuracy_pct:.2f}%" + "\n")
            f.write(f"Avg Log10 Error: {avg_log_error:.4f} (Lower is better)" + "\n")
            f.write("="*80 + "\n")

            ## Also print them
            print("\n" + "="*80 + "\n")
            print(f"RESULTS SUMMARY")
            print(f"Recall:      {recall:.2f}%")
            print(f"Precision:   {precision:.2f}%")
            print(f"F1 Score:    {f1:.2f}")
            print(f"Numeric Accuracy: {numeric_accuracy_pct:.2f}%")
            print(f"Avg Log10 Error: {avg_log_error:.4f} (Lower is better)")
            print("="*80)

        out_path = args.extracted.replace('.json', '_metrics.json')
        with open(out_path, 'w') as f:
            final_out = report.model_dump()
            final_out['metrics'] = metrics
            json.dump(final_out, f, indent=2)
            print(f"Saved to {out_path}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Validation Failed: {e}")

if __name__ == "__main__":
    main()