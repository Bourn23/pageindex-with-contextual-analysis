# Evaluates the advanced pipeline
### It splits the ground truth into chunks and processes them in parallel
import argparse
import json
import os
import time
import pandas as pd
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# ==============================================================================
# 1. The Schema for the Judge
# ==============================================================================
class MatchResult(BaseModel):
    ground_truth_id: str = Field(..., description="The ID or Index from the original data.")
    extracted_match: str = Field(..., description="The matching extracted composition/value, or 'MISSING'.")
    reason: str = Field(..., description="One sentence explanation. E.g. 'Li24... simplifies to Li6... and 0.00277 S/cm is approx 2.8 mS/cm'.")
    thinking: str = Field(..., description="Your thinking process. 2-3 sentences.")
    is_match: bool = Field(..., description="True if the extracted data represents the same physical reality as the ground truth.")

class BenchmarkReport(BaseModel):
    matches: List[MatchResult]

# ==============================================================================
# 2. The "Judge" Prompt
# ==============================================================================
JUDGE_PROMPT = """
You are a strict Scientific Data Auditor. 
Compare the "Ground Truth" data (from a reliable database) against the "Extracted Data" (from an AI pipeline).

Your Goal: Determine if the Extracted Data correctly captures the Ground Truth.

RULES FOR MATCHING:
1. Chemical Formulas: Normalize stoichiometries. 
   - Example: "Li24P4S20Br4" is chemically identical to "Li6PS5Br" (divide by 4).
   - Example: "Li7La3Zr2O12" is the same as "LLZO".
   
2. Units & Values: Check for unit conversions and allow for approximate matches.
   - Example: "0.00277 S/cm" is approx "2.8 mS/cm".
   - Allow for rounding differences (e.g. 0.00277 vs 0.0028 or 3.64mS/cm vs 3.7mS/cm).
   
3. Fuzzy Logic:
   - If the extraction says "RT" or "Room Temp" and Ground Truth is missing temp or implies 25C, that is acceptable.

INPUT DATA:
Ground Truth:
{ground_truth}

Extracted Data:
{extracted_data}

TASK:
For EVERY row in Ground Truth, find the corresponding row in Extracted Data.
Report if it was found and if the values match.
"""

def main():
    parser = argparse.ArgumentParser(description='Benchmark Validator: LLM-as-a-Judge')
    parser.add_argument('--ground-truth', '-gt', required=True, help='Path to Ground Truth CSV')
    parser.add_argument('--doi', '-i', required=True, help='DOI of the paper')
    parser.add_argument('--extracted', '-ex', required=True, help='Path to Extracted Results JSON')
    parser.add_argument('--model', default='gemini-2.5-flash', help='Model for judging')
    parser.add_argument('--output', '-o', default='validation_report.json', help='Output report path')
    parser.add_argument('--batch-size', '-b', type=int, default=10, help='Number of GT items to process per LLM call')
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found.")
        return

    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})

    # 1. Load Ground Truth
    try:
        df = pd.read_csv(args.ground_truth)
        df = df[df['DOI'] == args.doi]
        if df.empty:
            print(f"No GT data found for DOI: {args.doi}")
            return
    except Exception as e:
        print(f"Error loading GT file: {e}")
        return
    
    # 2. Load Extracted Data
    try:
        if args.extracted.endswith('.json'):
            ex_data = json.load(open(args.extracted, 'r'))
            ex_materials = ex_data.get('materials', [])
        elif args.extracted.endswith('.csv'):
            # Basic handler if CSV, though your code implied JSON structure
            ex_raw = open(args.extracted, 'r').read()
            # Placeholder: You'd need actual CSV parsing here if you support it
            ex_materials = [] 
    except Exception as e:
        print(f"Error loading Extracted file: {e}")
        return

    print(f"--- Starting Validation ---")
    print(f"Ground Truth Items: {len(df)}")
    print(f"Extracted Items:    {len(ex_materials)}")

    # 3. Pre-format Extracted Data (Done ONCE, passed to every batch)
    ex_str_repr = []
    for idx, row in enumerate(ex_materials):
        # Resolve Formula
        formula = ""
        if 'canonical_formula' in row:
            formula = row['canonical_formula']
        elif 'original_name' in row:
            formula = row['original_name']
        elif 'electrolyte_name' in row:
            e_name = row['electrolyte_name']
            if isinstance(e_name, dict):
                formula = e_name.get('full_name', '')
                if e_name.get('acronym'): formula += " acronym: " + e_name['acronym']
                if e_name.get('proportion'): formula += " proportion: " + e_name['proportion']
        
        # Resolve Conductivity
        cond = row.get('_norm_cond', row.get('ionic_conductivity_S_per_cm', ''))
        
        # Resolve Temp
        temp = row.get('_norm_temp', row.get('measurement_temperature', ''))
        
        ex_str_repr.append(f"EX-{idx}: {formula} | {cond} | {temp}")
    
    full_extracted_str = "\n".join(ex_str_repr)

    # 4. Batch Processing Loop
    all_matches = []
    total_gt_rows = len(df)
    
    # Iterate through dataframe in chunks
    for start_idx in range(0, total_gt_rows, args.batch_size):
        end_idx = min(start_idx + args.batch_size, total_gt_rows)
        batch_df = df.iloc[start_idx:end_idx]
        
        print(f"Processing Batch {start_idx}-{end_idx} of {total_gt_rows}...")

        # Create GT string for just this batch
        # We use the original DataFrame index for ID to ensure traceability
        batch_gt_str = "\n".join([
            f"GT-{idx}: {row['Composition']} | {row['Ionic conductivity (S cm-1)']} S/cm" 
            for idx, row in batch_df.iterrows()
        ])

        formatted_prompt = JUDGE_PROMPT.format(
            ground_truth=batch_gt_str, 
            extracted_data=full_extracted_str
        )

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

            if response.text:
                report = BenchmarkReport.model_validate_json(response.text)
                all_matches.extend(report.matches)
            
            # Sleep briefly to avoid rate limits on large datasets
            time.sleep(1)

        except Exception as e:
            print(f"Batch {start_idx}-{end_idx} Failed: {e}")
            # Optional: Add retry logic here

    # 5. Final Aggregation & Reporting
    total_matches_count = sum(1 for m in all_matches if m.is_match)
    score = (total_matches_count / total_gt_rows * 100) if total_gt_rows > 0 else 0

    print("\n" + "="*60)
    print(f"FINAL VALIDATION SCORE: {score:.1f}% ({total_matches_count}/{total_gt_rows})")
    print("="*60)
    print(f"{'GT ID':<10} | {'Match Status':<15} | {'Reason'}")
    print("-" * 60)
    
    # Save Text Report
    txt_path = args.extracted.replace('.json', '_advanced.txt')
    with open(txt_path, 'w') as f:
        f.write(f"FINAL SCORE: {score:.1f}%\n\n")
        for match in all_matches:
            status = "✅ MATCH" if match.is_match else "❌ MISSING/BAD"
            # Print to console
            print(f"{str(match.ground_truth_id)[:10]:<10} | {status:<15} | {match.reason}")
            # Write to file
            f.write(f"{str(match.ground_truth_id)[:10]:<10} | {status:<15} | {match.reason}\n")

    # Save JSON Report
    json_path = args.extracted.replace('.json', '_advanced.json')
    final_output = {
        "score": score,
        "total_ground_truth": total_gt_rows,
        "total_matches": total_matches_count,
        "details": [m.model_dump() for m in all_matches]
    }
    
    with open(json_path, 'w') as f:
        json.dump(final_output, f, indent=2)
    
    print(f"\nFull report saved to {json_path}")

if __name__ == "__main__":
    main()