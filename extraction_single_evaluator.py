
import os
import json
import asyncio
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
GROUND_TRUTH_CSV = "OBELiX/data/processed.csv"
PDF_DIR = "fetched_papers/obelix_pdf"
RESULTS_DIR = "fetched_papers/obelix_parsed_v5_test1"
TARGET_SCRIPT = "basic_extraction_auto_improved.py"

# --- Prompts ---
DIAGNOSIS_PROMPT = """
You are a Senior Data Extraction Debugger.
Your goal is to analyze why a specific "Ground Truth" data point was missed by our extraction pipeline.

CONTEXT:
1. **Paper**: {paper_title}
2. **Missing Data**: 
   - Composition: {gt_comp}
   - Conductivity: {gt_val} S/cm
   - (This is what *should* have been extracted)
3. **Actually Extracted**: 
   {extracted_summary}

TASK:
1.  **Locate the Data**: Look at the provided PDF pages. Find where this specific Composition and Conductivity are mentioned. It could be in a Table, a Figure (plot), or the Text.
2.  **Analyze the Failure**: Why did we miss it? 
    - Is it in a complex table? 
    - Is it a "x=..." variable in a stoichiometry plot?
    - Is the unit weird? 
    - Is it an Arrhenius plot (1000/T)?
3.  **Propose a Fix**: Suggest a specific improvement to the extraction logic.
    - "Add regex allow X format"
    - "Update prompt to explicitly look for Arrhenius plots with 1000/T"
    - "Look for 'x' variable substitution in captions"

OUTPUT JSON:
{{
  "location": "Table 1 / Figure 3 / Text Page 4",
  "reason": "Explanation of failure",
  "suggestion": "Specific actionable improvement for the code"
}}
"""

FRAMEWORK_IMPROVER_PROMPT = """
You are a Lead Python Engineer specializing in LLM pipelines.
We have collected a list of specific improvements needed for our `basic_extraction_auto_improved.py` script based on failure analysis.

YOUR TASK:
Rewrite the `basic_extraction_auto_improved.py` script to incorporate these improvements.

INPUTS:
1. **Current Code**: The existing python script.
2. **Improvements**:
{improvements_list}

CONSTRAINTS:
- **Preserve Structure**: Do NOT rewrite the whole architecture. Only modify Prompts, Regexes, and Helper Functions (like `process_image`, `process_text`, `canonicalize`, `normalizers`).
- **Safety**: Ensure JSON parsing is wrapped in try/except.
- **Robustness**: If adding a new heuristic, make sure it doesn't break existing good extractions.

Output the FULL, VALID Python script code.
"""

class ExtractionOptimizer:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})
        self.gt_df = pd.read_csv(GROUND_TRUTH_CSV)
        
    def get_ground_truth_for_paper(self, title: str) -> List[Dict]:
        """
        Fuzzy matches paper title to Ground Truth CSV to get expected values.
        Using a simplified title matching since we don't have the DOI mapping handy in this context perfectly.
        """
        # Normalize title for matching
        norm_title = "".join(x for x in title.lower() if x.isalnum())
        
        # This is a heuristic lookup. In full prod we'd use the DOI logic.
        # For now, we will assume the validation report already did this and try to re-link or just use the report's "matches".
        # Better approach: Read the validation report which links GT-ID to success/failure.
        return []

    async def analyze_failure(self, paper_title: str, gt_item: Dict, extracted_data: List[Dict], pdf_path: Path):
        """
        Calls Gemini 1.5 Pro with PDF to diagnose failure.
        """
        print(f"   🔍 Analyzing failure: {gt_item['Composition']} ({gt_item['Ionic conductivity (S cm-1)']} S/cm)...")
        
        extracted_summary = "\n".join([
            f"- {m['raw_composition']} | {m['raw_conductivity']} {m['raw_conductivity_unit']}" 
            for m in extracted_data
        ]) or "No data extracted."

        prompt = DIAGNOSIS_PROMPT.format(
            paper_title=paper_title,
            gt_comp=gt_item['Composition'],
            gt_val=gt_item['Ionic conductivity (S cm-1)'],
            extracted_summary=extracted_summary
        )

        try:
            # Upload PDF if needed or pass as inline data (Gemini 1.5 supports this well)
            pdf_bytes = pdf_path.read_bytes()
            
            response = await self.client.aio.models.generate_content(
                model="gemini-2.5-pro", # Strong reasoning needed
                contents=[
                    types.Part(text=prompt),
                    types.Part(
                        inline_data=types.Blob(
                            mime_type="application/pdf",
                            data=pdf_bytes
                        )
                    )
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2
                )
            )
            
            return json.loads(response.text)
        except Exception as e:
            print(f"   ⚠️ Diagnosis failed: {e}")
            return None

    async def run(self):
        print("🚀 Starting Extraction Optimizer...")
        
        # 1. Identify Failures
        failures = []
        
        results_path = Path(RESULTS_DIR)
        for report_file in results_path.glob("*_validation_report.json"):
            paper_name = report_file.name.replace("_v5_extracted_validation_report.json", "")
            
            # Load Report & Extracted Data
            try:
                with open(report_file) as f: report = json.load(f)
                extracted_file_path = results_path / f"{paper_name}_extracted.json" # Adjust name pattern if needed
                # Actually, the file pattern is: {PAPER}_v5_extracted.json
                # Wait, the validation report name is {PAPER}_v5_extracted_validation_report.json
                # So extraction is {PAPER}_v5_extracted.json
                extracted_file = results_path / f"{paper_name}_v5_extracted.json"
                
                with open(extracted_file) as f: 
                    ext_data = json.load(f)
                    extracted_measurements = ext_data.get("measurements", [])
            except Exception as e:
                print(f"Skipping {paper_name}: {e}")
                continue

            # Load PDF
            pdf_path = Path(PDF_DIR) / f"{paper_name}.pdf"
            if not pdf_path.exists():
                print(f"PDF not found for: {paper_name}")
                continue

            # Find Misses
            # We need the Original GT Data. The report has "GT-ID".
            # We must map "GT-ID" (index) back to the CSV row.
            
            for match in report.get("matches", []):
                if not match['is_match']:
                    gt_id_str = match['ground_truth_id'] # "GT-282"
                    try:
                        gt_idx = int(gt_id_str.replace("GT-", ""))
                        gt_row = self.gt_df.iloc[gt_idx].to_dict()
                        
                        failures.append({
                            "paper": paper_name,
                            "gt": gt_row,
                            "extracted": extracted_measurements,
                            "pdf": pdf_path
                        })
                    except:
                        pass

        print(f"Found {len(failures)} missing data points across the test set.")
        
        # 2. Analyze Failures (Batch limit to save tokens/time for this demo)
        diagnosis_results = []
        for fail in failures[:5]: # Limit to 5 for now
            diagnosis = await self.analyze_failure(fail['paper'], fail['gt'], fail['extracted'], fail['pdf'])
            if diagnosis:
                diagnosis_results.append(diagnosis)

        # 3. Aggregate Improvements
        suggestions = [d['suggestion'] for d in diagnosis_results if d.get('suggestion')]
        unique_suggestions = list(set(suggestions))
        
        print("\n💡 Suggested Improvements:")
        for s in unique_suggestions:
            print(f"- {s}")
            
        if not unique_suggestions:
            print("No improvements generated.")
            return

        # 4. Rewrite Code
        print("\n✍️  Rewriting Extraction Script...")
        current_code = Path(TARGET_SCRIPT).read_text()
        
        prompt = FRAMEWORK_IMPROVER_PROMPT.format(improvements_list="\n".join(f"- {s}" for s in unique_suggestions))
        
        # We pass the prompt + current code (context)
        # Using Gemini 3 Pro Preview as requested for code gen (though 3-flash is also good, prompt says 3-pro)
        try:
            response = await self.client.aio.models.generate_content(
                model="gemini-3-pro-preview", 
                contents=[prompt, "EXISTING CODE:\n" + current_code],
                config=types.GenerateContentConfig(
                    response_mime_type="text/x-python",
                    temperature=0.2
                )
            )
            
            new_code = response.text
            
            # Simple check
            if "def main" in new_code:
                Path(TARGET_SCRIPT).write_text(new_code)
                print(f"✅ Successfully updated {TARGET_SCRIPT}")
            else:
                print("❌ Generated code looks invalid (no main). Aborting write.")
                
        except Exception as e:
            print(f"❌ Code rewrite failed: {e}")

async def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Missing GEMINI_API_KEY")
        return
        
    optimizer = ExtractionOptimizer(api_key)
    await optimizer.run()

if __name__ == "__main__":
    asyncio.run(main())
