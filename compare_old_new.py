import os
import json
import asyncio
from pathlib import Path
from google import genai
from dotenv import load_dotenv

load_dotenv()

TARGET_DIR = Path("/Users/bourn23/Downloads/general/PageIndex/output/deb_downloaded_papers")
PARSED_DIR = Path("/Users/bourn23/Downloads/general/PageIndex/output_parsed/deb_downloaded_papers")

EVALUATION_MODEL = "gemini-3-pro-preview"

async def evaluate_old_vs_new(client, paper_name, old_data, new_data):
    prompt = f"""
I am trying to extract processing methods (synthesis, preparation, sintering, coating, parameters, etc.) from material science papers.

I have two approaches:
1. **Old Approach:** Performed extraction on a per-measurement basis. It extracts short `processing_method` strings and `material_definitions` text snippets for each ionic conductivity measurement found in the paper.
2. **New Approach:** Passes the entire document to the LLM and asks it to synthesize a comprehensive list of all processing methods, materials, and parameters found in the text.

Here is the data for the paper "{paper_name}":

--- OLD APPROACH EXTRACTIONS (aggregated from all measurements in the paper) ---
{old_data}

--- NEW APPROACH EXTRACTION (Whole Document Summary) ---
{new_data}

---
Please evaluate these two approaches. 
My goal is to decide whether I should:
A) Keep the Old Approach extractions as they are (since they are tied directly to specific measurements) and just run the New Approach to fill in the missing gaps for certain measurements.
OR
B) Completely discard the Old Approach extractions for processing methods and re-extract/re-map everything using the comprehensive list from the New Approach, because the old extractions are too sparse, inaccurate, or scattered.

Compare them based on:
1. Completeness / Detail of the processing methods and parameters.
2. Usefulness for a materials science database.
3. Your recommendation (Option A or Option B) with a brief justification.
"""
    try:
        response = await client.aio.models.generate_content(
            model=EVALUATION_MODEL,
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"Error: {e}"

async def process_paper(client, paper_dir):
    paper_name = paper_dir.name
    old_json_path = PARSED_DIR / f"{paper_name}_v5_extracted.json"
    new_json_path = paper_dir / "processing_extraction_comparison.json"

    if not old_json_path.exists() or not new_json_path.exists():
        return

    # Extract Old Data
    with open(old_json_path, 'r', encoding='utf-8') as f:
        old_json = json.load(f)
    
    old_extractions = []
    for m in old_json.get("measurements", []):
        comp = m.get("raw_composition", "Unknown Material")
        pm = m.get("processing_method", "None")
        defs = m.get("material_definitions", [])
        old_extractions.append(f"Material: {comp}\nProcessing Method: {pm}\nDefinitions: {defs}\n")
    old_data_str = "\n".join(old_extractions)

    # Extract New Data
    with open(new_json_path, 'r', encoding='utf-8') as f:
        new_json = json.load(f)
    new_data_str = new_json.get("method1", "No method 1 extraction.")

    print(f"--- Evaluating {paper_name} ---")
    eval_result = await evaluate_old_vs_new(client, paper_name, old_data_str, new_data_str)
    print(eval_result)
    print("\n======================================================\n")

async def main():
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"), http_options={'api_version': 'v1alpha'})
    
    # Evaluate 2 sample papers
    count = 0
    for paper_dir in TARGET_DIR.iterdir():
        if paper_dir.is_dir():
            await process_paper(client, paper_dir)
            count += 1
            if count >= 3:
                break

if __name__ == "__main__":
    asyncio.run(main())
