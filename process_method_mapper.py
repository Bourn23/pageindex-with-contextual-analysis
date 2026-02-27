"""
Process Method Mapper: Two-step pipeline to enrich measurements with processing methods.

Step 1: Reuse comprehensive whole-document extraction (from processing_extraction_comparison.json)
        or re-extract using gemini-2.5-flash if not available.
Step 2: Map comprehensive methods to individual measurements using gemini-3-flash,
        appending to (not replacing) existing context.
"""

import os
import json
import asyncio
import argparse
from pathlib import Path
from google import genai
from dotenv import load_dotenv

load_dotenv()

# Configuration
EXTRACTION_MODEL = "gemini-2.5-flash"
MAPPING_MODEL = "gemini-3-flash-preview"
CONCURRENCY_LIMIT = 4
BATCH_SIZE = 20  # measurements per LLM call

# Directories
PARENT_FOLDER = "wes_downloaded_papers"
PAPERS_DIR = Path("/Users/bourn23/Downloads/general/PageIndex/output/" + PARENT_FOLDER)
PARSED_DIR = Path("/Users/bourn23/Downloads/general/PageIndex/output_parsed/" + PARENT_FOLDER)


async def get_gemini_response(client, model_name, prompt):
    """Call the Gemini API."""
    try:
        response = await client.aio.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        return response.text
    except Exception as e:
        print(f"  Error calling {model_name}: {e}")
        return None


async def extract_processing_methods_full(client, md_content):
    """Step 1: Extract comprehensive processing methods from the whole document."""
    prompt = f"""
Extract all processing methods used to synthesize or prepare the samples/materials described in the following research paper.
List them concisely, including key parameters like temperatures, times, solvents, if mentioned.
Group them by material or method type when possible.

Paper Content:
{md_content}

Extracted Processing Methods:
"""
    return await get_gemini_response(client, EXTRACTION_MODEL, prompt)


async def get_comprehensive_methods(client, paper_dir):
    """Get the comprehensive methods list, preferring cached results."""
    # 1. Check for standard comparison file
    comparison_path = paper_dir / "processing_extraction_comparison.json"
    if comparison_path.exists():
        try:
            with open(comparison_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            method1 = data.get("method1")
            if method1 and method1 != "null" and len(method1) > 50:
                print(f"  Found cached 'method1' in {comparison_path.name}")
                return method1
        except Exception as e:
            print(f"  Error reading {comparison_path.name}: {e}")

    # 2. Search for ANY JSON file that might contain 'method1' or looks like an extraction result
    for json_file in paper_dir.glob("*.json"):
        if json_file.name == "processing_extraction_comparison.json":
            continue
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Check for common extraction keys
            for key in ["method1", "processing_methods", "comprehensive_methods"]:
                val = data.get(key)
                if val and isinstance(val, str) and len(val) > 50:
                    print(f"  Found cached methods in {json_file.name} (key: {key})")
                    return val
        except Exception:
            continue

    # 3. Fallback: Re-extract if not available
    print(f"  No cached extraction found. Re-extracting for {paper_dir.name}...")
    md_files = list(paper_dir.glob("*.md"))
    if not md_files:
        print(f"  Warning: No MD file found in {paper_dir.name}")
        return None
    with open(md_files[0], 'r', encoding='utf-8') as f:
        md_content = f.read()
    return await extract_processing_methods_full(client, md_content)


async def map_methods_to_batch(client, sem, paper_name, comprehensive_methods, batch, batch_idx):
    """Step 2: Map comprehensive methods to a batch of measurements."""
    async with sem:
        # Build the batch description
        measurements_desc = []
        for i, m in enumerate(batch):
            old_pm = m.get("processing_method") or "None"
            old_defs = m.get("material_definitions", [])
            old_defs_str = "; ".join(old_defs) if old_defs else "None"
            comp = m.get("raw_composition", "Unknown")
            formula = m.get("canonical_formula", "Unknown")
            source = m.get("source", "Unknown")

            measurements_desc.append(
                f"[{i}] Material: {comp} | Formula: {formula} | Source: {source}\n"
                f"    Old processing_method: {old_pm}\n"
                f"    Old material_definitions: {old_defs_str}"
            )

        measurements_block = "\n".join(measurements_desc)

        prompt = f"""
You are a materials science expert. I need you to assign the correct processing method to each measurement.

I have a comprehensive list of ALL processing methods found in the paper "{paper_name}":

--- COMPREHENSIVE METHODS LIST ---
{comprehensive_methods}
--- END OF METHODS LIST ---

Below are {len(batch)} measurements extracted from this paper. Each has an existing (possibly empty) `processing_method` and `material_definitions` from a previous extraction.

Your task:
1. For each measurement, find the BEST matching processing method from the comprehensive list above.
2. MERGE the old context with the new information — do NOT discard the old processing_method or material_definitions if they contain useful info.
3. If the old processing_method is "None" or missing, fill it in from the comprehensive list.
4. If the old processing_method already has useful info, APPEND any additional detail from the comprehensive list.
5. If a material's processing method truly cannot be determined from the comprehensive list, set it to "not reported".

--- MEASUREMENTS ---
{measurements_block}
--- END ---

Return your answer as a valid JSON array of objects, one per measurement, in order. Each object should have:
- "index": the measurement index [0, 1, 2, ...]
- "processing_method": the enriched processing method string (concise but detailed)
- "processing_method_detail": a longer description with parameters if available

Return ONLY the JSON array, no markdown code fences, no explanation.
"""
        result = await get_gemini_response(client, MAPPING_MODEL, prompt)
        if not result:
            return None

        # Parse the JSON response
        try:
            # Strip markdown fences if present
            cleaned = result.strip()
            if cleaned.startswith("```"):
                # Handle possible language identifier like ```json
                lines = cleaned.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].strip() == "```":
                    lines = lines[:-1]
                cleaned = "\n".join(lines).strip()
            parsed = json.loads(cleaned)
            return parsed
        except json.JSONDecodeError as e:
            print(f"  Batch {batch_idx}: JSON parse error: {e}")
            # print(f"  Raw response (first 300 chars): {result[:300]}")
            return None


async def process_paper(client, paper_name, sem, force=False):
    """Process a single paper: get methods, map to measurements, save."""
    paper_dir = PAPERS_DIR / paper_name
    output_path = PARSED_DIR / f"{paper_name}_v5_extracted_mapped.json"

    # Avoid re-processing if exists unless forced
    if output_path.exists() and not force:
        print(f"  Skipping {paper_name}: already processed ({output_path.name})")
        return 0

    # Find corresponding old JSON
    old_json_path = PARSED_DIR / f"{paper_name}_v5_extracted.json"
    if not old_json_path.exists():
        print(f"  Skipping {paper_name}: no old JSON found at {old_json_path}")
        return 0

    # Load old data
    with open(old_json_path, 'r', encoding='utf-8') as f:
        old_data = json.load(f)

    measurements = old_data.get("measurements", [])
    if not measurements:
        print(f"  Skipping {paper_name}: no measurements in JSON")
        return 0

    # Step 1: Get comprehensive methods (with avoidance of re-extraction)
    comprehensive_methods = await get_comprehensive_methods(client, paper_dir)
    if not comprehensive_methods:
        print(f"  Skipping {paper_name}: could not get comprehensive methods context")
        return 0

    # Step 2: Map in batches
    batches = [measurements[i:i + BATCH_SIZE] for i in range(0, len(measurements), BATCH_SIZE)]
    print(f"  {len(measurements)} measurements in {len(batches)} batches")

    all_mappings = {}
    tasks = []
    for batch_idx, batch in enumerate(batches):
        tasks.append(map_methods_to_batch(client, sem, paper_name, comprehensive_methods, batch, batch_idx))

    results = await asyncio.gather(*tasks)

    # Collect results
    offset = 0
    for batch_idx, (batch, result) in enumerate(zip(batches, results)):
        if result:
            for item in result:
                idx = item.get("index", 0) + offset
                all_mappings[idx] = item
        else:
            print(f"  Batch {batch_idx} failed for {paper_name}")
        offset += len(batch)

    # Apply mappings to measurements
    enriched_count = 0
    for i, m in enumerate(measurements):
        if i in all_mappings:
            mapping = all_mappings[i]
            new_pm = mapping.get("processing_method", "")
            new_detail = mapping.get("processing_method_detail", "")

            old_pm = m.get("processing_method")

            # Append logic: merge old and new
            if old_pm and old_pm.lower() not in ("none", "null", ""):
                if new_pm and new_pm.lower() not in ("none", "null", "not reported", ""):
                    # If new info is substantially different, append
                    if new_pm.lower() != old_pm.lower():
                        m["processing_method"] = f"{old_pm}; {new_pm}"
                    else:
                        m["processing_method"] = old_pm
                # else keep old
            else:
                m["processing_method"] = new_pm if new_pm else "not reported"

            # Add detail as a new field
            if new_detail:
                m["processing_method_detail"] = new_detail

            enriched_count += 1

    print(f"  Enriched {enriched_count}/{len(measurements)} measurements")

    # Save to new file (non-destructive)
    old_data["measurements"] = measurements
    old_data["mapping_metadata"] = {
        "comprehensive_methods_source": "cached or whole-document extraction",
        "mapping_model": MAPPING_MODEL,
        "total_measurements": len(measurements),
        "enriched_count": enriched_count,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(old_data, f, indent=2, ensure_ascii=False)

    print(f"  Saved: {output_path.name}")
    return enriched_count


async def main():
    parser = argparse.ArgumentParser(description="Map processing methods to measurements")
    parser.add_argument("--sample", help="Process only this paper name")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of papers (0 = all)")
    parser.add_argument("--force", action="store_true", help="Re-process even if output exists")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: Please set GEMINI_API_KEY or GOOGLE_API_KEY environment variable.")
        return

    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)

    print(f"PAPERS_DIR: {PAPERS_DIR}")
    print(f"PARSED_DIR: {PARSED_DIR}")

    if not PAPERS_DIR.exists():
        print(f"Error: PAPERS_DIR does not exist: {PAPERS_DIR}")
        return

    # Find papers that have both dirs
    if args.sample:
        paper_names = [args.sample]
    else:
        paper_names = []
        for d in PAPERS_DIR.iterdir():
            if d.is_dir():
                old_json = PARSED_DIR / f"{d.name}_v5_extracted.json"
                if old_json.exists():
                    paper_names.append(d.name)
                else:
                    # Optional: print for debugging
                    # print(f"  Note: Missing old JSON for {d.name}")
                    pass
        paper_names.sort()

    if not paper_names:
        print("No papers found that have both an output directory and an extracted JSON.")
        return

    print(f"Processing {len(paper_names)} papers...")
    total_enriched = 0
    for name in paper_names:
        print(f"\n--- {name} ---")
        result = await process_paper(client, name, sem, force=args.force)
        if result:
            total_enriched += result

    print(f"\n=== Done. Total enriched: {total_enriched} ===")


if __name__ == "__main__":
    asyncio.run(main())
