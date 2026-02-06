## In this version we added separate processing for figures
import os
import argparse
import base64
import json
import time
import asyncio
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# ==============================================================================
# 1. Data Schema
# ==============================================================================
class SimpleDataPoint(BaseModel):
    composition: str = Field(..., description="The chemical formula (e.g., Li6PS5Cl).")
    conductivity: str = Field(..., description="Value (e.g., 1.2e-3).")
    unit: str = Field(..., description="Unit (e.g., S/cm).")
    temp: str = Field(..., description="Temperature (e.g., 25 C, RT).")
    source: str = Field(..., description="Where this specific point came from (e.g. 'Table 1', 'Figure 3b', 'Main text, introduction').")
    confidence: str = Field(..., description="high (explicit text/table), medium (plot reading), low (inferred).")

class ExtractionResult(BaseModel):
    measurements: List[SimpleDataPoint]

# ==============================================================================
# 2. Prompts
# ==============================================================================
TEXT_PROMPT = """
Analyze the provided scientific paper text. 
Extract ALL ionic conductivity measurements found in **text tables, paragraphs, and figure captions**.

Ignore actual image data (plots); only extract what is explicitly written in the text.
If a value is "Not Specified", do not extract it.
"""

VISION_PROMPT = """
Analyze this specific scientific image.
If this is a **Data Plot** (e.g., Arrhenius, Conductivity vs X):
1. Identify the X and Y axes labels and units.
2. Extract EVERY visible data point.
3. If multiple series exist (different materials), extract all of them.
4. Estimate values as precisely as possible.

If this is a **Table** image:
1. Extract all rows containing conductivity data.

If this is an SEM/XRD/Schematic:
1. Return an empty list.
"""

# ==============================================================================
# 3. Async Helpers for Gemini 3
# ==============================================================================
async def process_text(client, model, text_content):
    print(f"   ... Processing Text ({len(text_content)} chars)...")
    try:
        # GEMINI 3 SPECIFIC: Temperature 1.0 is recommended for reasoning models.
        # We use thinking_level="low" here because text extraction is logically simple 
        # and we want to save latency/tokens.
        response = await client.aio.models.generate_content(
            model=model,
            contents=[TEXT_PROMPT, text_content],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=ExtractionResult.model_json_schema(),
                temperature=1.0, 
                thinking_config=types.ThinkingConfig(thinking_level="medium") 
            )
        )
        return ExtractionResult.model_validate_json(response.text).measurements
    except Exception as e:
        print(f"   [Text Error]: {e}")
        return []

async def process_image(client, model, img_path):
    # Skip small icons/logos to save cost/time
    if "logo" in img_path.name.lower() or img_path.stat().st_size < 1000: 
        return []
    
    try:
        img_bytes = img_path.read_bytes()
        prompt_with_context = f"{VISION_PROMPT}\n\nContext: This image is named '{img_path.name}'."
        
        # GEMINI 3 SPECIFIC: 
        # 1. media_resolution="media_resolution_high" (1120 tokens) is CRITICAL for plots.
        # 2. temperature=1.0 (Required).
        # 3. thinking_level defaults to "high" (implicit), which is what we want for complex plot reading.
        
        # Create the Content payload with the new resolution parameter
        content = types.Content(
            parts=[
                types.Part(text=prompt_with_context),
                types.Part(
                    inline_data=types.Blob(
                        mime_type="image/png" if img_path.suffix.lower() == '.png' else "image/jpeg",
                        data=base64.b64encode(img_bytes).decode('utf-8')
                    ),
                    # This tells Gemini 3 to use maximum token budget for this image
                    media_resolution={"level": "media_resolution_high"} 
                )
            ]
        )

        response = await client.aio.models.generate_content(
            model=model,
            contents=[content],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=ExtractionResult.model_json_schema(),
                temperature=1.0, 
                # We do NOT set thinking_level here, allowing it to default to 'high' 
                # for maximum reasoning depth on the chart.
            )
        )
        
        res = ExtractionResult.model_validate_json(response.text).measurements
        if res:
            print(f"   ✓ {img_path.name}: Found {len(res)} points")
        return res
    except Exception as e:
        print(f"   [Image Error] {img_path.name}: {e}")
        return []

# ==============================================================================
# 4. Main Logic
# ==============================================================================
async def run_pipeline(markdown_file, asset_dir, model):
    api_key = os.getenv("GEMINI_API_KEY")
    # GEMINI 3 SPECIFIC: Must use v1alpha for media_resolution
    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})
    
    # 1. Read Text
    try:
        text_content = markdown_file.read_text(encoding='utf-8')
    except Exception as e:
        print(f"Error reading markdown file: {e}")
        return []
    
    # 2. Find Images
    image_files = sorted(list(asset_dir.glob("*.png")) + list(asset_dir.glob("*.jpg")) + list(asset_dir.glob("*.jpeg")))
    print(f"--- Starting Gemini 3 Extraction: {markdown_file.name} ---")
    print(f"Model: {model} | Images: {len(image_files)}")

    # 3. Create Tasks
    # Task A: Process Text (Low Thinking, Fast)
    text_task = process_text(client, model, text_content)
    
    # Task B: Process Images (High Resolution, High Thinking, Parallel)
    # Semaphore controls concurrency. Gemini 3 limits might require tuning this.
    sem = asyncio.Semaphore(5) 
    
    async def safe_image_call(img):
        async with sem:
            return await process_image(client, model, img)

    image_tasks = [safe_image_call(img) for img in image_files]

    # 4. Run Everything
    results = await asyncio.gather(text_task, *image_tasks)
    
    # 5. Flatten Results
    all_measurements = []
    for res_list in results:
        all_measurements.extend(res_list)

    return all_measurements

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('markdown_file', type=Path, help="Path to the markdown file")
    parser.add_argument('--asset_dir', type=Path, help="Path to assets (default: same as md)")
    # Defaulting to Gemini 3 Flash as requested
    parser.add_argument('--model', default='gemini-3-flash-preview', help="Gemini 3 model string") 
    args = parser.parse_args()

    # Fallback logic if user provides Gemini 3 string manually or relies on default
    # Note: Ensure the model string matches the actual deployed name (e.g. gemini-3-flash-preview)
    # when it becomes fully available in your region.
    
    start = time.time()
    
    # Run Async Loop
    measurements = asyncio.run(run_pipeline(
        args.markdown_file, 
        args.asset_dir or args.markdown_file.parent, 
        args.model
    ))
    
    # Save
    output = {
        "metadata": {
            "model": args.model,
            "timestamp": time.time(),
            "duration_seconds": time.time() - start, 
            "total_points": len(measurements)
        },
        "data": [m.model_dump() for m in measurements]
    }
    
    out_path = args.markdown_file.parent / "basic_extraction_md_v2_results.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nDone! Extracted {len(measurements)} points.")
    print(f"Saved to: {out_path}")

if __name__ == "__main__":
    main()