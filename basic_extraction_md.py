import os
import argparse
import base64
import json
import time
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==============================================================================
# 1. The Simplest Possible Schema
# ==============================================================================
class SimpleDataPoint(BaseModel):
    composition: str = Field(
        ..., 
        description="The chemical formula of the material (e.g., Li6.0Hf12.0P18.0O72.0)."
    )
    ionic_conductivity_value: str = Field(
        ..., 
        description="The conductivity value (preserve scientific notation if present, e.g., 6.85e-07)."
    )
    ionic_conductivity_unit: str = Field(
        ...,
        description="The unit of conductivity (e.g., S/cm, mS/cm)."
    )
    temperature: str = Field(
        ..., 
        description="The temperature of measurement (e.g., 25 C, RT, 300 K)."
    )
    source: str = Field(
        ...,
        description="Where the data was found (e.g., 'Figure 6a', 'Table 2', 'conclusion section')."
    )

class BenchmarkResult(BaseModel):
    measurements: List[SimpleDataPoint]

# ==============================================================================
# 2. The Simplest Possible Prompt
# ==============================================================================
# BASELINE_PROMPT = """
# You are a scientific data extractor. 
# Analyze the provided full text and attached figures of a research paper.
# Extract a list of ALL material compositions and their corresponding ionic conductivity values.
# """

BASELINE_PROMPT = """
You are a scientific data extractor specializing in solid-state electrolyte materials.

Your task: Extract ALL ionic conductivity measurements from this research paper, including data from:
1. Text (tables, prose, figure captions)
2. **Figures and plots** (read data points from graphs)
3. Supplementary information references

For each data point, extract:
- **Composition**: Full chemical formula with exact stoichiometry (e.g., Li6.25P0.875Si0.125S5Br)
- **Conductivity**: Numerical value with units (convert to S/cm if needed: 1 mS/cm = 0.001 S/cm)
- **Temperature**: Measurement temperature (e.g., RT, 25°C, 300K)
- **Source**: Where you found this data (e.g., "Figure 6a", "Table 2", "main text page 5")

**Critical instructions:**
- Extract EVERY data point from plots/graphs, not just values mentioned in text
- For conductivity plots (y-axis) vs composition (x-axis), read all visible data points
- Pay attention to figure captions that describe what data is shown
- If a range of compositions is studied (e.g., Li6+xP1-xSixS5Br where x varies), extract each x value
- Include error bars or uncertainties if provided
- If units are given as mS/cm, convert to S/cm by dividing by 1000

**Output format:**
Return a structured list where each entry contains:
{
  "composition": "exact formula",
  "ionic_conductivity_value": numerical value,
  "ionic_conductivity_unit": "S/cm",
  "temperature": "RT" or specific value,
  "source": "where found"
}

Extract ALL available data points - a typical study may have 10-20+ measurements across a composition range.
"""

def main():
    parser = argparse.ArgumentParser(description='Baseline Benchmark: Full-Context Extraction')
    parser.add_argument('markdown_file', type=Path, help='Path to the full markdown file')
    parser.add_argument('--asset_dir', type=Path, help='Directory containing images referenced in markdown')
    parser.add_argument('--model', default='gemini-3-flash-preview', help='Model to use (recommend high context model)')
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found.")
        return
    print("USING EMINI API KEY", api_key)

    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})

    print(f"--- Starting Baseline Benchmark ---")
    print(f"Input: {args.markdown_file}")
    start_time = time.time()

    # 1. Read Markdown Text
    try:
        text_content = args.markdown_file.read_text(encoding='utf-8')
    except Exception as e:
        print(f"Error reading markdown: {e}")
        return

    # 2. Load ALL Images in Asset Directory
    # In the naive approach, we dump every image found into the context 
    # and let the model figure out what matches what.
    media_parts = []
    asset_dir = args.markdown_file.parent
    if asset_dir.exists():
        image_files = list(asset_dir.glob("*.png")) + \
                      list(asset_dir.glob("*.jpg")) + \
                      list(asset_dir.glob("*.jpeg"))
        
        print(f"Attaching {len(image_files)} images found in asset dir...")
        
        for img_path in image_files:
            try:
                img_bytes = img_path.read_bytes()
                mime_type = "image/png" if img_path.suffix.lower() == '.png' else "image/jpeg"
                
                media_parts.append(
                    types.Part(
                        inline_data=types.Blob(
                            mime_type=mime_type, 
                            data=base64.b64encode(img_bytes).decode('utf-8'),
                        ),
                        media_resolution={"level": "media_resolution_high"}
                    )
                )
            except Exception as e:
                print(f"Skipping image {img_path.name}: {e}")

    # 3. Construct Payload
    contents = [
        types.Content(
            parts=[
                types.Part(text=BASELINE_PROMPT),
                types.Part(text=f"--- START OF PAPER TEXT ---\n{text_content}\n--- END OF PAPER TEXT ---")
            ] + media_parts
        )
    ]

    # 4. API Call
    print(f"Sending request to {args.model}...")
    try:
        response = client.models.generate_content(
            model=args.model,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=BenchmarkResult.model_json_schema(),
                temperature=1.0
            )
        )
        
        # 5. Output Handling
        if response.text:

            print(f"Response Usage Metadata: {response.usage_metadata}")
            result = BenchmarkResult.model_validate_json(response.text)
            elapsed = time.time() - start_time
            
            output_data = {
                "metadata": {
                    "model": args.model,
                    "time_elapsed_seconds": elapsed,
                    "images_attached": len(media_parts),
                    "input_token_count": response.usage_metadata.prompt_token_count,
                    "output_token_count": response.usage_metadata.candidates_token_count
                },
                "data": [m.model_dump() for m in result.measurements]
            }

            with open(args.markdown_file.parent / "basic_extraction_results.json", 'w') as f:
                json.dump(output_data, f, indent=2)

            print(f"Success! Extracted {len(result.measurements)} points in {elapsed:.2f}s.")
            print(f"Results saved to {args.markdown_file.parent / 'basic_extraction_results.json'}")
            
            # Print a preview to console to compare with your GT
            print("\nExtracted Points:")
            # with open("extracted_points.txt", "w") as f:
            # so we will use the parent directory of the markdown file
            with open(args.markdown_file.parent / "extracted_points.txt", "w") as f:
                print(f"{'Composition':<40} | {'Conductivity':<20} | {'Temp'}")
                print("-" * 70)
                for item in result.measurements:
                    print(f"{item.composition:<40} | {item.ionic_conductivity_value:<20} | {item.temperature}")
                    f.write(f"{item.composition:<40} | {item.ionic_conductivity_value:<20} | {item.temperature}\n")

    except Exception as e:
        print(f"LLM Call Failed: {e}")

if __name__ == "__main__":
    main()