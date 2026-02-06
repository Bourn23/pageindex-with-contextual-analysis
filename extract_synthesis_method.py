import os
import argparse
import base64
import time
import pandas as pd
from pathlib import Path
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# ==============================================================================
# The Prompt
# ==============================================================================
SYNTHESIS_PROMPT_TEMPLATE = """
You are an expert Materials Scientist.
Read the attached paper text (and figures if available).

**Task:**
Write a summary of the **Processing / Synthesis Method** used to create the materials listed below. 

**Context (Target Materials):**
The user is specifically interested in the method used for:
{target_materials_list}

**Output Requirements:**
Provide a plain text response with the following sections:

1. **Method Name:** (e.g., Solid State Reaction, Sol-Gel, etc.)
2. **Procedure Summary:** A concise paragraph describing the steps (mixing, precursors, shaping, etc.).
3. **Key Conditions:** List the specific temperatures, dwell times, and atmospheres used (e.g., "Sintered at 800°C for 12 hours in Argon").

Do not format this as JSON. Just write clear, human-readable text.
"""

def main():
    parser = argparse.ArgumentParser(description='Plain Text Synthesis Extraction')
    parser.add_argument('markdown_file', type=Path, help='Path to the full markdown file')
    parser.add_argument('--ground-truth', '-gt', type=Path, help='Path to Ground Truth CSV (optional context)')
    parser.add_argument('--doi', help='DOI to filter the GT CSV')
    parser.add_argument('--asset_dir', type=Path, help='Directory containing images referenced in markdown')
    parser.add_argument('--model', default='gemini-2.5-flash', help='Model to use')
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found.")
        return

    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})

    print(f"--- Starting Plain Text Synthesis Extraction ---")
    
    # 1. Prepare Target Material Context from GT
    # If no GT is provided, we use a generic catch-all.
    target_materials_str = "The main solid electrolyte materials discussed in this paper."
    
    if args.ground_truth:
        try:
            df = pd.read_csv(args.ground_truth)
            if args.doi and 'DOI' in df.columns:
                df = df[df['DOI'] == args.doi]
            
            if 'Composition' in df.columns:
                compositions = df['Composition'].unique().tolist()
                if compositions:
                    target_materials_str = ", ".join(map(str, compositions))
                    print(f"Context loaded: Focused on {len(compositions)} materials.")
        except Exception as e:
            print(f"Warning: Could not load GT context ({e}). Using generic prompt.")

    formatted_prompt = SYNTHESIS_PROMPT_TEMPLATE.format(target_materials_list=target_materials_str)

    # 2. Read Markdown
    try:
        text_content = args.markdown_file.read_text(encoding='utf-8')
    except Exception as e:
        print(f"Error reading markdown: {e}")
        return

    # 3. Load Images (Optional)
    media_parts = []
    if args.asset_dir and args.asset_dir.exists():
        image_files = list(args.asset_dir.glob("*.png")) + \
                      list(args.asset_dir.glob("*.jpg")) + \
                      list(args.asset_dir.glob("*.jpeg"))
        
        # Limit to 10 images to stay lightweight
        for img_path in image_files[:10]:
            try:
                img_bytes = img_path.read_bytes()
                mime_type = "image/png" if img_path.suffix.lower() == '.png' else "image/jpeg"
                media_parts.append(
                    types.Part(
                        inline_data=types.Blob(
                            mime_type=mime_type, 
                            data=base64.b64encode(img_bytes).decode('utf-8'),
                        )
                    )
                )
            except Exception:
                pass

    # 4. Construct Payload
    contents = [
        types.Content(
            parts=[
                types.Part(text=formatted_prompt),
                types.Part(text=f"--- START OF PAPER TEXT ---\n{text_content}\n--- END OF PAPER TEXT ---")
            ] + media_parts
        )
    ]

    # 5. API Call (Standard Text Generation)
    print(f"Sending request to {args.model}...")
    try:
        response = client.models.generate_content(
            model=args.model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.2  # Low temperature to prevent hallucinations
            )
        )
        
        if response.text:
            print("\n" + "="*60)
            print("EXTRACTION RESULT:")
            print("="*60)
            print(response.text)
            print("="*60)
            
            # Save to a text file next to the markdown
            output_path = args.markdown_file.parent / "synthesis_method.txt"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(response.text)
            print(f"Saved to {output_path}")

    except Exception as e:
        print(f"LLM Call Failed: {e}")

if __name__ == "__main__":
    main()