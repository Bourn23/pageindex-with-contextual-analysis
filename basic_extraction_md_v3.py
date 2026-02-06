## In addition to splitting the processing of text and images
# we also add context to the images, add post processing to the units and resolve the material names
import os
import re
import argparse
import base64
import json
import time
import asyncio
import math
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv
import uuid
from dataclasses import dataclass

load_dotenv()

# ==============================================================================
# 1. Data Schema (Enhanced)
# ==============================================================================
@dataclass
class SectionInfo:
    title: str
    content: str
    line_num: int
    end_line_num: int
    id: str


class MeasuredPoint(BaseModel):
    # We capture the "Raw" string for provenance, and "Normalized" for database
    raw_composition: str = Field(..., description="The name as it appears in the source (e.g. 'x=0.1', 'Sample A').")
    
    # We will fill these in via Post-Processing
    canonical_formula: Optional[str] = Field(None, description="Normalized chemical formula (e.g. Li3.8Mg0.1Ti1.63O4).")
    
    raw_conductivity: str = Field(..., description="Value as extracted (e.g. '1.24e-4', '5.2').")
    raw_unit: str = Field(..., description="Unit as extracted (e.g. 'mS/cm', 'S cm-1').")
    normalized_conductivity: Optional[float] = Field(None, description="Value in S/cm.")
    
    raw_temperature: str = Field(..., description="Temperature as extracted (e.g. '1000/T = 3.2', '300 K').")
    normalized_temperature_c: Optional[float] = Field(None, description="Temperature in Celsius.")
    
    source_figure_id: Optional[str] = Field(None, description="The real Figure ID (e.g. 'Fig. 5') if known.")
    source_caption: Optional[str] = Field(None, description="The context from the figure caption.")
    confidence: str = Field(..., description="high/medium/low")

class ExtractionResult(BaseModel):
    measurements: List[MeasuredPoint]
    # We also extract "Material Definitions" from text to help us resolve "x=0.1" later
    material_definitions: List[str] = Field(
        default_factory=list, 
        description="Text snippets that define material series (e.g. 'solid solutions Li(4-2x)MgxTi(5-x)/3O4')."
    )

# ==============================================================================
# 2. Context Parsing (Solves Problem 1: Context Loss)
# ==============================================================================
class MarkdownContextParser:
    """
    Parses Markdown to extract both document structure (Sections/Headers)
    and context for images (Figure Captions).
    """

    def parse_structure(self, md_text: str) -> Tuple[str, List[SectionInfo]]:
        """
        Robustly parses markdown into a Document Title and a list of Sections.
        """
        lines = md_text.split('\n')
        
        headers = []
        in_code_block = False
        
        # Regex handles: ## Title, ## **Title**, ## Title ##
        header_pattern = re.compile(r'^(#{1,6})\s+(.*?)(?:\s+#+)?$')

        # Pass 1: Scan for headers and code blocks
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Toggle code block state
            if stripped.startswith('```'):
                in_code_block = not in_code_block
                continue
            
            # Skip headers if inside a code block
            if in_code_block:
                continue
            
            match = header_pattern.match(stripped)
            if match:
                title_text = match.group(2).strip()
                headers.append({
                    'level': len(match.group(1)),
                    'title': title_text,
                    'line_num': i
                })

        # Default values
        doc_title = "Untitled Document"
        sections: List[SectionInfo] = []
        
        # Handle Case: No headers
        if not headers:
            if not md_text.strip():
                return doc_title, []
            return doc_title, [SectionInfo(title="Full Text", content=md_text, line_num=0, end_line_num=len(lines), id=str(uuid.uuid4()))]
            
        # Pass 2: Determine Document Title (Prioritize H1)
        h1_headers = [h for h in headers if h['level'] == 1]
        doc_title = h1_headers[0]['title'] if h1_headers else headers[0]['title']
        
        # Slicing Logic
        slice_points = [h['line_num'] for h in headers] + [len(lines)]
        
        # 1. Handle Pre-Header Content (Introduction)
        if headers[0]['line_num'] > 0:
            pre_text = "\n".join(lines[0:headers[0]['line_num']]).strip()
            if pre_text and re.search(r'[a-zA-Z0-9]', pre_text):
                sections.append(SectionInfo(
                    title="Introduction", 
                    content=pre_text, 
                    line_num=0,
                    end_line_num=headers[0]['line_num'],
                    id=str(uuid.uuid4())
                ))

        # 2. Iterate through headers to build sections
        for i, header in enumerate(headers):
            start_line = header['line_num'] + 1
            end_line = slice_points[i+1]
            
            content_lines = lines[start_line:end_line]
            content_text = "\n".join(content_lines).strip()

            # Use current header title unless it's the Doc title then call it abstract
            sec_title = "Abstract / Introduction" if (header['level'] == 1 and header['title'] == doc_title) else header['title']

            if content_text: 
                sections.append(SectionInfo(
                    title=sec_title,
                    content=content_text,
                    line_num=start_line,
                    end_line_num=end_line,
                    id=str(uuid.uuid4())
                ))

        return doc_title, sections

    def parse_image_context(self, text: str) -> Dict[str, Dict[str, str]]:
        """
        Scans Markdown to map 'image_filename' -> 'Figure Caption'.
        """
        context_map = {}
        lines = text.split('\n')
        
        img_pattern = re.compile(r'!\[.*?\]\((.*?)\)')
        fig_pattern = re.compile(r'^\s*(\*\*|#+)?\s*(Fig\.?|Figure\.?)\s*(\d+[a-z]?)[:\.]?\s*(.*)', re.IGNORECASE)
        
        for i, line in enumerate(lines):
            img_match = img_pattern.search(line)
            if img_match:
                filename = Path(img_match.group(1)).name

                # Default info if no caption found
                img_info = {
                    "id": "Unknown", 
                    "caption": "No caption found", 
                    "line_index": i  # <--- CRITICAL: Store where the image is
                }
                
                # Look Ahead (next 5 lines) for a caption
                found_caption = False
                for j in range(1, 6):
                    if i + j >= len(lines): break
                    cap_match = fig_pattern.match(lines[i+j])
                    if cap_match:
                        img_info["id"] = f"{cap_match.group(2)} {cap_match.group(3)}"
                        img_info["caption"] = cap_match.group(4).strip("**").strip()
                        found_caption = True
                        break
                
                # If not found ahead, Look Behind (prev 5 lines)
                if not found_caption:
                    for j in range(1, 6):
                        if i - j < 0: break
                        cap_match = fig_pattern.match(lines[i-j])
                        if cap_match:
                            img_info["id"] = f"{cap_match.group(2)} {cap_match.group(3)}"
                            img_info["caption"] = cap_match.group(4).strip("**").strip()
                            break

                context_map[filename] = img_info
                            
        return context_map


    def map_images_to_sections(self, sections: List[SectionInfo], image_map: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Enrich the image map by appending the section info to the image node.
        """
        for filename, img_data in image_map.items():
            img_line = img_data['line_index']
            
            # Default assignment if no section matches
            img_data['section_title'] = "Unassigned"
            img_data['section_id'] = None
            
            # Find which section range covers this image line
            for section in sections:
                # Check if image line is within section (Start <= Image < End)
                if section.line_num <= img_line < section.end_line_num:
                    img_data['section_title'] = section.title
                    img_data['section_id'] = section.id
                    # Add section content snippet to image node context?
                    img_data['section_context_snippet'] = section.content 
                    break
                    
        return image_map


# ==============================================================================
# 3. Normalizer Logic (Solves Problem 2: Normalization)
# ==============================================================================
def calculate_standard_units(cond_val_str: str, cond_unit: str, temp_val_str: str) -> dict:
    """
    Deterministic normalization logic re-used from your original code.
    """
    try:
        # --- 1. Conductivity Normalization ---
        # Clean value
        c_val = float(re.sub(r'[^\d\.eE-]', '', str(cond_val_str)))
        
        u_clean = cond_unit.lower().replace(" ", "").replace("·", "").replace(".", "")
        multiplier = 1.0
        
        if "ms" in u_clean: multiplier = 1e-3
        elif "us" in u_clean or "μs" in u_clean: multiplier = 1e-6
        elif "ns" in u_clean: multiplier = 1e-9
        
        # Geometry check (S/m vs S/cm)
        if "m" in u_clean and "cm" not in u_clean and "mm" not in u_clean:
             if "m-1" in u_clean or "/m" in u_clean: # S m-1 -> S/cm
                 multiplier *= 0.01

        norm_cond = c_val * multiplier

        # --- 2. Temperature Normalization ---
        # Handle "1000/T" which is common in Arrhenius plots
        t_str = str(temp_val_str).lower().strip()
        norm_temp = None
        
        if "rt" in t_str or "room" in t_str:
            norm_temp = 25.0
        elif "1000/t" in t_str or "1000t" in t_str:
            # Assume the value passed is the result of 1000/T(K)
            # Extractor usually extracts the X-axis value, e.g., "2.5"
            val = float(re.sub(r'[^\d\.eE-]', '', t_str))
            if val > 0:
                kelvin = 1000.0 / val
                norm_temp = kelvin - 273.15
        else:
            # Standard number
            val = float(re.sub(r'[^\d\.eE-]', '', t_str))
            
            # Heuristic: If T > 200, it's likely Kelvin. If T < 100, likely Celsius.
            # Unless unit is explicitly C or K (we should extract unit separately ideally)
            if val > 200: # Likely Kelvin
                norm_temp = val - 273.15
            else: # Likely Celsius
                norm_temp = val

        return {"cond": norm_cond, "temp": norm_temp}
    except Exception as e:
        # print(f"Normalization error for {cond_val_str}: {e}")
        return {"cond": None, "temp": None}

# ==============================================================================
# 4. Canonicalizer (Solves Problem 3: Useless Names)
# ==============================================================================
async def canonicalize_materials(client, measurements: List[MeasuredPoint], definitions: List[str]):
    """
    Uses Gemini to resolve "x=0.1" -> "Li3.8Mg0.1..." using the extracted text definitions.
    """
    if not measurements: return measurements
    
    # Filter points that need resolution (short names or variable 'x')
    to_resolve = []
    for i, m in enumerate(measurements):
        if len(m.raw_composition) < 10 or "=" in m.raw_composition or "sample" in m.raw_composition.lower():
            to_resolve.append(i)
    
    if not to_resolve: return measurements

    print(f"   ... Resolving {len(to_resolve)} ambiguous material names...")

    # Build Context
    context_str = "\n".join([f"- {d}" for d in definitions])
    items_str = "\n".join([f"ID {i}: {measurements[i].raw_composition} (Source: {measurements[i].source_caption or 'Text'})" for i in to_resolve])

    prompt = f"""
    You are a Chemical Context Resolver.
    I have a list of abbreviated material names extracted from figures (e.g., "x=0.1").
    I have a list of Material Definitions found in the paper text.

    Your Task: Map the abbreviated names to their Full Canonical Chemical Formulas.

    DEFINITIONS FOUND IN TEXT:
    {context_str}

    ITEMS TO RESOLVE:
    {items_str}

    Logic:
    - If text says "series Li(4-2x)MgxTi(5-x)/3O4" and item is "x=0.1", calculate the formula:
      Li(4-0.2)Mg(0.1)Ti(1.63)O4 -> Li3.8Mg0.1Ti1.63O4.
    - If exact calculation isn't possible, return the General Formula with the specific variable (e.g. "Li(4-2x)MgxTi(5-x)/3O4 (x=0.1)").

    Return JSON: {{ "mappings": {{ "ID": "Canonical Formula" }} }}
    """
    
    try:
        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash", # Thinking model is great for stoichiometry math
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        mapping = json.loads(response.text).get("mappings", {})
        
        # Apply updates
        for i_str, formula in mapping.items():
            # Extract just the digits from keys like "ID 2" or "2"
            digits = re.search(r'\d+', str(i_str))
            if digits:
                idx = int(digits.group())
                # Ensure index is safe
                if 0 <= idx < len(measurements):
                    measurements[idx].canonical_formula = formula
            
    except Exception as e:
        print(f"   [Canonicalizer Error]: {e}")
    
    return measurements

# ==============================================================================
# 5. Gemini Pipelines (Text & Vision)
# ==============================================================================
async def process_text(client, model, text_content):
    print(f"   ... Extracting from Text...")
    prompt = """
    Extract ALL ionic conductivity measurements from the text.
    ALSO, extract any "Material Definitions" - sentences that describe the chemical formula of the samples (e.g., "solid solutions of Li(1-x)Mx...").
    
    For measurements, extract the raw strings exactly as they appear.
    """
    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents=[prompt, text_content],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=ExtractionResult.model_json_schema(),
                temperature=1.0, 
                thinking_config=types.ThinkingConfig(thinking_level="low")
            )
        )
        return ExtractionResult.model_validate_json(response.text)
    except Exception as e:
        print(f"   [Text Error]: {e}")
        return ExtractionResult(measurements=[])

async def process_image(client, model, img_path, context_info: dict):
    if "logo" in img_path.name.lower(): return []
    
    try:
        img_bytes = img_path.read_bytes()
        
        # INJECT CONTEXT HERE (Solves Problem 1)
        fig_id = context_info.get("id", "Unknown Figure")
        caption = context_info.get("caption", "No caption found.")
        
        prompt = f"""
        Analyze this scientific image.
        CONTEXT: This is {fig_id}.
        CAPTION: "{caption}"

        Task:
        1. If this is a Data Plot (Arrhenius or Conductivity vs X):
           - Use the CAPTION to understand what materials are being measured.
           - If the plot uses symbols (circles, squares), try to map them to the materials described in the caption or legend.
           - Extract x and y values. 
           - Note: If X-axis is 1000/T, capture that in 'raw_temperature'.
           - Note: If X-axis is 'x' (composition), capture that in 'raw_composition'.
        
        2. If this is a Table: extract rows.

        Extract raw strings. Do not convert units yet.
        """

        content = types.Content(
            parts=[
                types.Part(text=prompt),
                types.Part(
                    inline_data=types.Blob(
                        mime_type="image/png" if img_path.suffix.lower() == '.png' else "image/jpeg",
                        data=base64.b64encode(img_bytes).decode('utf-8')
                    ),
                    media_resolution={"level": "media_resolution_high"} # Essential
                )
            ]
        )

        response = await client.aio.models.generate_content(
            model=model,
            contents=[content],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=ExtractionResult.model_json_schema(),
                temperature=1.0
            )
        )
        
        result = ExtractionResult.model_validate_json(response.text)
        
        # Tag the source
        for m in result.measurements:
            m.source_figure_id = fig_id
            m.source_caption = caption # Store caption for the canonicalizer to use later
            if m.raw_composition == "Not Specified" and caption:
                 # Temporary fallback: put caption in composition so canonicalizer sees it
                 m.raw_composition = f"Series from {fig_id}" 
        
        print(f"   ✓ {img_path.name} ({fig_id}): Found {len(result.measurements)} points")
        return result.measurements

    except Exception as e:
        print(f"   [Image Error] {img_path.name}: {e}")
        return []

# ==============================================================================
# 6. Main Orchestrator
# ==============================================================================
async def run_pipeline(markdown_file, asset_dir, model):
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})
    
    # 1. Parse Markdown & Build Context Map
    text_content = markdown_file.read_text(encoding='utf-8')
    parser = MarkdownContextParser()
    # 1. Parse Sections and Title (New Functionality)
    doc_title, sections = parser.parse_structure(text_content)
    print(f"   --- Document Title: {doc_title}")
    print(f"   --- Found {len(sections)} sections.")


    # 2. Parse Image Contexts (Original Functionality)
    raw_image_map = parser.parse_image_context(text_content)
    print(f"   --- Context Map Built ({len(raw_image_map)} images linked) ---")
    
    # 3. Merge: Append Section Info to Image Nodes
    final_image_map = parser.map_images_to_sections(sections, raw_image_map)

    print(f"   --- Processed {len(final_image_map)} images.")

    if final_image_map:
        print('all keys', final_image_map.keys())
        for idx, k in enumerate(final_image_map.keys()):
            first_key = list(final_image_map.keys())[idx]
            print(f"   --- Example Node ({first_key}):")
            print(f"       Caption: {final_image_map[first_key]['caption']}")
            print(f"       Found In Section: {final_image_map[first_key]['section_title']}")

    for sec in sections:
        print(f"    {sec.title}")

    
    # 2. Extract from Text
    # text_result = await process_text(client, model, text_content)
    # all_measurements = text_result.measurements
    # material_defs = text_result.material_definitions
    
    # # 3. Extract from Images (Parallel)
    # image_patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    # image_files = []
    # for pattern in image_patterns:
    #     image_files.extend(list(asset_dir.glob(pattern)))
    # image_files = sorted(image_files)
    # sem = asyncio.Semaphore(5)
    
    # async def safe_image_call(img):
    #     async with sem:
    #         # Pass the context map info specifically for this image
    #         ctx = image_context_map.get(img.name, {})
    #         return await process_image(client, model, img, ctx)

    # image_results = await asyncio.gather(*[safe_image_call(img) for img in image_files])
    # for res in image_results:
    #     all_measurements.extend(res)

    # # 4. Canonicalize Names (Solves Problem 3)
    # # We pass the definitions found in text + the measurements from figures
    # all_measurements = await canonicalize_materials(client, all_measurements, material_defs)

    # # 5. Normalize Values (Solves Problem 2)
    # print("   ... Normalizing Units & Temperatures...")
    # for m in all_measurements:
    #     norm = calculate_standard_units(m.raw_conductivity, m.raw_unit, m.raw_temperature)
    #     m.normalized_conductivity = norm['cond']
    #     m.normalized_temperature_c = norm['temp']
        
    #     # Fallback: if canonical name is still empty, copy raw
    #     if not m.canonical_formula:
    #         m.canonical_formula = m.raw_composition

    # return all_measurements

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('markdown_file', type=Path)
    parser.add_argument('--asset_dir', type=Path)
    parser.add_argument('--model', default='gemini-3-flash-preview')
    args = parser.parse_args()

    start = time.time()
    measurements = asyncio.run(run_pipeline(args.markdown_file, args.asset_dir or args.markdown_file.parent, args.model))
    
    # Save
    out_path = args.markdown_file.parent / "robust_results.json"
    with open(out_path, 'w') as f:
        json.dump([m.model_dump() for m in measurements], f, indent=2)

    print(f"\nDone! Extracted {len(measurements)} points.")
    print(f"Saved to: {out_path}")

if __name__ == "__main__":
    main()