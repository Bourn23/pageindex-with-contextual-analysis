## In addition to splitting the processing of text and images
# we also add context to the images, add post processing to the units and resolve the material names
## Optimized how we process the nodes to prevent duplicate processing of text
## Also added feature for table and figure detection in the sections (so we can add the context to the images)
import os
import re
import argparse
import base64
import json
import time
import asyncio
import math
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Set
from pydantic import BaseModel, Field, ValidationError
import random
from google import genai
from google.genai import types
from dotenv import load_dotenv
import uuid
from dataclasses import dataclass, field

load_dotenv()


VISION_MODEL = "gemini-3-flash-preview"
TEXT_MODEL = "gemini-2.5-flash"
NUM_WORKERS = 5

try:
    import spacy
    from spacy.symbols import ORTH
    nlp = spacy.load("en_core_web_sm")
    
    # Add special cases to prevent splitting on abbreviations
    special_cases = ["Fig.", "Figs.", "Eq.", "Eqs.", "Tab.", "Tabs.", "Ref.", "Refs.", "al.", "vs.", "i.e.", "e.g."]
    for case in special_cases:
        nlp.tokenizer.add_special_case(case, [{ORTH: case}])
        
    SPACY_AVAILABLE = True
    print("✅ SpaCy loaded with custom tokenization rules.")
except ImportError:
    SPACY_AVAILABLE = False
    nlp = None
    print("⚠️ SpaCy not found. Using Regex fallback.")



## Utils
# ============================================================================
# UTILITIES for Robust LLM Execution + Cost Tracking
# ============================================================================
class CostTracker:
    # Pricing Estimates (USD per 1M tokens) - Update as pricing changes
    # Logic: Defaults to standard Pro/Flash tiers if exact model string isn't found
    PRICING_TIERS = [
        # Model Substring          Input Cost    Output Cost
        ("2.5-flash-lite",       {"input": 0.10, "output": 0.40}),
        ("2.5-flash",            {"input": 0.30, "output": 2.50}),
        ("2.5-pro",              {"input": 1.25, "output": 10.00}),
        ("3-flash",              {"input": 0.50, "output": 3.00}), # Matches "gemini-3-flash-preview"
        ("3-pro",                {"input": 2.00, "output": 12.00}), # Matches "gemini-3-pro-preview"
    ]

    # Fallback for older models (Gemini 2.5 Flash, etc.) if needed
    DEFAULT_PRICING = {"input": 0.30, "output": 2.50}

    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self.call_counts = {}

    def track(self, response, model_name: str):
        """Parses response metadata and accumulates cost."""
        if not response or not hasattr(response, 'usage_metadata') or not response.usage_metadata:
            return

        usage = response.usage_metadata
        in_tok = usage.prompt_token_count or 0
        try :out_tok = usage.total_token_count-in_tok or 0
        except: 
            print("Failed to calculate output tokens")
            print(usage)
            out_tok = usage.candidates_token_count if hasattr(usage, 'candidates_token_count') else 0
        
        
        # 2. Determine Pricing Tier
        model_name_lower = model_name.lower()
        pricing = self.DEFAULT_PRICING
        
        for substring, prices in self.PRICING_TIERS:
            if substring in model_name_lower:
                pricing = prices
                break # Stop at first match (vital for lite vs flash)
            
        # 3. Calculate Cost (Price per 1M tokens)
        cost = (in_tok / 1_000_000 * pricing["input"]) + \
               (out_tok / 1_000_000 * pricing["output"])

        # 4. Update Totals
        self.total_input_tokens += in_tok
        self.total_output_tokens += out_tok
        self.total_cost_usd += cost
        
        # Track calls per model
        self.call_counts[model_name] = self.call_counts.get(model_name, 0) + 1

    def print_summary(self):
        print("\n" + "="*50)
        print("💰 PIPELINE COST SUMMARY")
        print("="*50)
        print(f"{'Total Calls:':<20} {sum(self.call_counts.values())}")
        print(f"{'Total Input:':<20} {self.total_input_tokens:,} tokens")
        print(f"{'Total Output:':<20} {self.total_output_tokens:,} tokens")
        print("-" * 50)
        print(f"{'TOTAL COST:':<20} ${self.total_cost_usd:.4f}")
        print("-" * 50)
        print("Breakdown by Model:")
        for model, count in self.call_counts.items():
            print(f"  - {model:<30}: {count} calls")
        print("="*50 + "\n")

# Global singleton instance
tracker = CostTracker()

async def safe_text_call_with_retry(sec, client, model_name, sem, timeout=60, max_retries=3):
    async with sem:
        for attempt in range(max_retries):
            try:
                # Add a timeout to the specific model call
                result, raw_response = await asyncio.wait_for(
                    process_text(client, model_name, sec.content, sec.title), 
                    timeout=timeout
                )
                
                if raw_response:
                    tracker.track(raw_response, model_name)
                return result

            except asyncio.TimeoutError:
                print(f"   \033[93m[Timeout]\033[0m {sec.title} (Attempt {attempt+1})")
                if attempt == max_retries - 1: return None
                
            except Exception as e:
                err_str = str(e).lower()
                if "503" in err_str or "overloaded" in err_str:
                    wait_time = (2 ** attempt) + random.random()
                    print(f"   [Retry] {sec.title} - Model overloaded. Waiting {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"   \033[91m[Text Error]\033[0m {sec.title}: {e}")
                    break 
        return None

async def safe_image_call_with_retry(img_path, context, client, model_name, sem, timeout=90, max_retries=3):
    async with sem:
        for attempt in range(max_retries):
            try:
                # Images usually need a slightly longer timeout window
                result, raw_response = await asyncio.wait_for(
                    process_image(client, model_name, img_path, context), 
                    timeout=timeout
                )
                
                if raw_response:
                    tracker.track(raw_response, model_name)
                    return result
                
            except asyncio.TimeoutError:
                print(f"   \033[93m[Timeout]\033[0m {img_path.name} (Attempt {attempt+1})")
                
            except Exception as e:
                if "503" in str(e) or "overloaded" in str(e).lower():
                    wait_time = (2 ** attempt) + random.random()
                    print(f"   \033[93m[Retry]\033[0m {img_path.name} - Overloaded. Waiting {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)
                    continue
                
                # Check for that specific Pydantic JSON error
                if "validation error" in str(e).lower():
                    print(f"   \033[91m[Image Error]\033[0m {img_path.name}: Model returned malformed JSON")
                    return []
                
                print(f"   \033[91m[Image Error]\033[0m {img_path.name}: {e}")
                break
        return []
# ==============================================================================
# 1. Data Schema (Enhanced)
# ==============================================================================
@dataclass(frozen=True)
class ImageInfo:
    filename: str
    id: str          # e.g. "Fig 1"
    caption: str
    line_index: int

@dataclass(frozen=True)
class TableInfo:
    id: str          # e.g. "Table 1"
    content: str     # The raw markdown table text
    caption: str     # Found above/below table
    line_index: int

@dataclass
class SectionInfo:
    title: str
    content: str
    line_num: int
    end_line_num: int
    id: str
    # The section "owns" the assets discussed within it
    images: List[ImageInfo] = field(default_factory=list)
    tables: List[TableInfo] = field(default_factory=list)


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
    
    # source_figure_id: Optional[str] = Field(None, description="The real Figure ID (e.g. 'Fig. 5') if known.")
    # source_caption: Optional[str] = Field(None, description="The context from the figure caption.")
    source: str = Field(..., description="The source of the data choose from: 'figure', 'table', 'text'.")
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
    # --- IMPROVED REGEX ---
    # Matches: Fig 1, Fig. 1, Fig 1a, Fig 1(a), Figure 1(a), etc.
    # Logic:
    # 1. Prefix: Fig/Figure/Tab/Table (case insensitive)
    # 2. Separator: Optional dot + optional space
    # 3. Number: \d+
    # 4. Suffix: Optional letter (a) OR parenthesized letter (a)
    REF_PATTERN = re.compile(r'\b(Fig(?:\.|ure)?|Tab(?:\.|le)?)\s*(\d+)(?:[\s-]?(\(?[a-zA-Z]\)?))?', re.IGNORECASE)

    def parse_structure(self, md_text: str) -> Tuple[str, List[SectionInfo]]:
        """Parses headers to build document sections."""
        lines = md_text.split('\n')
        headers = []
        in_code_block = False
        header_pattern = re.compile(r'^(#{1,6})\s+(.*?)(?:\s+#+)?$')

        for i, line in enumerate(lines):
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                continue
            if in_code_block: continue
            
            match = header_pattern.match(line.strip())
            if match:
                headers.append({'level': len(match.group(1)), 'title': match.group(2).strip(), 'line_num': i})

        doc_title = "Untitled Document"
        sections: List[SectionInfo] = []

        # Helper to create section
        def create_section(title, start, end):
            content = "\n".join(lines[start:end]).strip()
            if content:
                return SectionInfo(title=title, content=content, line_num=start, end_line_num=end, id=str(uuid.uuid4()))
            return None

        if not headers:
            if not md_text.strip(): return doc_title, []
            return doc_title, [create_section("Full Text", 0, len(lines))]

        # Get Doc Title
        h1 = next((h for h in headers if h['level'] == 1), headers[0])
        doc_title = h1['title']
        
        slice_points = [h['line_num'] for h in headers] + [len(lines)]

        # Pre-header (Intro)
        if headers[0]['line_num'] > 0:
            sections.append(create_section("Introduction", 0, headers[0]['line_num']))

        # Main Sections
        for i, header in enumerate(headers):
            start = header['line_num'] + 1
            end = slice_points[i+1]
            title = "Abstract / Introduction" if (header['level'] == 1 and header['title'] == doc_title) else header['title']
            sec = create_section(title, start, end)
            if sec: sections.append(sec)

        return doc_title, sections


    def _normalize_id(self, prefix: str, number: str, suffix: str = None) -> str:
        """
        Standardizes IDs so "Fig. 4(a)" and "Figure 4a" both become "Fig 4a".
        """
        # Normalize Prefix
        clean_prefix = "Table" if "tab" in prefix.lower() else "Fig"
        
        # Normalize Suffix (remove parens, lower case)
        clean_suffix = ""
        if suffix:
            clean_suffix = suffix.replace('(', '').replace(')', '').strip().lower()
            
        return f"{clean_prefix} {number}{clean_suffix}"

    def parse_images(self, text: str) -> List[ImageInfo]:
        """Extracts images and nearby captions."""
        images = []
        lines = text.split('\n')
        img_pattern = re.compile(r'!\[.*?\]\((.*?)\)')

        for i, line in enumerate(lines):
            img_match = img_pattern.search(line)
            if img_match:
                filename = Path(img_match.group(1)).name
                caption, fig_id = "No caption found", "Unknown"
                
                # Context Search (Look Ahead 5, Behind 5)
                for direction in [1, -1]:
                    for j in range(1, 6):
                        idx = i + (j * direction)
                        if 0 <= idx < len(lines):
                            # Search line for "Fig X" pattern
                            # We use search() instead of match() to find it anywhere in the line
                            cap_match = self.REF_PATTERN.search(lines[idx])
                            if cap_match:
                                # Standardize: "Fig. 4(a)" -> "Fig 4a"
                                fig_id = self._normalize_id(cap_match.group(1), cap_match.group(2), cap_match.group(3))
                                caption = lines[idx].strip() # Use full line as caption
                                break
                    if fig_id != "Unknown": break

                images.append(ImageInfo(filename, fig_id, caption, i))
        return images

    def parse_tables(self, text: str) -> List[TableInfo]:
        """Extracts Markdown tables and nearby captions."""
        tables = []
        lines = text.split('\n')
        
        # Regex for table caption: "Table 1: Results"
        # Updated Regex:
        # 1. Removed ^ anchor and added support for leading HTML tags
        # 2. Made the capture of the description more robust
        tab_cap_pattern = re.compile(
            r'(?:<[^>]+>)?\s*(?:\*\*|#+)?\s*(Table|Tab\.?)\s*(\d+[a-z]?)\s*[:\.]?\s*(.*)', 
            re.IGNORECASE
        )

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            # Detect start of table (must start with |)
            if line.startswith('|'):
                start_line = i
                # Consume table lines
                table_lines = []
                while i < len(lines) and lines[i].strip().startswith('|'):
                    table_lines.append(lines[i])
                    i += 1
                
                # Search for caption (Look Behind 3, Ahead 3)
                caption, tab_id = "No caption found", "Unknown"
                search_indices = list(range(start_line - 3, start_line)) + list(range(i, i + 3))
                
                for idx in search_indices:
                    if 0 <= idx < len(lines):
                        cap_match = tab_cap_pattern.search(lines[idx])
                        if cap_match:
                            tab_id = f"Table {cap_match.group(2)}" # Normalize
                            caption = cap_match.group(3).strip()
                            break
                            
                tables.append(TableInfo(tab_id, "\n".join(table_lines), caption, start_line))
            else:
                i += 1
        return tables
    def _find_references_robust(self, text: str) -> Set[str]:
        """
        Unified extraction that works for both Regex and SpaCy modes.
        Directly scanning text is often more robust for IDs than token-walking.
        """
        refs = set()
        matches = self.REF_PATTERN.findall(text)
        
        for prefix, number, suffix in matches:
            # Normalize found ref to match image IDs
            # e.g. Found "Fig. 4(a)" -> Normalizes to "Fig 4a"
            norm_id = self._normalize_id(prefix, number, suffix)
            refs.add(norm_id)
            
            # OPTIONAL: Also add the "parent" ID. 
            # If text says "Fig 4a", also link "Fig 4" just in case the image is labeled "Fig 4"
            if suffix:
                 refs.add(self._normalize_id(prefix, number, None))
                 
        return refs

    def _find_references_spacy(self, text: str) -> Set[str]:
        """Uses SpaCy to find 'Fig 1', 'Table 2' references in text."""
        doc = nlp(text)
        refs = set()
        
        # Iterate through tokens to find patterns
        for i, token in enumerate(doc):
            # Check for "Fig", "Figure", "Table", "Tab"
            t_text = token.text.replace('.', '') # Handle "Fig." -> "Fig"
            
            if t_text in ["Fig", "Figure", "Table", "Tab"] and i + 1 < len(doc):
                next_token = doc[i+1]
                # Check if next token is a number (e.g., "1", "2a")
                # We use a simple regex check on the token text to allow "1a"
                if re.match(r'^\d+[a-z]?$', next_token.text):
                    # Standardize output
                    prefix = "Table" if t_text.startswith("Tab") else "Fig"
                    refs.add(f"{prefix} {next_token.text}")
                    
        return refs

    def _find_references_regex(self, text: str) -> Set[str]:
        """Fallback Regex if SpaCy is missing."""
        refs = set()
        # Pattern: (Fig|Table) [dot?] (Number)
        pattern = re.compile(r'\b(Fig|Figure|Tab|Table)\.?\s+(\d+[a-z]?)', re.IGNORECASE)
        matches = pattern.findall(text)
        for type_str, num in matches:
            prefix = "Table" if type_str.lower().startswith("tab") else "Fig"
            refs.add(f"{prefix} {num}")
        return refs

    def link_assets_to_sections(self, sections: List[SectionInfo], images: List[ImageInfo], tables: List[TableInfo]):
        """
        Links Images and Tables to Sections based on text references.
        """
        # Create lookups
        img_lookup = {img.id.lower(): img for img in images}
        tab_lookup = {tab.id.lower(): tab for tab in tables}

        for section in sections:
            # Use the robust regex scanner on the full section text
            # This bypasses tokenization issues with "(a)" completely
            refs = self._find_references_robust(section.content)
            
            # 2. Link Assets
            for ref in refs:
                ref_lower = ref.lower() # e.g. "fig 1" or "table 2"
                
                if ref_lower in img_lookup:
                    img = img_lookup[ref_lower]
                    if img not in section.images:
                        section.images.append(img)
                        
                elif ref_lower in tab_lookup:
                    tab = tab_lookup[ref_lower]
                    if tab not in section.tables:
                        section.tables.append(tab)

        # 3. Handle Orphans (Assets never mentioned in text)
        # Fallback to physical location
        linked_imgs = {img for sec in sections for img in sec.images}
        linked_tabs = {tab for sec in sections for tab in sec.tables}
        
        for img in images:
            if img not in linked_imgs:
                for sec in sections:
                    if sec.line_num <= img.line_index < sec.end_line_num:
                        sec.images.append(img); break
                        
        for tab in tables:
            if tab not in linked_tabs:
                for sec in sections:
                    if sec.line_num <= tab.line_index < sec.end_line_num:
                        sec.tables.append(tab); break

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
async def process_text(client, model, text_content, text_title):
    prompt = """
    Extract ALL ionic conductivity measurements from the text.
    ALSO, extract any "Material Definitions" - sentences that describe the chemical formula of the samples (e.g., "solid solutions of Li(1-x)Mx...").
    
    For measurements, extract the raw strings exactly as they appear.
    """
    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents=[prompt, text_title + "\n\n" + text_content],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=ExtractionResult.model_json_schema(),
                temperature=0.7 if '2.5' in model else 1.0, 
                # thinking_config=types.ThinkingConfig(thinking_level="low") if '2.5' in model else None
            )
        )
        if not response.text:
            print(f"   [Text Warning] Empty response for {text_title}")
            return ExtractionResult(measurements=[]), response

        return ExtractionResult.model_validate_json(response.text), response
    except Exception as e:
        print(f"   [Text Error] {text_title}: {e}")
        return ExtractionResult(measurements=[]), None

async def process_image(client, model, img_path, context_dict: dict):
    if "logo" in img_path.name.lower(): return []
    
    try:
        img_bytes = img_path.read_bytes()
        
        # INJECT CONTEXT HERE (Solves Problem 1)
        fig_id = context_dict.get("id", "Unknown Figure")
        caption = context_dict.get("caption", "No caption found.")
        section_title = context_dict.get("section_title", "Unknown Section")
        section_content = context_dict.get("section", "No section content found.")
        
        prompt = f"""
        Analyze this scientific image. and ONLY extract from the IMAGE.
        **Metadata:**
        - Figure ID: {fig_id}
        - Caption: {caption}
        - Found in Section: {section_title}
        - Section Content: {section_content}

        **Task**:
        1. If this is a Data Plot (Arrhenius or Conductivity vs X):
           - Use the CAPTION to understand what materials are being measured.
           - If the plot uses symbols (circles, squares), try to map them to the materials described in the caption or legend.
           - Extract x and y values. 
           - Note: If X-axis is 1000/T, capture that in 'raw_temperature'.
           - Note: If X-axis is 'x' (composition), capture that in 'raw_composition'.
        
        2. If this is a Table: extract rows.

        Extract raw strings. Do not convert units yet.
        """

        # print('>>>> prompt >>>>\n', prompt)
        # print('\n\n')

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
        # response = types.Content(text="")
        # response.text = ""
        
        if not response.text:
            print(f"   [Image Warning] {img_path.name}: Empty response (Safety filter?)")
            return [], response

        result = ExtractionResult.model_validate_json(response.text)
        
        # Tag the source and also check the source if it's other than figure we should skip those measurements
        for m in result.measurements:
            if m.source != "figure":
                print(f"   \033[91m[Image Warning]\033[0m {img_path.name}: {m.raw_composition} not from figure, skipping measurement")
                m.raw_composition = "Not Specified"
                m.raw_temperature = "Not Specified"
                m.normalized_temperature_c = None
                m.ionic_conductivity = None
                m.confidence = "low"
            # m.source_figure_id = fig_id
            # m.source_caption = caption # Store caption for the canonicalizer to use later
            if m.raw_composition == "Not Specified" and caption:
                 # Temporary fallback: put caption in composition so canonicalizer sees it
                 m.raw_composition = f"Series from {fig_id}" 
        
        if len(result.measurements) > 0:
            print(f"   ✓ {img_path.name} ({fig_id}): Found {len(result.measurements)} points")
        return result, response

    except Exception as e:
        print(f"   [Image Error] {img_path.name}: {e}")
        return [], None

# ==============================================================================
# 6. Main Orchestrator
# ==============================================================================
async def run_pipeline(markdown_file, asset_dir, model):
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})

    sem = asyncio.Semaphore(NUM_WORKERS)
    
    # 1. Parse Markdown & Build Context Map
    text_content = markdown_file.read_text(encoding='utf-8')
    parser = MarkdownContextParser()
    # 1. Parse Sections and Title (New Functionality)
    doc_title, sections = parser.parse_structure(text_content)
    all_images = parser.parse_images(text_content)
    all_tables = parser.parse_tables(text_content)
    
    # 2. Linking (The Magic Step)
    parser.link_assets_to_sections(sections, all_images, all_tables)

    # 3. Reporting
    print(f"   --- Document: {doc_title}")
    print(f"   --- Found: {len(sections)} Sections")
    print(f"   --- Found: {len(all_images)} Figures, {len(all_tables)} Tables")

    # check the length of all images found are the same as the image files in the asset directory
    img_patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    img_files = []
    for pattern in img_patterns:
        img_files.extend(list(asset_dir.glob(pattern)))
    img_files = sorted(img_files)
    if len(all_images) != len(img_files):
        print("   \033[91m[Warning]\033[0m Number of images found does not match number of image files in asset directory")

    # Example Output
    print("\n   --- Found Assets in Sections:")
    for sec in sections:
        has_assets = sec.images or sec.tables
        if has_assets:
            print(f"\n   >>> Section '{sec.title}' contains:")
            if sec.images: print(f"       - {len(sec.images)} Images: {[img.id for img in sec.images]}")
            if sec.tables: print(f"       - {len(sec.tables)} Tables: {[tab.id for tab in sec.tables]}")
    print("\n   --- Found Assets in Sections ---\n\n")
    
    # Build Image Context Registry
    # We need to map filename -> {caption, section_info} so the image processor can find it.
    image_context_registry = {}
    
    # Initialize with basic info from all_images
    for img in all_images:
        image_context_registry[img.filename.lower()] = {
            "id": img.id,
            "caption": img.caption,
            "section_title": "Unassigned", # Default
            "section_summary": ""
        }

    # print('>>> IMAGE CONTEXT REGISTRY >>>\n', image_context_registry)
    # print('\n\n')

    # Enrich with Section Data
    # (Since sections "own" images now, we reverse-lookup to fill the registry)
    for sec in sections:
        for img in sec.images:
            if img.filename.lower() in image_context_registry:
                
                image_context_registry[img.filename.lower()]["section_title"] = sec.title
                # We can pass the first 500 chars of the section as "background context" for the image
                image_context_registry[img.filename.lower()]["section"] = sec.content
                # print('>>> UPDATING IMAGE CONTEXT REGISTRY >>>', image_context_registry[img.filename.lower()])

    
    # Build tasks
    tasks = []
    # Extract from text (section-by-section)
    all_measurements = []
    material_defs = [] # You might want to accumulate these or merge them later

    # We process sections in parallel batches (or sequentially if rate limits matter)
    # Here is a sequential loop for safety, or you can use asyncio.gather for speed.
    skip_sections = ['acknowledgements', 'references']
    for sec in sections:
        if any(skip in sec.title.lower() for skip in skip_sections):
            print(f"   \033[91m[Warning]\033[0m Skipping section: {sec.title}")
            continue        
        tasks.append(safe_text_call_with_retry(sec, client, TEXT_MODEL, sem))

    # # 3. Extract from Images (Parallel)
    # Add Image Tasks
    img_files = []
    for pattern in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
        img_files.extend(list(asset_dir.glob(pattern)))
    
    for img_path in sorted(img_files):
        # Retrieve context from registry (as you already do)
        filename = img_path.name

        # this is a temporary fix
        try:
            context = image_context_registry[filename]
        except:
            context = image_context_registry.get('_'+filename, {"id": "Unknown", "caption": "No caption"})
        tasks.append(safe_image_call_with_retry(img_path, context, client, VISION_MODEL, sem))
    # 4. Execute Simultaneously
    print(f"   ... Processing {len(tasks)} items (Text + Images) simultaneously ...")
    results = await asyncio.gather(*tasks)

    # Merge Image Results
    all_measurements = []
    material_defs = []

    for res in results:
        if not res: continue
        
        # Distinguish between Text ExtractionResult and Image Measurement list
        # This assumes process_text returns an object and process_image returns a list
        if hasattr(res, 'measurements'):
            all_measurements.extend(res.measurements)
            material_defs.extend(res.material_definitions)
        else:
            # Result from image processing
            all_measurements.extend(res.measurements)
            material_defs.extend(res.material_definitions)

    # # 4. Canonicalize Names (Solves Problem 3)
    # # We pass the definitions found in text + the measurements from figures
    # all_measurements = await canonicalize_materials(client, all_measurements, material_defs)

    # # 5. Normalize Values (Solves Problem 2)
    print("   ... Normalizing Units & Temperatures...")
    for m in all_measurements:
        norm = calculate_standard_units(m.raw_conductivity, m.raw_unit, m.raw_temperature)
        m.normalized_conductivity = norm['cond']
        m.normalized_temperature_c = norm['temp']
        
        # Fallback: if canonical name is still empty, copy raw
        if not m.canonical_formula:
            m.canonical_formula = m.raw_composition

    return all_measurements

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
    tracker.print_summary()

if __name__ == "__main__":
    main()