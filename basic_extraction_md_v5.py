## In addition to splitting the processing of text and images
# we also add context to the images, add post processing to the units and resolve the material names
## Optimized how we process the nodes to prevent duplicate processing of text
## Also added feature for table and figure detection in the sections (so we can add the context to the images)

## V4->v5: add limits on the token output (it was 104k token output lool)
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
from scifigure_parser import SciFigureParser

load_dotenv()


VISION_MODEL = "gemini-3-flash-preview"
TEXT_MODEL = "gemini-flash-latest"
NUM_WORKERS = 5
FILE_DIR = ""

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
                result, raw_response, success = await asyncio.wait_for(
                    process_text(client, model_name, sec.content, sec.title), 
                    timeout=timeout
                )
                
                if raw_response:
                    tracker.track(raw_response, model_name)
                return result, success

            except asyncio.TimeoutError:
                print(f"   \033[93m[Timeout]\033[0m {sec.title} (Attempt {attempt+1})")
                if attempt == max_retries - 1: return None, False
                
            except Exception as e:
                err_str = str(e).lower()
                if "503" in err_str or "overloaded" in err_str:
                    wait_time = (2 ** attempt) + random.random()
                    print(f"   [Retry] {sec.title} - Model overloaded. Waiting {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"   \033[91m[Text Error]\033[0m {sec.title}: {e}")
                    break 
        return None, False

async def safe_image_call_with_retry(img_path, context, client, model_name, sem, sf_parser=None, timeout=120, max_retries=3):
    """
    This function now only retries on ACTUAL failures, not empty results.
    """
    async with sem:
        try:
            result, raw_response, success = await asyncio.wait_for(
                process_image(client, model_name, img_path, context, sf_parser=sf_parser), 
                timeout=timeout
            )
            
            if raw_response:
                tracker.track(raw_response, model_name)
            
            # Return regardless of success - we'll handle failures upstream
            return result, success
            
        except asyncio.TimeoutError:
            print(f"   ⏱️  {img_path.name}: Timeout")
            return [], False
            
        except Exception as e:
            # Handle 503 overload errors with retry
            if "503" in str(e) or "overloaded" in str(e).lower():
                wait_time = random.uniform(1, 3)
                print(f"   🔄 {img_path.name}: Overloaded, retrying in {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
                # Retry once on overload
                try:
                    result, raw_response, success = await asyncio.wait_for(
                        process_image(client, model_name, img_path, context, sf_parser=sf_parser), 
                        timeout=timeout
                    )
                    if raw_response:
                        tracker.track(raw_response, model_name)
                    return result, success
                except:
                    pass
            
            print(f"   ❌ {img_path.name}: {e}")
            return [], False
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
    

    material_definitions: List[str] = Field(
        default_factory=list, 
        description="Brief 4-5 concise sentences that define material series (e.g. 'solid solutions Li(4-2x)MgxTi(5-x)/3O4')."
    )

    raw_conductivity: str = Field(..., description="Ionic conductivity value as extracted (e.g. '1.24e-4', '5.2').")
    raw_conductivity_unit: str = Field(..., description="Corresponding ionic conductivity unit as extracted (e.g. 'mS/cm', 'S cm-1').")
    normalized_conductivity: Optional[float] = Field(None, description="Normalized ionic conductivity value in S/cm.")
    
    raw_temperature: str = Field(
        ..., 
        description="The numeric value ONLY. Example: if axis says '1000/T = 3.35', extract '3.35'."
    )
    raw_temperature_unit: str = Field(
        ..., 
        description="The unit or axis label exactly as shown. Example: 'C', 'K', '1000/T (K-1)', '10^3/T'."
    )

    normalized_temperature_c: Optional[float] = Field(None, description="Temperature in Celsius.")
    
    # source_figure_id: Optional[str] = Field(None, description="The real Figure ID (e.g. 'Fig. 5') if known.")
    # source_caption: Optional[str] = Field(None, description="The context from the figure caption.")
    source: str = Field(..., description="The source of the data choose from: 'figure', 'table', 'text'.")
    confidence: str = Field(..., description="high/medium/low")

class ExtractionResult(BaseModel):
    measurements: List[MeasuredPoint]
    # We also extract "Material Definitions" from text to help us resolve "x=0.1" later
    

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

    def _extract_nearby_text(self, img: ImageInfo, section_text: str, window_lines=5, max_chars=1000) -> str:
        """
        Extract text near the image reference for context.
        
        Args:
            img: ImageInfo object with line_index
            section_text: Full section content
            window_lines: How many lines above/below to include
            max_chars: Maximum characters to return
            
        Returns:
            Truncated nearby text
        """
        lines = section_text.split('\n')
        
        # Calculate window bounds
        start_idx = max(0, img.line_index - window_lines)
        end_idx = min(len(lines), img.line_index + window_lines + 1)
        
        # Extract nearby lines
        nearby_lines = lines[start_idx:end_idx]
        nearby_text = '\n'.join(nearby_lines).strip()
        
        # Truncate to max_chars
        if len(nearby_text) > max_chars:
            nearby_text = nearby_text[:max_chars] + "..."
        
        return nearby_text
    

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
def calculate_standard_units(cond_val: str, cond_unit: str, temp_val: str, temp_unit: str) -> dict:
    """
    Robust normalizer using the split Value/Unit fields.
    """
    try:
        # --- 1. Temperature Normalization ---
        # The LLM has already cleaned 'temp_val' to be just a number (e.g. "2.4")
        raw_t = float(temp_val) 
        unit_clean = temp_unit.lower().strip()
        
        temp_k = None      # Kelvin (needed for conductivity calc)
        norm_temp_c = None # Celsius (for DB)

        # CHECK 1: Is this an Arrhenius inverse scale?
        # We look for "1000" or "10^3" combined with "T" in the UNIT field
        if ("1000" in unit_clean or "10^3" in unit_clean) and "t" in unit_clean:
            # Formula: X = 1000/T  ->  T = 1000/X
            if raw_t > 0:
                temp_k = 1000.0 / raw_t
                norm_temp_c = temp_k - 273.15
        
        # CHECK 2: Standard Kelvin
        elif "k" in unit_clean and "c" not in unit_clean: 
             # (checking 'c' prevents matching 'black' or 'thick')
             temp_k = raw_t
             norm_temp_c = raw_t - 273.15
             
        # CHECK 3: Standard Celsius
        elif "c" in unit_clean:
            norm_temp_c = raw_t
            temp_k = raw_t + 273.15
            
        # Fallback: Guess based on magnitude if unit is ambiguous
        else:
            if raw_t > 200: # Likely Kelvin
                temp_k = raw_t
                norm_temp_c = raw_t - 273.15
            else: # Likely Celsius
                norm_temp_c = raw_t
                temp_k = raw_t + 273.15

        # --- 2. Conductivity Normalization ---
        # Same logic as before, but using the robust 'temp_k'
        raw_c = float(cond_val) # Assuming LLM cleaned this too
        cond_u_clean = cond_unit.lower().strip()
        norm_cond = None

        if "log" in cond_u_clean and temp_k:
            # Handle log(σT) -> σ = (10^y)/T
            sigma_times_t = 10 ** raw_c
            norm_cond = sigma_times_t / temp_k
            
        elif "ln" in cond_u_clean and temp_k:
             # Handle ln(σT) -> σ = (e^y)/T
            import math
            sigma_times_t = math.exp(raw_c)
            norm_cond = sigma_times_t / temp_k
            
        else:
            # Standard Linear Units
            multiplier = 1.0
            if "ms" in cond_u_clean: multiplier = 1e-3
            elif "us" in cond_u_clean: multiplier = 1e-6
            elif "ns" in cond_u_clean: multiplier = 1e-9
            
            # Geometry fix (S/m -> S/cm)
            if "m" in cond_u_clean and "cm" not in cond_u_clean:
                if "m-1" in cond_u_clean or "/m" in cond_u_clean:
                    multiplier *= 0.01

            norm_cond = raw_c * multiplier

        return {"cond": norm_cond, "temp": round(norm_temp_c, 2)}

    except Exception as e:
        print(f"Norm Error: {e}")
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
async def process_text(client, model, text_content, text_title, max_retries: int = 3):
    # prompt = """
    # Extract ALL ionic conductivity measurements from the text.
    # ALSO, extract any "Material Definitions" - sentences that describe the chemical formula of the samples (e.g., "solid solutions of Li(1-x)Mx...").
    
    # For measurements, extract the raw strings exactly as they appear.
    # """

    prompt = """Extract ionic conductivity data points. Return Format: JSON only. 
        For each measurement:
        - raw_composition: Material name
        - raw_conductivity: Numeric value (e.g. "1.2e-4") without any ~ or < or > symbols
        - raw_conductivity_unit: Unit (e.g. "S/cm")
        - raw_temperature: Only the temperature value (e.g. "25", "298", "2.0", "2.4")
        - raw_temperature_unit: Unit (e.g. "Celsius", "Kelvin", "1000/T (K-1)", "10^3/T / K-1")
        - source: "text"
        - confidence: "high" if it is explicitly stated / "low" if it is inferred or calculated or was cited from another source

        Optional: Extract up to 4-5 material definition sentences that summarizes chemical formula, processing method, and any other information about the samples mentioned in the text."""

    last_exception = None

    # Store the results to a file
    with open(f"{FILE_DIR}/results_log_v5.json", "a") as f:
        

        for attempt in range(1, max_retries+1):
            try:
                response = await client.aio.models.generate_content(
                    model=model,
                    contents=[prompt, text_title + "\n\n" + text_content],
                    config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema=ExtractionResult.model_json_schema(),
                    temperature=0.7 if '2.5' in model else 1.0, 
                    max_output_tokens=4096,
                    # thinking_config=types.ThinkingConfig(thinking_level="low") if '2.5' in model else None
                    )
                )
                if not response.text:
                    print(f"   [Text Warning] Empty response for {text_title}")
                    f.write(f"\n\n--- DEBUG INFO FOR {text_title} ---\n\n")
                    f.write(f"\n\n--- [Text Warning] Empty response for {text_title}")

                    return ExtractionResult(measurements=[]), response

                # if response:
                #     print(f"\n   [DEBUG] {text_title}:")
                #     print(f"   - Response length: {len(response.text) if response.text else 0} chars")
                #     print(f"   - Usage metadata: {response.usage_metadata}")
                #     print(f"   - First 500 chars: {response.text}")
            
                # Write detailed debug info to file
                f.write(f"\n\n[DEBUG] {text_title}:\n")
                f.write(f"- Response length: {len(response.text) if response.text else 0} chars\n")
                f.write(f"- Usage metadata: {response.usage_metadata}\n")
                f.write(f"- chars: {response.text}\n")

                result = ExtractionResult.model_validate_json(response.text)
                
                return result, response, True
            except (ValidationError, json.JSONDecodeError) as e:
                # Capture specific JSON errors (Truncated JSON, Invalid JSON)
                last_exception = e
                print(f"   [Retry {attempt}/{max_retries}] JSON Error for {text_title}: {str(e)[:100]}...")
                
                # Optional: Exponential backoff
                await asyncio.sleep(1 * attempt)
            except Exception as e:
                # Capture other API errors (503, etc)
                last_exception = e
                print(f"   [Retry {attempt}/{max_retries}] API Error for {text_title}: {e}")
                await asyncio.sleep(1 * attempt)
        
        # FINAL FAILURE HANDLER
        print(f"   \033[91m[Text Error]\033[0m {text_title}: Failed after {max_retries} attempts. Last error: {last_exception}")
        return ExtractionResult(measurements=[]), None, False

async def process_image(client, model, img_path, context_dict: dict, max_retries: int = 3, sf_parser: Optional[SciFigureParser] = None):
    if "logo" in img_path.name.lower(): 
        return [], None, True
    
    try:
        fig_id = context_dict.get("id", "Unknown Figure")
        caption = context_dict.get("caption", "No caption found.")
        section_title = context_dict.get("section_title", "Unknown Section")
        section_content = context_dict.get("section", "No section content found.")
    except Exception as e:
        print(f"   [Context Error] {img_path.name}: {e}")
        return [], None, False

    # --- SCI-FIGURE PARSER INTEGRATION ---
    if sf_parser:
        try:
            print(f"   🔍 {img_path.name} ({fig_id}): Detecting subplot...")
            box = sf_parser.detect_subplot(str(img_path), "ionic conductivity")
            
            # Check if SciFigureParser detected ionic conductivity data
            if not box.get("isIonicConductivity", True):
                print(f"   \033[93m[Skip]\033[0m {img_path.name} ({fig_id}): No ionic conductivity measurements detected by SciFigureParser.")
                return ExtractionResult(measurements=[]), None, True

            print(f"   ✂️ {img_path.name}: Cropping relevant region...")
            cropped_path = sf_parser.crop_image(str(img_path), box, padding=80)
            
            print(f"   📊 {img_path.name}: Extracting data with grid grounding...")
            sf_result = sf_parser.extract_data(cropped_path, grid_config={"enabled": True, "rows": 2, "cols": 2})
            
            # Map SciFigureParser result to ExtractionResult
            measurements = []
            # We relax the check: if isIonicConductivity is missing or True, AND we have data points, we extract them.
            # Since we specifically asked for ionic conductivity subplots, we can be a bit more trustful here.
            should_extract = sf_result.get("isIonicConductivity", True)
            if should_extract:
                for dp in sf_result.get("dataPoints", []):
                    m = MeasuredPoint(
                        raw_composition=dp.get("label", "Unknown"),
                        raw_conductivity=str(dp.get("yValue")),
                        raw_conductivity_unit=sf_result.get("yAxis", {}).get("unit", "S/cm"),
                        raw_temperature=str(dp.get("xValue")),
                        raw_temperature_unit=sf_result.get("xAxis", {}).get("unit", "Celsius"),
                        source="figure",
                        confidence="high"
                    )
                    measurements.append(m)
            
            result = ExtractionResult(measurements=measurements)
            
            # Write to debug log
            log_dir = FILE_DIR if FILE_DIR else img_path.parent
            with open(f"{log_dir}/results_log_v5.json", "a") as f:
                f.write(f"\n\n[SCI-FIGURE DEBUG] {img_path.name}:\n")
                f.write(f"- Result: {json.dumps(sf_result, indent=2)}\n")

            if len(result.measurements) > 0:
                print(f"   ✓ {img_path.name} ({fig_id}): Found {len(result.measurements)} points via SciFigureParser")
            
            return result, None, True # raw_response is None for now as it's handled inside parser
            
        except Exception as e:
            print(f"   ⚠️ {img_path.name}: SciFigureParser failed: {e}. Falling back to standard processing...")
            # Fall through to standard processing

    # Standard processing as fallback
    try:
        img_bytes = img_path.read_bytes()
    except Exception as e:
        print(f"   [Image Read Error] {img_path.name}: {e}")
        return [], None, False
        
    prompt = f"""
    Analyze this scientific image and determine if it contains ionic conductivity measurements.

    **Metadata:**
    - Figure ID: {fig_id}
    - Caption: {caption}
    - Found in Section: {section_title}
    - Section Content: {section_content}

    *** EXAMPLES OF TEMPERATURE EXTRACTION ***
    Case 1: Standard Celsius
    Input: "Temperature was maintained at 25 °C"
    Output:
    {{
    "raw_temperature": "25",
    "raw_temperature_unit": "Celsius"
    }}

    Case 2: Standard Kelvin
    Input: "Measured at 298 K"
    Output:
    {{
    "raw_temperature": "298",
    "raw_temperature_unit": "K"
    }}

    Case 3: Arrhenius Plot (Inverse Temperature)
    Input: "The x-axis shows 1000/T (K⁻¹) ranging from 2.0 to 3.5"
    Output:
    {{
    "raw_temperature": "2.0",
    "raw_temperature_unit": "1000/T (K-1)"
    }}

    Case 4: Complex Axis Label
    Input: "Graph axis: 10³/T / K⁻¹ value is 2.4"
    Output:
    {{
    "raw_temperature": "2.4",
    "raw_temperature_unit": "10^3/T / K-1"
    }}

    **Task**:
    **Step 1: Classify the image**
    Is this a:
    - [ ] Data plot with conductivity values (Arrhenius plot, bar chart, etc.)
    - [ ] Table with conductivity measurements
    - [ ] Structural diagram / schematic / photo (NO DATA)
    
    **Step 2: Extract (ONLY if you checked the first two options)**
    If this contains conductivity data, extract measurements. Otherwise return empty.

    **Step 3: Extract Material Definitions**
    - If the image contains a description of the material composition, extract it.
    - Return the definition in a concise format.

    Return JSON with measurements array (can be empty). Do not convert units yet.
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
                media_resolution={"level": "media_resolution_high"}
            )
        ]
    )


    # Re-try logic
    last_exception = None
    log_dir = FILE_DIR if FILE_DIR else img_path.parent
    for attempt in range(1, max_retries+1):
        with open(f"{log_dir}/results_log_v5.json", "a") as f:
            try:
                response = await client.aio.models.generate_content(
                    model=model,
                    contents=[content],
                    config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema=ExtractionResult.model_json_schema(),
                    temperature=1.0,
                    max_output_tokens=16384,
                    # thinking_config=types.ThinkingConfig(thinking_level="medium")
                )
            )
                # response = types.Content(text="")
                # response.text = ""
            
                if not response.text:
                    print(f"   ⚠️  {img_path.name}: Empty response (likely safety filter)")
                    f.write(f"\n\n--- [Image Warning] Empty response for {img_path.name}")
                    last_exception = "Empty response"
                    await asyncio.sleep(1 * attempt)
                    continue

                # Write extensive debugging info to the file
                f.write(f"\n\n[DEBUG] {img_path.name}:\n")
                f.write(f"- Response length: {len(response.text) if response.text else 0} chars\n")
                f.write(f"- Usage metadata: {response.usage_metadata}\n")
                f.write(f"- chars: {response.text}\n")

                result = ExtractionResult.model_validate_json(response.text)
            
                # Tag the source and also check the source if it's other than figure we should skip those measurements
                for m in result.measurements:
                    if m.source != "figure":
                        print(f"   \033[91m[Image Warning]\033[0m {img_path.name}: {m.raw_composition} not from figure, skipping measurement")
                        m.raw_composition = "Not Specified"
                        m.raw_temperature = "Not Specified"
                        m.normalized_temperature_c = None
                        m.confidence = "low"
                    if m.raw_composition == "Not Specified" and caption:
                        # Temporary fallback: put caption in composition so canonicalizer sees it
                        m.raw_composition = f"Series from {fig_id}" 
                
                if len(result.measurements) > 0:
                    print(f"   ✓ {img_path.name} ({fig_id}): Found {len(result.measurements)} points")
                return result, response, True
            except (ValidationError, json.JSONDecodeError) as e:
                # These are REAL failures - malformed JSON
                last_exception = e
                print(f"   [Retry {attempt}/{max_retries}] {img_path.name}: {str(e)[:100]}...")
                await asyncio.sleep(1 * attempt)
                
            except Exception as e:
                # API errors
                last_exception = e
                print(f"   [Retry {attempt}/{max_retries}] {img_path.name}: {e}")
                await asyncio.sleep(1 * attempt)
        # All retries exhausted
    print(f"   ❌ {img_path.name}: Failed after {max_retries} attempts - {last_exception}")
    return ExtractionResult(measurements=[]), None, False


# ==============================================================================
# 6. Main Orchestrator
# ==============================================================================
async def run_pipeline(markdown_file, asset_dir, model):
    global FILE_DIR
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})

    sem = asyncio.Semaphore(NUM_WORKERS)

    # Create a log file
    FILE_DIR = markdown_file.parent
    with open(f"{FILE_DIR}/results_log_v5.json", "w") as f:
        f.write(f"\n\n--- [Document] {markdown_file.name} ---\n")
    
    # 1. Parse Markdown & Build Context Map
    text_content = markdown_file.read_text(encoding='utf-8')
    parser = MarkdownContextParser()
    # 1. Parse Sections and Title (New Functionality)
    doc_title, sections = parser.parse_structure(text_content)
    all_images = parser.parse_images(text_content)
    all_tables = parser.parse_tables(text_content)
    
    # 2. Linking (The Magic Step)
    parser.link_assets_to_sections(sections, all_images, all_tables)

    # 2.5 Initialize SciFigureParser
    sf_parser = SciFigureParser(api_key=api_key, model_name=VISION_MODEL, debug=True)

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
                image_context_registry[img.filename.lower()]["section"] = parser._extract_nearby_text(
                        img, sec.content, window_lines=5, max_chars=1000
                    )
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
        tasks.append(safe_image_call_with_retry(img_path, context, client, VISION_MODEL, sem, sf_parser=sf_parser))

    # 4. Execute Simultaneously
    print(f"   ... Processing {len(tasks)} items (Text + Images) simultaneously ...")
    results = await asyncio.gather(*tasks)

    # Merge Image Results
    all_measurements = []
    failed_count = 0
    skipped_count = 0

    for i, res in enumerate(results):
        # Handle exceptions from gather
        if isinstance(res, Exception):
            print(f"   ❌ Task {i} raised exception: {res}")
            failed_count += 1
            continue
        
        # Check if this is an image result (tuple) or text result (object)
        if isinstance(res, tuple): # this is for image results
            result, success = res
            if success:
                if len(result.measurements) == 0:
                    skipped_count += 1  # No data found (valid)
                else:
                    all_measurements.extend(result.measurements)
            else:
                failed_count += 1  # Actual failure
        elif hasattr(res, 'measurements'):
            # Text extraction result
            if len(res.measurements) == 0:
                skipped_count += 1
            else:
                all_measurements.extend(res.measurements)
        elif res is None:
            failed_count += 1

    # # 4. Canonicalize Names (Solves Problem 3)
    # # We pass the definitions found in text + the measurements from figures
    # all_measurements = await canonicalize_materials(client, all_measurements, material_defs)

    # # 5. Normalize Values (Solves Problem 2)
    print("   ... Normalizing Units & Temperatures...")
    for m in all_measurements:
        norm = calculate_standard_units(m.raw_conductivity, m.raw_conductivity_unit, m.raw_temperature, m.raw_temperature_unit)
        m.normalized_conductivity = norm['cond']
        m.normalized_temperature_c = norm['temp']
        
        # Fallback: if canonical name is still empty, copy raw
        if not m.canonical_formula:
            m.canonical_formula = m.raw_composition

    # Report statistics
    print(f"\n   📊 Extraction Summary:")
    print(f"      - Extracted: {len(all_measurements)} measurements")
    print(f"      - Skipped (no data): {skipped_count} images")
    print(f"      - Failed: {failed_count} items")

    pipeline_stats = {
        "extracted_count": len(all_measurements),
        "skipped_images": skipped_count,
        "failed_items": failed_count,
        "total_sections_processed": len(sections),
        "total_images_processed": len(img_files)
    }

    # [CHANGED] Return stats along with data
    return all_measurements, material_defs, pipeline_stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('markdown_file', type=Path)
    parser.add_argument('--asset_dir', type=Path)
    parser.add_argument('--model', default='gemini-3-flash-preview')
    args = parser.parse_args()

    start_time = time.time()
    measurements, material_definitions, stats = asyncio.run(run_pipeline(args.markdown_file, args.asset_dir or args.markdown_file.parent, args.model))
    
    elapsed_time = time.time() - start_time

    # Save
    out_path = args.markdown_file.parent / "robust_results_v5.json"
    output_data = {
        'doc_name': args.markdown_file.stem,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'execution_time_seconds': round(elapsed_time, 2),
        
        # 1. Configuration & Provenance
        'config': {
            'vision_model': VISION_MODEL,
            'text_model': TEXT_MODEL,
            'normalization_engine': "v2_hybrid_llm_python", # Tracking your version
        },
        
        # 2. Cost & Usage (Crucial for tracking)
        'cost_summary': {
            'total_input_tokens': tracker.total_input_tokens,
            'total_output_tokens': tracker.total_output_tokens,
            'total_cost_usd': round(tracker.total_cost_usd, 4),
            'call_counts': tracker.call_counts
        },
        
        # 3. Pipeline Health Stats
        'extraction_stats': stats,
        
        # 4. The Data
        'material_count': len(measurements),
        'measurements': [m.model_dump() for m in measurements],
        'material_definitions': material_definitions
    }

    with open(out_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nDone! Extracted {len(measurements)} points.")
    print(f"Saved to: {out_path}")
    tracker.print_summary()

if __name__ == "__main__":
    main()