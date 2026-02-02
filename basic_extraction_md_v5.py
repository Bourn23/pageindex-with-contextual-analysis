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
from dataclasses import dataclass, field, asdict
from scifigure_parser import SciFigureParser
from pageindex.llm_client import get_llm_client
import numpy as np


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

async def safe_image_call_with_retry(img_path, context, client, model_name, sem, sf_parser=None, timeout=240, max_retries=3):
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
            
async def safe_table_call_with_retry(table_data, client, model_name, sem, timeout=120, max_retries=3):
    async with sem:
        for attempt in range(max_retries):
            try:
                result, raw_response, success = await asyncio.wait_for(
                    process_table_node(client, model_name, table_data), 
                    timeout=timeout
                )
                
                if raw_response:
                    tracker.track(raw_response, model_name)
                return result, success

            except asyncio.TimeoutError:
                print(f"   \033[93m[Timeout]\033[0m {table_data['caption']} (Attempt {attempt+1})")
                if attempt == max_retries - 1: return None, False
                
            except Exception as e:
                err_str = str(e).lower()
                if "503" in err_str or "overloaded" in err_str:
                    wait_time = (2 ** (attempt + 1)) + random.random()
                    print(f"   [Retry] {table_data['caption']} - Model overloaded. Waiting {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"   \033[91m[Table Error]\033[0m {table_data['caption']}: {e}")
                    break 
    return ExtractionResult(measurements=[]), False
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
    
    source_figure_id: Optional[str] = Field(None, description="The real Figure ID (e.g. 'Fig. 5') if known.")
    source_caption: Optional[str] = Field(None, description="The context from the figure caption.")
    source: str = Field(..., description="The source of the data choose from: 'figure', 'table', 'text'.")
    confidence: str = Field(..., description="high/medium/low")
    warnings: List[str] = Field(default_factory=list, description="Warnings about the data.")

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
                
                # Search for caption (Look Behind 5, Ahead 5)
                caption, tab_id = "No caption found", "Unknown"
                search_indices = list(range(start_line - 5, start_line)) + list(range(i, i + 5))
                
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
    Handles non-numeric values like "room temperature" or "RT".
    """
    def safe_float(val: str) -> float:
        if not val:
            raise ValueError("Empty value")
        
        # Clean string
        clean = str(val).lower().strip().replace(',', '')
        
        # Handle "room temperature" and variations
        if clean in ["room temperature", "rt", "room temp", "room-temperature"]:
            return 25.0
            
        # Try direct conversion
        try:
            return float(clean)
        except ValueError:
            # Try to extract the first number
            import re
            match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", clean)
            if match:
                return float(match.group())
            raise ValueError(f"Could not convert '{val}' to float")

    try:
        # --- 1. Temperature Normalization ---
        # Get raw value, handling room temperature and known non-numeric strings
        temp_val_clean = str(temp_val).lower().strip()
        if temp_val_clean in ["n/a", "none", "unknown", "arrhenius plot", "not specified"]:
             return {"cond": None, "temp": None}

        raw_t = safe_float(temp_val)
        unit_clean = temp_unit.lower().strip()
        
        # If it was "RT", force Celsius unit if not already set specifically
        if temp_val_clean in ["room temperature", "rt", "room temp", "room-temperature"]:
            if not unit_clean or unit_clean == "celsius":
                unit_clean = "c" # Force Celsius path

        temp_k = None      # Kelvin (needed for conductivity calc)
        norm_temp_c = None # Celsius (for DB)
        
        # Logic Branch: Composition vs Arrhenius
        # If the extracted "temperature" is actually a small number (x < 1.0) and unit is ambiguous,
        # it is likely a Composition value (x in Li...x...), NOT temperature.
        # In this case, we assume Room Temperature (25 C).
        # ADDED: Check for common stoichiometry labels in unit string
        if (raw_t < 1.0 or any(m in unit_clean for m in ['x=', 'z=', 'y='])) and "k" not in unit_clean and "c" not in unit_clean:
             norm_temp_c = 25.0
             temp_k = 298.15
        
        # CHECK 1: Is this an Arrhenius inverse scale?
        # CHECK 1: Is this an Arrhenius inverse scale?
        if ("1000" in unit_clean or "10^3" in unit_clean) and "t" in unit_clean:
            if raw_t > 0:
                temp_k = 1000.0 / raw_t
                norm_temp_c = temp_k - 273.15
        
        # CHECK 1b: Implicit Arrhenius (Unit is just K-1 but values are 1000/T range)
        elif ("k-1" in unit_clean or "1/k" in unit_clean) and 0.2 < raw_t < 10.0:
             # Heuristic: 1000/T usually falls between 0.5 (2000K) and 5.0 (200K)
             # If it were really 1/T, values would be ~0.001 - 0.005
             temp_k = 1000.0 / raw_t
             norm_temp_c = temp_k - 273.15

        # CHECK 2: Standard Kelvin
        elif "k" in unit_clean and "c" not in unit_clean: 
             temp_k = raw_t
             norm_temp_c = temp_k - 273.15
             
        # CHECK 3: Standard Celsius
        elif "c" in unit_clean:
            norm_temp_c = raw_t
            temp_k = raw_t + 273.15
            
        # Fallback: Guess based on magnitude
        else:
            if raw_t > 200: # Likely Kelvin
                temp_k = raw_t
                norm_temp_c = raw_t - 273.15
            else: # Likely Celsius
                norm_temp_c = raw_t
                temp_k = raw_t + 273.15

        # --- 2. Conductivity Normalization ---
        cond_val_clean = str(cond_val).lower().strip()
        if cond_val_clean in ["n/a", "none", "unknown", "not specified"]:
            return {"cond": None, "temp": norm_temp_c}

        raw_c = safe_float(cond_val)
        cond_u_clean = cond_unit.lower().strip()
        norm_cond = None

        if "log" in cond_u_clean:
            # Case A: log(Sigma * T)
            if ("t" in cond_u_clean) and temp_k:
                sigma_times_t = 10 ** raw_c
                norm_cond = sigma_times_t / temp_k
            # Case B: just log(Sigma)
            else:
                norm_cond = 10 ** raw_c
            
        elif "ln" in cond_u_clean:
            import math
            # Case A: ln(Sigma * T)
            if ("t" in cond_u_clean) and temp_k:
                sigma_times_t = math.exp(raw_c)
                norm_cond = sigma_times_t / temp_k
            # Case B: ln(Sigma)
            else:
                 norm_cond = math.exp(raw_c)
            
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

        return {"cond": norm_cond, "temp": round(norm_temp_c, 2) if norm_temp_c is not None else None}

    except Exception as e:
        # Only log if it's not a known non-numeric string that somehow got through
        if "could not convert" in str(e).lower() and any(x in str(e) for x in ["'N/A'", "'Arrhenius plot'"]):
            pass
        else:
            print(f"Norm Error: {e}")
        return {"cond": None, "temp": None}

    # --- 3. Activation Energy Filter ---
    # Convert extracted unit to lower case for check
    cond_u_clean_final = cond_unit.lower().strip()
    if any(x in cond_u_clean_final for x in ['ev', 'kj', 'joule', 'mol']):
        return {"cond": None, "temp": None} # discard activation energy

    return {"cond": norm_cond, "temp": round(norm_temp_c, 2) if norm_temp_c is not None else None}

# ==============================================================================
# 4. Canonicalizer (Solves Problem 3: Useless Names)
# ==============================================================================
async def canonicalize_materials(client, measurements: List[MeasuredPoint], definitions: List[str], model_name: str = None):
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
    I have a list of abbreviated material names extracted from figures (e.g., "x=0.1", "Square", "Series 1").
    I have a list of Material Definitions found in the paper text.

    Your Task: Map the abbreviated names to their Full Canonical Chemical Formulas.

    DEFINITIONS FOUND IN TEXT:
    {context_str}

    ITEMS TO RESOLVE:
    {items_str}

    Logic:
    1. **Formula Calculation**: If text says "series Li(4-2x)MgxTi(5-x)/3O4" and item is "x=0.1", calculate the formula:
       Li(4-0.2)Mg(0.1)Ti(1.63)O4 -> Li3.8Mg0.1Ti1.63O4.
    2. **Legend/Symbol Mapping**: 
       - If item is "Square", "Triangle", etc., LOOK for text in the Source Caption or Definitions like "Squares represent In-doped samples".
       - Example: "Squares" -> In -> Li...In...
    3. **General Fallback**: If exact calculation isn't possible, return the General Formula with the specific variable.

    Return JSON: {{ "mappings": {{ "ID": "Canonical Formula" }} }}
    """
    
    try:
        response = await client.aio.models.generate_content(
            model=model_name or "gemini-3-flash-preview", 
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

def _map_sf_to_measurements(sf_result: Dict[str, Any], fig_id: str = None, caption: str = None) -> List[MeasuredPoint]:
    """Helper to map SciFigureParser output to MeasuredPoint list."""
    measurements = []

    # Handle the new "data_series" key
    all_series = sf_result.get('data_series', [])
    axis_meta = sf_result.get('axis_metadata', {})

    for series in all_series:
        label = series['series_label']
        axis_key = series['mapped_y_axis'] # left or right

        # Get units from the injected metadata
        y_unit = "Unknown"
        if axis_meta.get(axis_key):
             y_unit = axis_meta[axis_key].get('unit')

        x_vals = series['x_values']
        y_vals = series['y_values']


        
    # Try to extract a fixed temperature from the caption if it's a stoichiometry plot
    fixed_temp = "Not Specified"
    fixed_temp_unit = "Celsius"
    if caption:
        # Heuristic: search for "room temperature", "RT", "298 K", "25 °C"
        cap_lower = caption.lower()
        if "room temperature" in cap_lower or " rt " in cap_lower or " at rt" in cap_lower:
            fixed_temp = "25"
            fixed_temp_unit = "Celsius"
        elif "298 k" in cap_lower:
            fixed_temp = "298"
            fixed_temp_unit = "K"
        elif "25 °c" in cap_lower or "25 c" in cap_lower:
            fixed_temp = "25"
            fixed_temp_unit = "Celsius"

    if should_extract:
        x_axis_type = sf_result.get("xAxis", {}).get("axisType", "temperature")
        y_axes = sf_result.get("yAxes", [])
        
        # Backward compatibility for single yAxis
        if not y_axes and "yAxis" in sf_result:
            y_axes = [sf_result["yAxis"]]

        for dp in sf_result.get("dataPoints", []):
            raw_comp = dp.get("label", "Unknown")
            raw_temp = str(dp.get("xValue"))
            
            # Use X-axis unit as default temp unit
            raw_temp_unit = sf_result.get("xAxis", {}).get("unit", "Celsius")
            
            # Get Y-axis info from yAxisIndex
            y_idx = dp.get("yAxisIndex", 0)
            if 0 <= y_idx < len(y_axes):
                target_y_axis = y_axes[y_idx]
                raw_cond_unit = target_y_axis.get("unit", "S/cm")
                # Prepend the Y-axis label to help filter if it's activation energy
                y_label = target_y_axis.get("label", "").lower()
                if any(x in y_label for x in ["activation", "energy", "ea"]):
                     # If the axis itself is labeled as activation energy, 
                     # we should ensure the unit reflects that so it gets filtered out
                     if "ev" not in raw_cond_unit.lower() and "kj" not in raw_cond_unit.lower():
                          raw_cond_unit = f"{raw_cond_unit} (Activation Energy)"
            else:
                raw_cond_unit = "S/cm"

            if x_axis_type == "stoichiometry":
                # Special handling for stoichiometry axes:
                # 1. The xValue is actually part of the composition (x=...)
                # 2. The temperature is likely fixed in the caption
                stoich_val = str(dp.get("xValue"))
                if "x=" not in raw_comp.lower() and "x =" not in raw_comp.lower():
                    raw_comp = f"{raw_comp} (x={stoich_val})"
                
                raw_temp = fixed_temp
                raw_temp_unit = fixed_temp_unit

            m = MeasuredPoint(
                raw_composition=raw_comp,
                raw_conductivity=str(dp.get("yValue")),
                raw_conductivity_unit=raw_cond_unit,
                raw_temperature=raw_temp,
                raw_temperature_unit=raw_temp_unit,
                source="figure",
                source_figure_id=fig_id,
                source_caption=caption,
                confidence="high"
            )
            measurements.append(m)
    return measurements




from dataclasses import dataclass, asdict
from typing import List, Optional
import numpy as np

@dataclass
class MeasurementSeries:
    """
    Represents a full curve (e.g., 'x=0.1') extracted from a plot.
    Keeps data grouped for cleaner logging and analysis.
    """
    series_label: str
    
    # Vectorized Data (Standard Python floats for JSON compatibility)
    temperature_c: List[float]
    conductivity_s_cm: List[float]
    
    # Metadata
    confidence: str
    warnings: List[str]
    source_figure: str
    
    def to_dict(self):
        """Helper to ensure safe serialization (handles numpy types)"""
        return asdict(self)

class MeasurementProcessor:
    def __init__(self):
        # Physical Bounds
        self.MAX_REALISTIC_COND_RT = 0.5  # S/cm (Liquid electrolytes ~0.01-0.1, Solids rarely >0.05)
        self.MIN_REALISTIC_COND = 1e-12   # S/cm
    
    def process_extraction(self, sf_result: Dict[str, Any], fig_id: str, context: str) -> List[MeasuredPoint]:
        """
        Main entry point: Flattens LLM output, normalizes units, and applies guardrails.
        """
        measurements = []
        
        # 1. Unpack the Vectorized Data
        all_raw_series = sf_result.get('data_series', [])
        axis_meta = sf_result.get('axis_metadata', {})
        
        for raw_s in all_raw_series:
            # Get axis metadata for this specific series
            axis_key = raw_s.get('mapped_y_axis', 'left')
            y_axis_def = axis_meta.get(axis_key, {})
            x_axis_def = axis_meta.get('x_axis', {})
            
            x_vals = raw_s.get('x_values', [])
            y_vals = raw_s.get('y_values', [])
            label = raw_s.get('series_label', 'Unknown')

            # 2. Physics Check: Arrhenius Slope
            # If X is 1000/T, Y (log sigma) should DECREASE as X INCREASES.
            # (Because higher 1000/T = colder temp = lower conductivity)
            slope_warning = self._check_arrhenius_slope(x_vals, y_vals, x_axis_def)
            if slope_warning: warnings.append(slope_warning)

            for x, y in zip(x_vals, y_vals):
                warnings = []
                if slope_warning: warnings.append(slope_warning)

                # 3. Normalize Temperature
                temp_c = self._normalize_temperature(x, x_axis_def)
                if temp_c is None:
                    warnings.append("Temperature normalization failed")
                    temp_c = 25.0 # Fallback to RT if unknown, but flagged

                # 4. Normalize Conductivity
                try:
                    cond_s_cm = self._normalize_conductivity(y, y_axis_def)
                except Exception as e:
                    cond_s_cm = y
                    warnings.append("Conductivity normalization failed")
                    print(f"⚠️ Conductivity normalization failed for {y} >> error: {e}")
                
                # 5. Physical Bound Checks
                if cond_s_cm > self.MAX_REALISTIC_COND_RT and temp_c < 100:
                    warnings.append(f"Suspiciously high conductivity ({cond_s_cm:.4f} S/cm) for solid state")
                
                if cond_s_cm < self.MIN_REALISTIC_COND:
                    warnings.append(f"Value below realistic detection limit ({cond_s_cm:.2e})")

                # 6. Create Record
                # meas = Measurement(
                #     raw_composition=label,
                #     temperature_c=round(temp_c, 2),
                #     conductivity_s_cm=cond_s_cm,
                #     confidence="low" if warnings else "high",
                #     warnings=warnings,
                #     source_figure=fig_id
                # )

                meas = MeasuredPoint(
                    raw_composition=label,
                    raw_conductivity=str(cond_s_cm),
                    raw_conductivity_unit="S/cm",
                    raw_temperature=str(temp_c),
                    raw_temperature_unit="C",
                    source="figure",
                    source_figure_id=fig_id,
                    source_caption=context,
                    confidence="low" if warnings else "high",
                    warnings=warnings
                )
                measurements.append(meas)
                
        return measurements

    def _normalize_temperature(self, x_val: float, x_meta: Dict) -> float:
        """
        Converts X-value (1000/T, T, etc.) to Celsius.
        """
        q_type = x_meta.get('quantity_type', 'other')
        unit = str(x_meta.get('unit', '')).lower()

        try:
            # Case A: Arrhenius (1000/T)
            if q_type == 'temperature_inverse' or '1000' in unit or 'k-1' in unit:
                # T(K) = 1000 / x
                temp_k = 1000.0 / x_val
                return temp_k - 273.15
            
            # Case B: Absolute T (Kelvin)
            if 'k' in unit and '1000' not in unit:
                return x_val - 273.15
                
            # Case C: Celsius
            if 'c' in unit:
                return x_val
                
            # Case D: Stoichiometry (Not a temperature)
            if q_type == 'stoichiometry':
                return 25.0 # Assume Room Temp for compositional plots unless specified
                
        except ZeroDivisionError:
            return 25.0
            
        return 25.0

    def _normalize_conductivity(self, y_val: float, y_meta: Dict) -> float:
        """
        Converts Y-value (log(S/cm), mS/cm, etc.) to S/cm.
        """
        unit = str(y_meta.get('unit', '')).lower()
        title = str(y_meta.get('title_text', '')).lower()
        
        # 1. Handle Logarithmic Scales first
        # Often plots are ln(sigma) or log(sigma) even if the unit just says "S/cm"
        is_log = 'log' in title or 'ln' in title
        
        val = y_val
        
        # If the value is negative (e.g. -4) and unit is S/cm, it's almost certainly log10
        if is_log or (val < 0 and 's' in unit):
             # Heuristic: Is it ln (base e) or log (base 10)?
             # ln: -10 is ~4.5e-5. log10: -10 is 1e-10. 
             # Usually "log" implies base 10 in plots, "ln" implies base e.
             if 'ln' in title:
                 val = np.exp(y_val)
             else:
                 val = 10 ** y_val

        # 2. Handle Prefix Units
        if 'ms' in unit: # mS/cm -> S/cm
            val = val / 1000.0
        elif 'us' in unit or 'µs' in unit: # µS/cm -> S/cm
            val = val / 1e6
            
        return val

    def _check_arrhenius_slope(self, x_vals: List[float], y_vals: List[float], x_meta: Dict) -> Optional[str]:
        """
        Validates the physics of the slope.
        For 1000/T plots, slope must be NEGATIVE (Log Cond vs 1/T).
        """
        if len(x_vals) < 2: return None
        
        q_type = x_meta.get('quantity_type', '')
        
        # Simple linear regression slope
        slope, _ = np.polyfit(x_vals, y_vals, 1)
        
        if q_type == 'temperature_inverse':
            # As 1000/T increases (getting colder), Conductivity should decrease.
            # So slope should be NEGATIVE.
            if slope > 0:
                return "Physical Violation: Conductivity increases as Temp decreases (Positive Arrhenius slope)"
                
        return None


processor = MeasurementProcessor()

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
    # Note: sf_parser should be initialized with save_debug=False for production speed.
    if sf_parser:
        try:
            print(f"   🔍 {img_path.name} ({fig_id}): Detecting subplot...")
            # [OPTIMIZED] Using async detection
            detection_result = await sf_parser.detect_subplot_async(str(img_path), "ionic conductivity measurement")
            
            print(f"   🔍 {img_path.name} ({fig_id}): Detection result: {detection_result}")

            is_multi = detection_result.get("is_multi_panel", False)
            detections = detection_result.get("subplots", [])
            
            all_measurements = []

            if not is_multi:
                # Check if SciFigureParser detected ionic conductivity data
                print(">> DEBUG : Detection Data in single plot: ", detection_result['subplots'][0])
                if not detection_result['subplots'][0].get("contains_conductivity_data", True):
                    print(f"   \033[93m[Skip]\033[0m {img_path.name} ({fig_id}): No ionic conductivity measurements detected by SciFigureParser.")
                    return ExtractionResult(measurements=[]), None, True
                # Case 1: Single plot or no specific detections - process original image directly
                print(f"   📊 {img_path.name}: Single plot detected (or no specific subplots). Extracting directly...")
                # [OPTIMIZED] Using async extraction
                sf_result = await sf_parser.extract_data_async(str(img_path), grid_config={"enabled": True, "rows": 2, "cols": 2}, context=caption)
                clean_measurements = processor.process_extraction(sf_result, fig_id=fig_id, context=caption)
                all_measurements.extend(clean_measurements)
            else:
                # Case 2: Multi-plot - crop and extract for each detection IN PARALLEL
                print(f"   ✂️ {img_path.name}: Multi-plot detected ({len(detections)} panels). Processing each in parallel...")
                
                async def process_subplot(detection_data, idx):
                    label = detection_data.get('label', f'Panel {idx+1}')
                    if not detection_data.get('contains_conductivity_data', False):
                        print(f"   \033[93m[Skip]\033[0m {img_path.name} ({fig_id}): {label} - No ionic conductivity measurements detected in this subplot by SciFigureParser.")
                        return ExtractionResult(measurements=[]), None, True
                    print(">> DEBUG : Detection Data in multi plot (after filtering): ", detection_data)
                    # print(f"      - Processing {label}...") # Can be too noisy in parallel
                    # The detection schema uses 'title', but we want to pass a clean dict to the next step
                    def clean_axis(ax_data):
                        if not ax_data: return None
                        return {
                            "title_text": ax_data.get('title'), # Remap title -> title_text
                            "unit": ax_data.get('unit'),
                            "quantity_type": ax_data.get('quantity_type')
                        }

                    axis_hints = {
                        "x_axis": clean_axis(detection_data.get('x_axis')),
                        "left_y_axis": clean_axis(detection_data.get('left_y_axis')),
                        "right_y_axis": clean_axis(detection_data.get('right_y_axis'))
                    }
                    
                    # Create a safe suffix from the label
                    safe_label = re.sub(r'[^a-zA-Z0-9]', '_', label)
                    unique_suffix = f"_cropped_{safe_label}"
                    
                    box_list = detection_data.get('box_2d')
                    print("Passing box coords ", box_list)
                    box_dict = {
                        'ymin': box_list[0],
                        'xmin': box_list[1],
                        'ymax': box_list[2],
                        'xmax': box_list[3]
                    }
                    cropped_path = sf_parser.crop_image(str(img_path), box_dict, padding=80, suffix=unique_suffix)
                    sf_result = await sf_parser.extract_data_async(cropped_path, grid_config={"enabled": True, "rows": 2, "cols": 2}, context=caption, axis_hints=axis_hints)
                    # print(">> DEBUG : SciFigure Result: ", sf_result)
                    clean_measurements = processor.process_extraction(sf_result, fig_id=fig_id, context=caption)
                    # print(">> DEBUG : Cleaned Result: ", clean_measurements)
                    return clean_measurements

                # [OPTIMIZED] Parallel processing of subplots
                subplot_tasks = [process_subplot(box, i) for i, box in enumerate(detections)]
                subplot_results = await asyncio.gather(*subplot_tasks)
                
                # print(">> DEBUG : Subplot Results: ", subplot_results)
                for res in subplot_results:
                    if isinstance(res, list):
                        # Only extend if the item is a list of MeasuredPoint
                        all_measurements.extend(res)
                    elif hasattr(res, 'measurements'):
                        # If it returned an ExtractionResult object instead of a list
                        all_measurements.extend(res.measurements)
            
            result = ExtractionResult(measurements=all_measurements)
            
            # Write to debug log
            log_dir = FILE_DIR if FILE_DIR else img_path.parent
            with open(f"{log_dir}/results_log_v5.json", "a") as f:
                f.write(f"\n\n[SCI-FIGURE DEBUG] {img_path.name}:\n")
                f.write(f"- Detection Result: {json.dumps(detection_result, indent=2)}\n")
                f.write(f"- Combined Measurements: {len(all_measurements)}\n")

            if len(result.measurements) > 0:
                print(f"   ✓ {img_path.name} ({fig_id}): Found {len(result.measurements)} points total via SciFigureParser")
            
            return result, None, True 
            
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

    Case 4: Stoichiometry Plot (Composition vs Conductivity)
    Input: "The x-axis represents the variable x in Li1+xAlxTi2-x(PO4)3, and caption states 298 K."
    Output:
    {{
    "raw_temperature": "298",
    "raw_temperature_unit": "K",
    "raw_composition": "[Material Name] (x=0.2)"
    }}

    **Task**:
    **Step 1: Classify the image**
    Is this a:
    - [ ] Data plot with conductivity values (Arrhenius plot, stoichiometry plot, bar chart, etc.)
    - [ ] Table with conductivity measurements
    - [ ] Structural diagram / schematic / photo (NO DATA)
    
    **Step 2: Extract (ONLY if you checked the first two options)**
    If this contains conductivity data, extract measurements. 
    - CRITICAL: Detect if the X-axis is stoichiometry (e.g., 'x', 'z', 'composition').
    - If it is stoichiometry, extract the x-value and append it to 'raw_composition' (e.g. "Al (x=0.2)").
    - If the caption specifies a fixed temperature for the whole plot, use it for 'raw_temperature'.
    Otherwise return empty.

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
                    m.source_figure_id = fig_id
                    m.source_caption = caption
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


async def process_table_node(client, model, table_data: dict, max_retries: int = 3):
    """
    Extract data from a Markdown table found via regex.
    """
    prompt = f"""
    Extract ionic conductivity data points from this Markdown table.
    
    **Table Caption:** {table_data['caption']}
    
    **Table Content:**
    ```markdown
    {table_data['content']}
    ```

    **Task**:
    Extract all Ionic Conductivity measurements.
    For each:
    - raw_composition: Material name
    - raw_conductivity: Numeric value
    - raw_conductivity_unit: Unit (e.g. "S/cm", "mS cm-1")
    - raw_temperature: Value (e.g. "25", "room temperature")
    - raw_temperature_unit: Unit (e.g. "Celsius", "K")
    - source: "markdown_table"
    - confidence: "high"

    If activation energy is present, you can include it if it's the only value, but prioritize conductivity.
    If the table ONLY contains activation energy, extract that as a value but note the unit as "eV".

    Return JSON with measurements array.
    """

    last_exception = None
    log_dir = FILE_DIR 
    for attempt in range(1, max_retries+1):
        with open(f"{log_dir}/results_log_v5.json", "a") as f:
            try:
                response = await client.aio.models.generate_content(
                    model=model,
                    contents=[prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_json_schema=ExtractionResult.model_json_schema(),
                        temperature=0.0,
                    )
                )
            
                if not response.text:
                    print(f"   ⚠️  {table_data['caption']}: Empty response")
                    last_exception = "Empty response"
                    continue

                f.write(f"\n\n[TABLE DEBUG] {table_data['caption']}:\n")
                f.write(f"- Response: {response.text}\n")

                result = ExtractionResult.model_validate_json(response.text)
                
                # Tag metadata
                for m in result.measurements:
                    m.source_caption = table_data['caption']
                    m.source_figure_id = table_data['caption'].split(':')[0] if ':' in table_data['caption'] else "Table"

                if len(result.measurements) > 0:
                    print(f"   ✓ {table_data['caption']}: Found {len(result.measurements)} points")
                return result, response, True
            except Exception as e:
                last_exception = e
                await asyncio.sleep(1 * attempt)
                
    print(f"   ❌ {table_data['caption']}: Failed - {last_exception}")
    return ExtractionResult(measurements=[]), None, False


# ==============================================================================
# 6. Main Orchestrator
# ==============================================================================
async def run_pipeline(markdown_file, asset_dir, model):
    global FILE_DIR
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})

    try:
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

        # 2.1 Table De-duplication: Replace MD tables with placeholders in section contents
        if all_tables:
            print(f"   ✂️ De-duplicating {len(all_tables)} tables from Markdown text...")
            for table_info in all_tables:
                for sec in sections:
                    if table_info.content in sec.content:
                        placeholder = f"\n\n[{table_info.id}: {table_info.caption} processed separately]\n\n"
                        sec.content = sec.content.replace(table_info.content, placeholder)
                        print(f"       - Replaced '{table_info.id}' in section '{sec.title}'")

        # 2.5 Initialize SciFigureParser - [OPTIMIZED] save_debug=False for speed
        sf_parser = SciFigureParser(api_key=api_key, model_name=VISION_MODEL, debug=True, save_debug=False)

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

        # Build tasks
        tasks = []
        # Extract from text (section-by-section)
        all_measurements = []
        material_defs = [] # You might want to accumulate these or merge them later

        # We process sections in parallel batches (or sequentially if rate limits matter)
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
            # Only grab original assets, skip debug/cropped/detected files
            candidates = list(asset_dir.glob(pattern))
            for cand in candidates:
                if "debug" not in cand.name.lower() and "cropped" not in cand.name.lower() and "detected" not in cand.name.lower():
                    img_files.append(cand)
        
        for img_path in sorted(img_files):
            # Retrieve context from registry (as you already do)
            filename = img_path.name

            # this is a temporary fix
            try:
                context = image_context_registry[filename]
            except:
                context = image_context_registry.get('_'+filename, {"id": "Unknown", "caption": "No caption"})
            tasks.append(safe_image_call_with_retry(img_path, context, client, VISION_MODEL, sem, sf_parser=sf_parser))

        # Add Table Tasks
        for table_info in all_tables:
            table_data = {
                'caption': f"{table_info.id}: {table_info.caption}",
                'content': table_info.content
            }
            tasks.append(safe_table_call_with_retry(table_data, client, TEXT_MODEL, sem))

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

        # 4. Canonicalize Names (Solves Problem 3)
        # We pass the definitions found in text + the measurements from figures
        print("   ... Canonicalizing Materials...")
        print(">> DEBUG: all_measurements: ", all_measurements)
        all_measurements = await canonicalize_materials(client, all_measurements, material_defs, model_name=TEXT_MODEL)

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
    finally:
        # [NEW] Ensure the Gemini client is closed gracefully to prevent SSL/Event loop errors on exit
        if 'client' in locals() and client:
            await client.aio.aclose()

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