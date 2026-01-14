#!/usr/bin/env python3
"""
Ionic Conductivity Data Extraction from PageIndex JSON structures.

Four-stage pipeline:
1. LLM Node Identification - Identify relevant nodes using keywords + LLM filter
2. Full Extraction with Provenance - Extract data with confidence and source tracking
3. Smart Deduplication - Remove duplicates using normalized values and cross-references
4. Validation - Verify consistency across all extracted data

Usage:
    python run_extraction.py results/paper_keywords_structure.json
    python run_extraction.py results/paper_keywords_structure.json --output materials.json
"""
import threading
import concurrent.futures
import asyncio
import logging
import time
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
import re
from PIL import Image
from dotenv import load_dotenv
import base64
from google.genai import types
import math

logging.basicConfig(filename='run_extraction.log', level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

SCIENTIFIC_NORMALIZER_MODEL = "gemini-2.5-pro" # normalizes the extracted value 
DATA_VALIDATOR_MODEL = "gemini-3-flash-preview" # validates the extraction
MATERIAL_NAME_NORMALIZER_MODEL = "gemini-3-flash-preview" # normalizes the material names
EXTRACTOR_TEXT_MODEL = "gemini-3-flash-preview" # extracts text from the node
EXTRACTOR_VISION_MODEL = "gemini-3-flash-preview" # extracts text from the node
SCIENTIFIC_NORMALIZER_THINKING_LEVEL = 'low' # or 'high
# ============================================================================
# Pydantic Models for Structured Output
# ============================================================================

class ElectrolyteName(BaseModel):
    """A structured representation of the electrolyte's name."""
    full_name: str = Field(
        ...,
        description=(
            "The complete, systematic name of the electrolyte. PRIORITY ORDER:\n"
            "1. Chemical formula if present in text/figure/table (e.g., Li6PS5I, Li7La3Zr2O12, Li1.3Al0.3Ti1.7(PO4)3)\n"
            "2. Systematic IUPAC name (e.g., Lithium Phosphorus Sulfide Iodide, Lithium Lanthanum Zirconate Oxide)\n"
            "3. Standard abbreviation (e.g., LLZO, LATP, PEO)\n"
            "AVOID generic terms like 'solid electrolyte', 'argyrodite', 'sample', 'material' unless no other information exists."
        )
    )
    acronym: Optional[str] = Field(
        None,
        description=(
            "Chemical formula OR standard abbreviation if different from full_name.\n"
            "Examples: 'LLZO', 'PEO', 'Li6PS5I'. Leave empty if full_name already contains the formula."
        )
    )
    proportion: Optional[str] = Field(
        None,
        description="Stoichiometric ratios, doping levels, or concentrations (e.g., 90:10 wt%, x=0.25, 10 mol%, Li/P=1.5)."
    )


class IonicConductivityDataPoint(BaseModel):
    """Represents a single extracted data point for ionic conductivity."""
    source_sentence_id: str = Field(
        ...,
        description="The ID of the specific source sentence (e.g., '0054') where this data point was extracted."
    )
    material_class: str = Field(
        ...,
        description="Primary functional class: Ceramic, Polymer, Composite, or Other."
    )
    electrolyte_name: ElectrolyteName = Field(
        ...,
        description="Structured name of the electrolyte."
    )
    ionic_conductivity_S_per_cm: str = Field(
        ...,
        description="Ionic conductivity value including the unit (e.g., '1.2 × 10⁻⁴ S cm⁻¹', '~10⁻⁵ S cm⁻¹')."
    )
    measurement_temperature: str = Field(
        ...,
        description="Temperature of measurement (e.g., '25°C', 'RT', 'room temperature')."
    )
    # Provenance tracking
    reason: str = Field(
        ...,
        description="A brief explanation of how and why this data point was extracted."
    )
    confidence: str = Field(
        ...,
        description="Confidence level: 'high' (primary data from this study), 'medium' (clearly stated cited data), 'low' (ambiguous or inferred)."
    )
    data_source: str = Field(
        ...,
        description="Data origin: 'primary' (this paper's measurement), 'internal-citation' (from a reference figure/table/section of this paper), 'external-citation' (from another paper), 'inferred' (calculated or estimated)."
    )
    # Material details
    material_description: str = Field(
        ...,
        description="Material properties, or 'N/A (Cited Work)' if from a reference."
    )
    processing_method: str = Field(
        ...,
        description="Synthesis steps, or 'N/A (Cited Work)' if from a reference."
    )
    

class MaterialExtractionResponse(BaseModel):
    """Response containing ionic conductivity data points extracted from a node."""
    materials: List[IonicConductivityDataPoint] = Field(default_factory=list)


class NodeRelevanceResponse(BaseModel):
    """Response for node relevance check."""
    is_relevant: bool = Field(
        ...,
        description="True if this node likely contains ionic conductivity measurements with numerical values."
    )
    relevance_reason: str = Field(
        ...,
        description="Brief explanation of why this node is or isn't relevant."
    )
    expected_data_points: int = Field(
        default=0,
        description="Estimated number of ionic conductivity measurements in this node."
    )


class ValidationVerdict(BaseModel):
    """The auditor's verdict on a single data point."""
    data_point_index: int = Field(..., description="The index of the data point in the provided list.")
    text_check: bool = Field(..., description="True if the text explicitly supports this extracted value.")
    figure_check: bool = Field(..., description="True if the figure explicitly supports this extracted value.")
    
    reason: str = Field(..., description="Explanation of why it is valid or invalid.")
    double_check: str = Field(..., description="Double check the correctness ofyour answer and the original answer including the reasoning for both.")

    is_valid: bool = Field(..., description="The final verdict on the data point: True if the data point is valid, False otherwise.")
    correction_temp: Optional[str] = Field(None, description="If invalid, provide the correct value/unit temperature.")
    correction_conductivity: Optional[str] = Field(None, description="If invalid, provide the correct value/unit conductivity.")
    
class BatchValidationResponse(BaseModel):
    """Response containing verdicts for a batch of data points."""
    verdicts: List[ValidationVerdict] = Field(default_factory=list)


class CanonicalMaterialName(BaseModel):
    """Canonical representation of a material in a document."""
    canonical_formula: str = Field(
        ...,
        description="The canonical chemical formula (e.g., Li6PS5I, Li7La3Zr2O12)"
    )
    canonical_name: str = Field(
        ...,
        description="The canonical systematic name (e.g., Lithium Phosphorus Sulfide Iodide)"
    )
    abbreviation: Optional[str] = Field(
        None,
        description="Standard abbreviation if commonly used (e.g., LLZO, LATP)"
    )
    variant_names: List[str] = Field(
        default_factory=list,
        description="All name variants found in the document that refer to this material"
    )


class DocumentNameMapping(BaseModel):
    """Mapping of material names across a document."""
    materials: List[CanonicalMaterialName] = Field(default_factory=list)


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

# At module level
_global_executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)


async def _safe_llm_call_async(func, *args, retries=3, timeout=60, default=None, **kwargs):
    """
    Executes a blocking LLM call (func) inside a DETACHED thread executor.
    1. Async Timeout (to kill blocking calls that hang)
    2. Retries (to handle transient failures or timeouts)
    3. Thread Isolation (Uses a fresh executor per call so hung threads don't starve the global pool)
    """
    loop = asyncio.get_running_loop()
    # print("arguments are ", args, kwargs)
    # Extract model name for tracking (usually passed in kwargs)
    model_name = kwargs.get('model', 'unknown_model')
    # print(">>> INSIDE ASYNC MODEL NAME", model_name)
    # model_name = 'gemini-3-flash-preview'

    for attempt in range(retries):
        executor = None
        try:           
            result = await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(
                    _global_executor, 
                    lambda: func(*args, **kwargs)
                ), 
                timeout=timeout
            )
            
            if result: 
                tracker.track(result, model_name)
            return result
            
        except asyncio.TimeoutError:
            print(f"⚠️ LLM Call Timed Out (Attempt {attempt+1}/{retries})")
            # CRITICAL: Do NOT join/wait for the executor, because the thread is stuck.
            # We effectively 'leak' this executor/thread pair until process exit.
            
        except Exception as e:
            print(f"⚠️ LLM Call Failed: {e} (Attempt {attempt+1}/{retries})")
            if executor:
                executor.shutdown(wait=False)
        
        # Backoff before retry (unless it's the last attempt)
        if attempt < retries - 1:
            await asyncio.sleep(1 * (attempt + 1))
            
    print("❌ All LLM retries failed.")
    return default
# Define the tool for the model
def normalize_scientific_data(
    conductivity_value: float, 
    conductivity_unit: str, 
    temperature_value: float, 
    temperature_unit: str
):
    """Call this to normalize ionic conductivity and temperature values."""
    # This is a dummy for the LLM to see the signature
    pass

# def calculate_standard_units(
#     cond_value: float, 
#     cond_unit: str, 
#     temp_value: Optional[float], 
#     temp_unit: Optional[str]
# ) -> dict:
#     """Deterministic conversion logic."""
#     # Unit mapping to S/cm
#     # 1 S/m = 0.01 S/cm
#     # 1 mS/cm = 0.001 S/cm
#     unit_multipliers = {
#         "s/cm": 1.0, "s·cm⁻¹": 1.0, "scm-1": 1.0,
#         "ms/cm": 1e-3, "ms·cm⁻¹": 1e-3,
#         "μs/cm": 1e-6, "us/cm": 1e-6, "µs/cm": 1e-6,
#         "s/m": 0.01, "s·m⁻¹": 0.01
#     }
    
#     # Normalize conductivity
#     multiplier = unit_multipliers.get(cond_unit.lower().strip(), 1.0)
#     norm_cond = cond_value * multiplier
    
#     # Normalize temperature to Celsius
#     norm_temp = temp_value
#     if temp_unit:
#         u = temp_unit.upper().strip()
#         if u == "K":
#             norm_temp = temp_value - 273.15
#         elif u in ["RT", "ROOM", "ROOM TEMPERATURE"]:
#             norm_temp = 25.0
            
#     return {"norm_cond": norm_cond, "norm_temp": norm_temp}

# def calculate_standard_units(
#     cond_value: float, 
#     cond_unit: str, 
#     temp_value: Optional[float] = None, 
#     temp_unit: Optional[str] = None
# ) -> dict:
#     """Robust conversion logic using Regex instead of exact string matching."""
    
#     # --- 1. Robust Conductivity Normalization ---
#     # Normalize string: lowercase, remove spaces, dots, and '1'
#     # "mS cm-1" -> "mscm-"
#     # "S/cm"    -> "s/cm"
#     u_clean = cond_unit.lower().replace(" ", "").replace("·", "").replace(".", "")
    
#     multiplier = 1.0
    
#     # Detect Prefix
#     if "ms" in u_clean:          # Milli
#         multiplier = 1e-3
#     elif "us" in u_clean or "μs" in u_clean or "µs" in u_clean: # Micro
#         multiplier = 1e-6
#     elif "ns" in u_clean:        # Nano
#         multiplier = 1e-9
#     elif "s" in u_clean:         # Siemens (Base)
#         multiplier = 1.0
        
#     # Detect Geometry (m vs cm)
#     # Standard is S/cm. If unit is /m, we need to divide by 100.
#     # 1 S/m = 0.01 S/cm
#     if "m" in u_clean and "cm" not in u_clean and "mm" not in u_clean:
#          # precise check to ensure it's meters, not milli-something else
#          # This is tricky, usually safe to assume S/cm unless explicitly S/m
#          if u_clean.endswith("/m") or "m-1" in u_clean:
#              multiplier *= 0.01

#     norm_cond = cond_value * multiplier
    
#     # --- 2. Robust Temperature Normalization ---
#     norm_temp = temp_value
#     if temp_unit:
#         tu_clean = temp_unit.lower().strip()
#         if "k" in tu_clean:
#             norm_temp = temp_value - 273.15
#         elif "f" in tu_clean: # F to C
#              norm_temp = (temp_value - 32) * 5/9
#         elif "rt" in tu_clean or "room" in tu_clean:
#             norm_temp = 25.0
            
#     return {"norm_cond": norm_cond, "norm_temp": norm_temp}

def calculate_standard_units(
    cond_value: float, 
    cond_unit: str, 
    temp_value: Optional[float] = None, 
    temp_unit: Optional[str] = None
) -> dict:
    
    # --- 1. Robust Conductivity Normalization ---
    # Clean string: lowercase, remove spaces/dots/weird chars
    # "mS cm-1" -> "mscm-1"
    if not cond_unit:
        return {"norm_cond": None, "norm_temp": None}

    u_clean = cond_unit.lower().replace(" ", "").replace("·", "").replace(".", "")
    
    # Base Multiplier
    multiplier = 1.0
    
    # Determine Metric Prefix (Order matters! Check longest first)
    if "ms" in u_clean:          # Milli (10^-3)
        multiplier = 1e-3
    elif "us" in u_clean or "μs" in u_clean or "µs" in u_clean: # Micro (10^-6)
        multiplier = 1e-6
    elif "ns" in u_clean:        # Nano (10^-9)
        multiplier = 1e-9
    elif "ks" in u_clean:        # Kilo (10^3) - Added this just in case
        multiplier = 1000.0
    elif "s" in u_clean:         # Base Siemens
        multiplier = 1.0
        
    # Determine Geometry (cm vs m)
    # Target is S/cm.
    # If unit is S/m, we must divide by 100 (1 S/m = 0.01 S/cm)
    # We look for explicit meter indicators WITHOUT centi markers
    if "m" in u_clean and "cm" not in u_clean and "mm" not in u_clean:
        # Check for inverse meters (m-1) or per meter (/m)
        if "m-1" in u_clean or "/m" in u_clean:
             multiplier *= 0.01

    norm_cond = cond_value * multiplier
    
    # --- 2. Robust Temperature Normalization ---
    norm_temp = temp_value
    
    # Handle the case where LLM passes "RT" as a unit (failsafe)
    # or converts per our new prompt instructions
    if temp_unit:
        tu_clean = temp_unit.lower().strip()
        
        # Kelvin
        if "k" in tu_clean:
            # Sanity check: If value is small (<100), it might be C labeled as K error, 
            # but usually we trust the unit.
            if norm_temp is not None:
                norm_temp = norm_temp - 273.15
                
        # Fahrenheit
        elif "f" in tu_clean:
            if norm_temp is not None:
                norm_temp = (norm_temp - 32) * 5/9
        
        # Failsafe for text residuals
        elif "rt" in tu_clean or "room" in tu_clean:
            norm_temp = 25.0

    # Final Failsafe: If temp is None but unit implies RT
    if norm_temp is None and temp_unit and ("rt" in temp_unit.lower() or "room" in temp_unit.lower()):
        norm_temp = 25.0
            
    return {"norm_cond": norm_cond, "norm_temp": norm_temp}

class ScientificNormalizer:
    def __init__(self, client, model_name=SCIENTIFIC_NORMALIZER_MODEL):
        self.client = client
        self.model_name = model_name
        # Define the tool signature for the LLM
        self.tools = [calculate_standard_units]

    async def normalize_batch(self, materials: List[dict]):
        print(f"  → Normalizing {len(materials)} data points [CONCURRENT, MAX 7]...")
        sem = asyncio.Semaphore(7)

        async def _norm_item(mat):
            # Skip empty or clearly invalid data to save tokens
            cond_raw = mat.get('ionic_conductivity_S_per_cm')
            temp_raw = mat.get('measurement_temperature')

            if not cond_raw or not temp_raw:
                # Debug logging if needed, or silent skip
                return

            prompt = (
                f"Extract numeric values and units for the following:\n"
                f"Conductivity: {cond_raw}\n"
                f"Temperature: {temp_raw}\n\n"
                f"RULES:\n"
                f"1. Conductivity: If no unit is written (e.g. '10^-4'), ASSUME 'S/cm'.\n"
                f"2. Temperature: If the text says 'RT', 'Room Temperature', 'Ambient', or similar, "
                f"YOU MUST set temp_value=25 and temp_unit='C'.\n"
                f"3. Do not omit the temperature if it is 'RT'. Convert it."
            )
            
            async with sem:
                try:
                    # Force function calling using safe wrapper
                    # Calls self.client.models.generate_content
                    # IMPORTANT: _safe_llm_call_async expects a function and its args
                    response = await _safe_llm_call_async(
                        self.client.models.generate_content,
                        model=self.model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            thinking_config=types.ThinkingConfig(thinking_level=SCIENTIFIC_NORMALIZER_THINKING_LEVEL) if self.model_name == 'gemini-3-flash-preview' else None,
                            tools=self.tools,
                            tool_config=types.ToolConfig(
                                function_calling_config=types.FunctionCallingConfig(
                                    mode="ANY"
                                )
                            )
                        )
                    )
                    
                    if not response or not response.candidates:
                        # Warning suppressed/handled
                        return

                    # check if content exists before accessing parts
                    first_candidate = response.candidates[0]
                    if not first_candidate.content or not first_candidate.content.parts:
                        print(f"    Warning: Empty content for {mat.get('electrolyte_name', 'Unknown')} (Finish Reason: {first_candidate.finish_reason})")
                        return

                    # Execute the tool call in Python
                    for part in first_candidate.content.parts:
                        if part.function_call:
                            # Extract args from Gemini and pass to our Python function
                            args = part.function_call.args

                            # DEBUG LOGGING -------------------------------------
                            # Un-comment this to see exactly which inputs are failing
                            print(f"    [DEBUG Norm] Input: Cond='{cond_raw}', Temp='{temp_raw}'")
                            # -------------------------------------------------------
                            print(f"    [DEBUG Norm] Tool Args: {args}")
                            # ---------------------------------------------------
                            results = calculate_standard_units(**args)
                            
                            mat['_norm_cond'] = results['norm_cond']
                            mat['_norm_temp'] = results['norm_temp']
                except Exception as e:
                    error_msg = f"    Warning: Could not normalize {mat.get('electrolyte_name')}: {e}"
                    logging.warning(error_msg)
                    mat['_norm_error'] = str(e)
                    mat['_norm_cond'] = None
                    mat['_norm_temp'] = None

        # Create tasks for all items
        tasks = [_norm_item(mat) for mat in materials]
        await asyncio.gather(*tasks)

        return materials

class MaterialNameNormalizer:
    """
    Normalizes material names across a document using LLM-based analysis.
    Resolves cases where the same material is referred to by different names.
    """
    
    def __init__(self, client, model_name=MATERIAL_NAME_NORMALIZER_MODEL):
        self.client = client
        self.model_name = model_name
    
    async def normalize_document_names(
        self, 
        materials: List[dict], 
        doc_title: str = "Unknown Document"
    ) -> List[dict]:
        """
        Normalize all material names in a document to canonical forms.
        
        Args:
            materials: List of extracted material dictionaries
            doc_title: Title of the source document for context
            
        Returns:
            Updated materials list with normalized names
        """
        if not materials:
            return materials
        
        print(f"\n[Stage 2.5] Normalizing material names across document...")
        
        # 1. Collect all unique material names WITH their proportions
        # We need to track name+proportion pairs to avoid over-grouping
        name_variants = set()
        name_with_proportion = []  # Track full context
        
        for mat in materials:
            full_name = mat.get('electrolyte_name', {}).get('full_name', '')
            acronym = mat.get('electrolyte_name', {}).get('acronym', '')
            proportion = mat.get('electrolyte_name', {}).get('proportion', '')
            
            if full_name:
                name_variants.add(full_name)
                # Store the combination for context
                if proportion:
                    name_with_proportion.append(f"{full_name} ({proportion})")
                else:
                    name_with_proportion.append(full_name)
            if acronym:
                name_variants.add(acronym)
        
        if len(name_variants) <= 1:
            print(f"  → Only {len(name_variants)} unique name(s), skipping normalization")
            return materials
        
        print(f"  → Found {len(name_variants)} unique name variants")
        
        # 2. Ask LLM to create canonical mapping
        # Show unique combinations to give LLM full context
        unique_combinations = sorted(set(name_with_proportion))
        
        prompt = f"""Analyze these material names from a scientific paper and create canonical mappings.

Paper Title: {doc_title}

Material Names Found (with proportions where applicable):
{chr(10).join(f'- "{name}"' for name in unique_combinations)}

Task:
1. Group names that refer to the SAME EXACT material (same composition AND same proportion)
2. For each group, determine:
   - canonical_formula: The chemical formula (e.g., Li6PS5I, Li7La3Zr2O12)
   - canonical_name: Systematic name (e.g., Lithium Phosphorus Sulfide Iodide)
   - abbreviation: Standard abbreviation if any (e.g., LLZO, LATP)
   - variant_names: List of ALL names from the input that refer to this material

CRITICAL Guidelines:
- Prioritize chemical formulas over generic names
- "Li6PS5I" and "lithium argyrodite" likely refer to the same material → GROUP THEM
- "Li7La3Zr2O12" and "LLZO" refer to the same material → GROUP THEM
- Different stoichiometries (e.g., Li6PS5Br vs Li6PS5I) are DIFFERENT materials → DO NOT GROUP
- Different proportions (e.g., x=0.1 vs x=0.35) are DIFFERENT materials → DO NOT GROUP
- "Li6PS5Br (x=0.1)" and "Li6PS5Br (x=0.35)" are DIFFERENT → DO NOT GROUP
- Only group if the name is just a synonym (e.g., formula vs generic name) for the SAME composition

Respond with JSON only."""
        
        try:
            print(" >>>> CANNONICAL PROMPT >>>>", prompt)
            from google.genai import types
            
            response = await _safe_llm_call_async(
                self.client.models.generate_content,
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_level=SCIENTIFIC_NORMALIZER_THINKING_LEVEL) if self.model_name == 'gemini-3-flash-preview' else None,
                    temperature=0.2,
                    max_output_tokens=4096,
                    response_mime_type="application/json",
                    response_json_schema=DocumentNameMapping.model_json_schema()
                )
            )
            
            if not response or not response.text:
                print("  → LLM normalization failed, keeping original names")
                return materials
            
            mapping = DocumentNameMapping.model_validate_json(response.text)
            print(" >>>> CANNONICAL RESPONSE >>>>", mapping)
            # 3. Build lookup dictionary: variant_name -> canonical_material
            variant_to_canonical = {}
            for canonical_mat in mapping.materials:
                for variant in canonical_mat.variant_names:
                    variant_to_canonical[variant.strip().lower()] = canonical_mat
            
            print(f"  → Created {len(mapping.materials)} canonical material(s)")
            
            # 4. Apply mapping to all materials
            updated_count = 0
            for mat in materials:
                original_full = mat.get('electrolyte_name', {}).get('full_name', '')
                original_acronym = mat.get('electrolyte_name', {}).get('acronym', '')
                original_proportion = mat.get('electrolyte_name', {}).get('proportion', '')
                
                # Build lookup key with proportion if it exists
                if original_proportion:
                    lookup_key_full = f"{original_full} ({original_proportion})".strip().lower()
                else:
                    lookup_key_full = original_full.strip().lower()
                
                # Try to find canonical mapping
                canonical = None
                if lookup_key_full in variant_to_canonical:
                    canonical = variant_to_canonical[lookup_key_full]
                elif original_full.strip().lower() in variant_to_canonical:
                    # Fallback: try without proportion
                    canonical = variant_to_canonical[original_full.strip().lower()]
                elif original_acronym and original_acronym.strip().lower() in variant_to_canonical:
                    canonical = variant_to_canonical[original_acronym.strip().lower()]
                
                if canonical:
                    # Update with canonical names
                    mat['electrolyte_name']['full_name'] = canonical.canonical_formula or canonical.canonical_name
                    mat['electrolyte_name']['acronym'] = canonical.abbreviation
                    mat['canonical_formula'] = canonical.canonical_formula
                    # Store original name for reference
                    mat['_original_name'] = original_full
                    updated_count += 1
            
            print(f"  → Updated {updated_count}/{len(materials)} material names")
            
        except Exception as e:
            print(f"  → Name normalization error: {e}")
            print("  → Keeping original names")
        
        return materials


# ============================================================================
# Stage 4: LLM-as-a-Judge for Validation
# ============================================================================

class DataValidator:
    """
    Validates extracted materials using a hybrid approach:
    1. Heuristic Checks: Fast physical bounds and sanity checks.
    2. LLM Semantic Verification: "Auditor" model checks extraction against source text.
    """
    def __init__(self, client, figure_index: Dict[str, List[Path]], model_name: str = DATA_VALIDATOR_MODEL):
        self.client = client
        self.model_name = model_name
        self.figure_index = figure_index

    def validate_all(self, materials: List[dict], nodes_map: Dict[str, dict]) -> List[dict]:
        """Main entry point: Runs heuristics, then runs LLM verification on survivors."""
        
        # 1. Run Heuristics (Fast, CPU-bound)
        print(f"  → Running physical heuristic checks on {len(materials)} points...")
        for i, mat in enumerate(materials):
            mat['_index'] = i  # Tag with ID for tracking
            mat = self._check_physics_and_metadata(mat)

        # 2. Group by Source Node for Batched LLM Verification
        # We only want to validate points that seem physically plausible but might be hallucinations
        to_verify = [m for m in materials if m.get('_validation', {}).get('is_valid', True)]
        
        grouped_by_node = {}
        for mat in to_verify:
            node_id = mat.get('source_node', {}).get('node_id')
            if node_id:
                if node_id not in grouped_by_node:
                    grouped_by_node[node_id] = []
                grouped_by_node[node_id].append(mat)

        # 3. Run LLM Verification (Slow, IO-bound)
        # In production, you would run this with asyncio.gather
        print(f"  → Running LLM semantic verification on {len(to_verify)} points...")
        if grouped_by_node:
            try:
                asyncio.run(self._batch_verify_with_llm(grouped_by_node, nodes_map))
            except Exception as e:
                print(f"    ! Warning: LLM verification failed: {e}")

        return materials

    def _check_physics_and_metadata(self, mat: dict) -> dict:
        """Run standard rule-based checks."""
        issues = []
        
        # Physics Checks (using pre-normalized values if available)
        cond = mat.get('_norm_cond')
        if cond and (cond >= 1.0 or cond < 1e-12):
            issues.append(f"Physical Improbability: Conductivity {cond:.2e} S/cm is outside typical bounds. (cond >= 1.0 or cond < 1e-12)")

        temp = mat.get('_norm_temp')
        if temp and (temp < -50 or temp > 1000):
            issues.append(f"Physical Improbability: Temperature {temp}°C is outside typical bounds. (temp < -50 or temp > 1000)")

        # Initialize validation state
        mat['_validation'] = {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'audited_by_llm': False
        }
        return mat

    async def _batch_verify_with_llm(self, groups: Dict[str, List[dict]], nodes_map: Dict[str, dict]):
        """
        Asynchronously verifies batches of data points against their source text.
        """
        sem = asyncio.Semaphore(7)
        tasks = []
        for node_id, batch in groups.items():
            # Ensure node exists and is a dict
            node = nodes_map.get(node_id)
            if not node:
                print(f"    Warning: Node ID {node_id} not found in map, skipping verification.")
                print(f"    FYI: {node}")
                continue

            if isinstance(node, list):
                # Handle edge case where node might be wrapped in a list
                node = node[0]
                print(f"    FYI: {node}")
            if node:
                tasks.append(self._verify_node_batch(node, batch, sem))
        
        await asyncio.gather(*tasks)

    async def _verify_node_batch(self, node: dict, batch: List[dict], sem: asyncio.Semaphore):
        """Ask LLM to audit a specific list of claims against a text WITH images attached."""
        from google.genai import types
        

        


        # Prepare the context
        claims_text = ""
        for item in batch:
            claims_text += (
                f"ID {item['_index']}: "
                f"Reason: {item['reason']}\n"
                f"Material='{item['electrolyte_name']['full_name']}', "
                f"Conductivity='{item['ionic_conductivity_S_per_cm']}' and Normalized Conductivity='{item['_norm_cond']}', "
                f"Temp='{item['measurement_temperature']}' and Normalized Temperature='{item['_norm_temp']}\n"
            )

        # detect if it's image node
        is_image_node = node.get('section') == 'Image-Only Extraction' or not node.get('text', '').strip()
        
        text_content = node.get('text', '')
        if is_image_node:
            text_content = f"[IMAGE CONTEXT]: The data was extracted exclusively from the attached image (Node ID: {node.get('node_id')}). Verify the claims by visually inspecting the image."
        # 2. RESOLVE IMAGES (Re-using logic from Extractor)
        # Regex to find "Figure 4", "Fig. 2", etc.
        refs = re.findall(r'\b(?:[Ff]ig\.?|[Ff]igure)\s*([A-Za-z]?\d+)', text_content, re.IGNORECASE)
        
        relevant_image_parts = []
        seen_paths = set()
        
        # Look up paths in the figure_index we passed in __init__
        for fig_num in refs:
            if fig_num in self.figure_index:
                for path in self.figure_index[fig_num]:
                    if path not in seen_paths:
                        try:
                            # Create the image part for Gemini
                            image_bytes = path.read_bytes()
                            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
                            mime = "image/png" if path.suffix.lower() == '.png' else "image/jpeg"
                            
                            relevant_image_parts.append(types.Part(
                                inline_data=types.Blob(mime_type=mime, data=image_b64),
                                media_resolution={"level": "media_resolution_high"} # Critical for reading charts
                            ))
                            seen_paths.add(path)
                        except Exception as e:
                            print(f"    Auditor Warning: Could not load {path}: {e}")
        
        prompt_text = f"""You are a Skeptical Scientific Data Auditor. Verify these extracted values against the provided text.

SOURCE TEXT:
"{text_content}"

CLAIMS TO VERIFY:
{claims_text}

INSTRUCTIONS:
1. If the Source Context indicates an image-only extraction, rely ENTIRELY on the attached image.
2. If text is present, check if the text *explicitly* supports the value OR refers to a figure that supports it.
3. If the Extractor said "not specified" but the Figure clearly shows a value, mark it as Invalid and provide the CORRECTION.

Respond with JSON."""
        
        # Combine text and images
        contents_payload = [types.Content(parts=[types.Part(text=prompt_text)] + relevant_image_parts)]

        async with sem:
            try:
                # Use safe wrapper instead of raw asyncio.to_thread
                response = await _safe_llm_call_async(
                    self.client.models.generate_content,
                    model=self.model_name,
                    contents=contents_payload,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_level=SCIENTIFIC_NORMALIZER_THINKING_LEVEL) if self.model_name == 'gemini-3-flash-preview' else None,
                        response_mime_type="application/json",
                        response_json_schema=BatchValidationResponse.model_json_schema()
                    )
                )
                print('OG Text>>>', text_content)
                print('OG Claims>>>', claims_text)
                print('Response>>>', response.text)
                print('\n\n\n')
                if not response or not response.text:
                    return
            
                # Parse and Apply Verdicts
                result = BatchValidationResponse.model_validate_json(response.text)
                
                # Map verdicts back to the original objects
                batch_map = {m['_index']: m for m in batch}
                
                for verdict in result.verdicts:
                    if verdict.data_point_index in batch_map:
                        mat = batch_map[verdict.data_point_index]
                        mat['_validation']['audited_by_llm'] = True
                        mat['_validation']['double_check'] = verdict.double_check
                        if not verdict.is_valid:
                            mat['_validation']['is_valid'] = False
                            mat['_validation']['issues'].append(f"LLM Audit Failed: {verdict.reason}")

                            correction_applied = False
                            if verdict.correction_temp:
                                print(f"  -> Applying Auditor Correction: {mat['measurement_temperature']} -> {verdict.correction_temp}")
                                mat['_validation']['old_measurement_temperature'] = mat['measurement_temperature']
                                mat['measurement_temperature'] = verdict.correction_temp
                                correction_applied = True
                            if verdict.correction_conductivity:
                                print(f"  -> Applying Auditor Correction: {mat['ionic_conductivity_S_per_cm']} -> {verdict.correction_conductivity}")
                                mat['_validation']['old_ionic_conductivity_S_per_cm'] = mat['ionic_conductivity_S_per_cm']
                                mat['ionic_conductivity_S_per_cm'] = verdict.correction_conductivity
                                correction_applied = True
                            if correction_applied:
                                mat['_validation']['is_valid'] = True
                                mat['_validation']['issues'].append("\n AUTO-CORRECTED by Auditor")
                        else:
                            mat['_validation']['is_valid'] = True
                            mat['_validation']['issues'].append(f"LLM Audit Success: {verdict.reason}")
                            correction_applied = False
            except Exception as e:
                print(f"    Validation Error on Node {node.get('node_id')}: {e}")


# ============================================================================
# Material Extractor - Four Stage Pipeline
# ============================================================================

class MaterialExtractor:
    """
    Extracts ionic conductivity data from PageIndex tree structures.
    
    Four-stage pipeline:
    1. LLM Node Identification
    2. Full Extraction with Provenance
    3. Smart Deduplication
    4. Validation
    """
    
    def __init__(self, model_text: str = "gemini-3-flash-preview", model_vision: str = 'gemini-3-flash-preview'):
        self.model_text = model_text
        self.model_vision = model_vision
        
        # Initialize Gemini client
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        self.client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})

        self.figure_index = {} # stores mapping: "2" -> Path/to/Figure2.jpg

    # ========================================================================
    # Pre-Processing: Build Figure Index
    # ========================================================================
    
    def _index_figures(self, base_path: Path, structure: List[dict]):
        """
        Scans document for images and maps Figure Number -> LIST of Image Paths.
        Handles cases where Figure 2 is split into Figure 2a/2b or detected multiple times.
        """
        print("  Indexing figures...")
        # Reset index to store lists: {'2': [PathA, PathB]}
        self.figure_index: Dict[str, List[Path]] = {} 
        
        def traverse_for_images(nodes):
            for node in nodes:
                # 1. Check if this is an Image Node
                if node.get('node_type') == 'image':
                    src = node.get('src', '')
                    if src:
                        # Construct full path
                        full_path = base_path / src
                        
                        # Extract Figure Number from filename
                        # Matches: "Figure_4", "Figure4", "Fig_2", etc.
                        # Adjust regex if your filenames vary (e.g. "image-05.jpg")
                        fig_match = re.search(r'(?:Figure|Fig)[-_]?(\d+)', src, re.IGNORECASE)
                        
                        if fig_match and full_path.exists():
                            fig_num = fig_match.group(1) # e.g., "4"
                            
                            if fig_num not in self.figure_index:
                                self.figure_index[fig_num] = []
                            
                            if full_path not in self.figure_index[fig_num]:
                                self.figure_index[fig_num].append(full_path)
                                print(f"    Indexed Figure {fig_num} -> {src}")
                        elif not full_path.exists():
                             print(f"    Warning: Image file missing at {full_path}")

                # 2. Recurse into children (if any)
                if 'nodes' in node and isinstance(node['nodes'], list):
                    traverse_for_images(node['nodes'])

        traverse_for_images(structure)
        
        # Stats
        total_images = sum(len(paths) for paths in self.figure_index.values())
        print(f"  Total figures indexed: {total_images} across {len(self.figure_index)} distinct figure numbers.")
    
    def _resolve_relevant_images(self, text: str) -> List[Path]:
        """
        Scans text for references like "Figure 2" or "Fig. 2" and returns matching image paths.
        This solves the 'Disconnected Context' problem.
        """
        # Find references like "Figure 2", "Fig. 2", "Fig 2"
        # We use \b to ensure we match whole words
        # Added support for Figure (letter)(digit) format
        refs = re.findall(r'\b(?:[Ff]ig\.?|[Ff]igure)\s*([A-Za-z]?\d+)', text, re.IGNORECASE)
        
        relevant_paths = []
        seen_paths = set()
        
        for fig_num in refs:
            if fig_num in self.figure_index:
                # Retrieve the LIST of paths for this figure number
                paths = self.figure_index[fig_num]
                
                for path in paths:
                    if path not in seen_paths:
                        relevant_paths.append(path)
                        seen_paths.add(path)
        
        return relevant_paths
  
    # ========================================================================
    # Stage 1: LLM Node Identification
    # ========================================================================
    
    def _collect_all_nodes(self, structure: List[dict]) -> List[dict]:
        """Collect all semantic_group nodes with their keywords from markdown_v3 structure."""
        nodes = []
        
        def traverse(node_list: List[dict], parent_title: str = "", section_title: str = ""):
            for node in node_list:
                node_type = node.get('node_type', 'section')
                title = node.get('title', '')
                text = node.get('text', '')
                
                current_section = section_title
                if node_type == 'section' or (not section_title and title):
                    current_section = title
                
                # Collect semantic_group nodes (these contain the meaningful text chunks)
                if node_type == 'semantic_group' and text:
                    # Extract keywords from sentence children
                    keywords = []
                    for sentence_child in node.get('nodes', []):
                        if sentence_child.get('node_type') == 'sentence':
                            for kw_child in sentence_child.get('nodes', []):
                                if kw_child.get('node_type') == 'keyword':
                                    kw_title = kw_child.get('title', '')
                                    kw_metadata = kw_child.get('metadata', {})
                                    kw_context = kw_metadata.get('relevance', '') or kw_metadata.get('summary', '')
                                    keywords.append({'term': kw_title, 'context': kw_context})

                    child_sentences = []
                    if 'nodes' in node:
                        for child in node['nodes']:
                            if child.get('node_type') == 'sentence':
                                child_sentences.append(
                                    {'node_id': child.get('node_id', ''),
                                    'text': child.get('text', '')}
                                )
                    
                    nodes.append({
                        'node_id': node.get('node_id', ''),
                        'title': title,
                        'text': text,
                        'summary': node.get('summary', ''),
                        'section_title': current_section,
                        'parent_title': parent_title,
                        'keywords': keywords,
                        'metadata': node.get('metadata', {}),
                        'sentences': child_sentences,
                    })
                
                # Recurse
                if 'nodes' in node and node['nodes']:
                    traverse(node['nodes'], title, current_section)
        
        traverse(structure)
        return nodes

    async def _check_node_relevance(self, node: dict, semaphore: asyncio.Semaphore, timeout: int = 30) -> tuple[dict, bool, str]:
        """Check if a node is relevant for ionic conductivity extraction using LLM."""
        from google.genai import types
        
        # Build compact context from keywords
        keyword_list = [kw['term'] for kw in node.get('keywords', [])]
        keywords_str = ', '.join(keyword_list) if keyword_list else 'None'
        
        prompt = f"""Determine if this section from a scientific paper contains ionic conductivity measurements with numerical values.

Section Title: {node.get('title', 'Unknown')}
Section text: {node.get('text', 'No text')}

Answer these questions:
1. is_relevant: Does this section likely contain ionic conductivity measurements (numerical values in S/cm or similar units)?
2. relevance_reason: Brief explanation (1 sentence)
3. expected_data_points: How many distinct measurements might be in this section? (0 if not relevant)

Consider:
- Sections about "results", "conductivity", "impedance", "EIS" are likely relevant
- Look for keywords like: conductivity, S/cm, impedance, measurement, temperature

Respond with JSON only."""

        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level=SCIENTIFIC_NORMALIZER_THINKING_LEVEL) 
                            if self.model_text == 'gemini-3-flash-preview' else None,
            temperature=0.7 if "2.5" in self.model_text else 1.0,
            max_output_tokens=4096,
            response_mime_type="application/json",
            response_json_schema=NodeRelevanceResponse.model_json_schema()
        )

        async with semaphore:
            # Use the safe wrapper (handles retries, thread pooling, and COST TRACKING)
            response = await _safe_llm_call_async(
                self.client.models.generate_content, # The function to call
                model=self.model_text,               # Passed to func AND used by cost tracker
                contents=prompt,                     # Passed to func
                config=config,                       # Passed to func
                timeout=timeout                      # Wrapper timeout
            )
            
            # Handle Failure (Wrapper returns None on persistent failure)
            if not response or not response.text:
                return (node, True, "LLM Call Failed/Timed out - including by default (Permissive)")

            # Handle Success
            try:
                result = NodeRelevanceResponse.model_validate_json(response.text)
                return (node, result.is_relevant, result.relevance_reason)
            except Exception as e:
                # If parsing fails, be permissive so we don't lose data
                return (node, True, f"JSON Parsing Error: {e} - including by default")
    
    async def _filter_relevant_nodes(self, nodes: List[dict], batch_size: int = 7) -> List[dict]:
        """
        Filter nodes to only those likely containing ionic conductivity data.
        Heuristic: if a node has an image or references a figure, AUTO-INCLUDE it.
        otherwise ask the LLM.
        
        """
        print(f"\n[Stage 1] Filtering {len(nodes)} nodes for relevance...")
        
        semaphore = asyncio.Semaphore(batch_size)
        # tasks = [self._check_node_relevance(node, semaphore) for node in nodes]
        tasks = []
        relevant_nodes = []

        # Regex for finding figure references
        fig_ref_regex = re.compile(r'\b(Figure|Fig|Fig.\s?\d+|fig.\s?\d+)\b', re.IGNORECASE)
        # img_embed_pattern = re.compile(r'!\[.*?\]\(.*?\)')
        img_embed_pattern = re.compile(r'__IMG_[a-zA-Z0-9]+__')
        
        for node in nodes:
            text = node.get('text', '')
            has_embedded_image = bool(img_embed_pattern.search(text))
            has_figure_reference = bool(fig_ref_regex.search(text))

            if has_figure_reference: # this is still important even though we separately handle images
                reason="Auto-included: References a figure"
                node['is_relevant'] = True
                node['relevance_reason'] = reason
                relevant_nodes.append(node)

            if has_embedded_image:
                # reason = "Auto-included: Contains figure"
                # print(f"    Auto-included: {node.get('title', 'Unknown')}")

                # node['is_relevant'] = True
                # node['relevance_reason'] = reason
                # relevant_nodes.append(node)
                tasks.append(self._check_node_relevance(node, semaphore))
                
            else:
                tasks.append(self._check_node_relevance(node, semaphore))   

        
        completed = 0
        
        if tasks:
            print(f"  → Checking {len(tasks)} text-only nodes with LLM...")
            for coro in asyncio.as_completed(tasks):
                node, is_relevant, reason = await coro
                completed += 1
                
                status = "✓" if is_relevant else "○"
                title = node.get('title', 'Unknown')[:40]
                print(f"  [{completed}/{len(nodes)}] {status} {title}...")
                
                if is_relevant:
                    relevant_nodes.append(node)
        
        print(f"  → {len(relevant_nodes)} relevant nodes identified")
        return relevant_nodes

    # ========================================================================
    # Stage 2a: Full Extraction with Provenance (Text Only)
    # ========================================================================
    
    def _build_extraction_prompt(self, node: dict) -> str:
        """Build extraction prompt with full context."""
        keywords_str = ', '.join([kw['term'] for kw in node.get('keywords', [])]) or 'None'
        
        # 1. FORMAT THE TEXT WITH IDs
        # If we have sentence breakdown, use it. Otherwise fall back to raw text.
        formatted_text = ""
        if node.get('sentences'):
            for sent in node['sentences']:
                # We use an explicit tag format that is easy for the LLM to reference
                formatted_text += f"[Sentence ID: {sent['node_id']}] {sent['text']}\n\n"
        else:
            # Fallback for nodes without children
            formatted_text = node.get('text', '')


        return f"""Extract ALL ionic conductivity measurements from this text.

Section: {node.get('section_title', 'Unknown')}
Title: {node.get('title', 'Unknown')}
Keywords: {keywords_str}

Text:
{formatted_text}

INSTRUCTIONS:
0. If specific numerical values are not present, DO NOT return a JSON object for that material.
1. Analyze the provided text text for ionic conductivity data.
2. If an image is provided along with this text, analyze it as well. 
   - If the image is a data plot (e.g., Arrhenius plot), extract the specific conductivity values from the data points in the plot.
   - If the image is a table, extract the values from the table rows.

For EACH ionic conductivity measurement, extract:
1. reason: A brief explanation of how and why this data point was extracted.
2. confidence: "high" (primary data), "medium" (clearly stated cited data), "low" (ambiguous or inferred)
3. data_source: "primary" (this paper), "internal-citation" (from reference figure/table/section of this paper), "external-citation" (from another paper), "inferred"
4. source_sentence_id: The ID of the sentence containing this measurement. Always in the format of 0000 (e.g., "0001", "0042", etc..)
5. material_class: Ceramic, Polymer, Composite, or Other
6. electrolyte_name: full_name, acronym, proportion
7. ionic_conductivity_S_per_cm: The NUMERIC value including units. 
   - BAD: "highest conductivity", "not specified", "see Fig 4"
   - GOOD: "2.4 x 10^-3 S/cm", "0.7 mS/cm"
   - If the text says "see Figure 4", you MUST estimate the value from the attached Figure 4. Do not return placeholders.
8. measurement_temperature: Temperature (e.g., "25°C", "RT")
9. material_description: Any material description as much as included in the text, or "N/A (Cited Work)"
10. processing_method: Synthesis details and method as much as included in the text, or "N/A (Cited Work)"

CRITICAL - Material Naming Guidelines (electrolyte_name field):
- ALWAYS extract the chemical formula if visible in text, figures, or tables
  ✓ Good: "Li6PS5I", "Li7La3Zr2O12", "Li1.3Al0.3Ti1.7(PO4)3"
  ✗ Avoid: "argyrodite", "garnet", "NASICON"
- If a formula appears in a figure caption, table header, or nearby text, use it
- Prefer specific formulas over generic class names:
  ✓ Prefer: "Li6PS5I" over "lithium argyrodite"
  ✓ Prefer: "Li7La3Zr2O12" over "LLZO garnet"
- For doped/substituted materials, include base formula in full_name and doping in proportion:
  ✓ full_name: "Li6PS5-xSexBr", proportion: "x=0.5"
  ✓ full_name: "Li7La3Zr2O12", proportion: "Al-doped"
- Only use generic names ("solid electrolyte", "sample") if NO formula or systematic name exists
- Check figure captions and table headers for formulas even if not in the sentence text

IMPORTANT:
- Extract EVERY measurement, even from cited references
- If same material has multiple temperatures, create separate entries
- Be precise with values - preserve scientific notation

Respond with JSON only."""

    async def _extract_from_node(self, node: dict, semaphore: asyncio.Semaphore, timeout: int = 60) -> tuple[str, List[dict]]:
        """Extract ionic conductivity data from a single node."""
        from google.genai import types
        
        prompt = self._build_extraction_prompt(node)
        node_title = node.get('title', 'Unknown')[:50]
        text_content = node.get('text', '')


        # 1. Resolve images (handle multiple figures + disconnected context)
        all_image_paths = self._resolve_relevant_images(text_content)

        # Record that we are about to process these images
        # Filter: only keep images that are not already processed
        image_paths = []
        for path in all_image_paths:
            with self._image_lock:
                if path not in self.processed_images:
                    image_paths.append(path)
                    self.processed_images.add(path) # Mark as seen immediately
                else:
                    print(f"        (Skipping) Image {path.name} already processed.")

        # 2. Build content payload
        prompt_text = self._build_extraction_prompt(node)
        
        raw_parts = [types.Part(text=prompt_text)]
        

        # Add images using Gemini 3.0 spec (media_resolution)
        for img_path in image_paths:
            try:
                # Read image bytes
                image_bytes = img_path.read_bytes()
                image_b64 = base64.b64encode(image_bytes).decode('utf-8')

                mime_type = "image/png" if img_path.suffix.lower() == '.png' else "image/jpeg"

                # Add Part with High Resolution setting
                # Construct raw dict matching the API spec
                image_part = types.Part(
                    inline_data=types.Blob(
                        mime_type=mime_type,
                        data=image_b64
                    ),
                    # Ensure your self.client was initialized with api_version='v1alpha'
                    media_resolution={"level": "media_resolution_high"}
                )

                raw_parts.append(image_part)

                print(f"        (Multimodal) Attached {img_path.name} to {node_title}")
            except Exception as e:
                print(f"    Error loading image {img_path}: {e}")
        
        # 3. Choose model & config
        # if we have images, use Gemini 3 Flash, if text only, use lighter model
        if image_paths:
            active_model = self.model_vision
            temperature = 1.0 # as recommended by Gemini docs
        else:
            active_model = self.model_text
            temperature = 0.0    
        
        async with semaphore:
            try:
                loop = asyncio.get_event_loop()
                response = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: self.client.models.generate_content(
                            model=active_model,
                            contents=[
                                types.Content(
                                    parts = raw_parts
                                )
                            ],
                            config=types.GenerateContentConfig(
                                temperature=temperature,
                                thinking_config=types.ThinkingConfig(thinking_level=SCIENTIFIC_NORMALIZER_THINKING_LEVEL) if active_model == 'gemini-3-flash-preview' else None,
                                max_output_tokens=8192,
                                response_mime_type="application/json",
                                response_json_schema=MaterialExtractionResponse.model_json_schema()
                            )
                        )
                    ),
                    timeout=timeout
                )
                print('*****************\n')
                if active_model == self.model_vision:
                    print('>>> Vision needed for ', node_title)
                    print(response.usage_metadata.prompt_tokens_details)
                result = MaterialExtractionResponse.model_validate_json(response.text)
                print(">>> Extract from node: ", node_title, "\n", result)
                print('----------------\n')
                # Add source node info
                materials = []
                for mat in result.materials:
                    mat_dict = mat.model_dump()
                    mat_dict['source_node'] = {
                        'node_id': node.get('node_id', ''),
                        'title': node.get('title', ''),
                        'section': node.get('section_title', '')
                    }
                    materials.append(mat_dict)
                
                return (node_title, materials)
                
            except asyncio.TimeoutError:
                print(f"    Timeout extracting from {node_title}")
                return (node_title, [])
            except Exception as e:
                print(f"    Error extracting from {node_title}: {e}")
                return (node_title, [])
    
    async def _extract_from_nodes(self, nodes: List[dict], batch_size: int = 7) -> List[dict]:
        """Extract from all relevant nodes."""
        print(f"\n[Stage 2] Extracting from {len(nodes)} relevant nodes...")
        
        semaphore = asyncio.Semaphore(batch_size)
        tasks = [self._extract_from_node(node, semaphore) for node in nodes]
        
        all_materials = []
        completed = 0
        
        for coro in asyncio.as_completed(tasks):
            node_title, materials = await coro
            completed += 1
            all_materials.extend(materials)
            
            status = "✓" if materials else "○"
            print(f"  [{completed}/{len(nodes)}] {status} {node_title}... ({len(materials)} data points)")
        
        print(f"  → {len(all_materials)} total data points extracted")
        return all_materials

    # ========================================================================
    # Stage 2b: Process images
    # ========================================================================
    def _collect_image_nodes(self, structure: List[dict]) -> List[dict]:
        """Collect all 'image' nodes from the structure."""
        image_nodes = []
        
        def traverse(nodes):
            for node in nodes:
                if node.get('node_type') == 'image':
                    image_nodes.append(node)
                
                if 'nodes' in node:
                    traverse(node['nodes'])
        
        traverse(structure)
        return image_nodes
    
    async def _extract_from_image_only(self, node: dict, base_path: Path, semaphore: asyncio.Semaphore) -> tuple[str, List[dict]]:
        """
        Directly extract data from an image node using the Vision model.
        """
        from google.genai import types
        
        src = node.get('src', '')
        # Fallback to title if src is missing/empty, assuming title contains the filename
        if not src: 
             src = node.get('title', '')
             
        full_path = base_path / src

        # Filter: Skip if this image was already attached to a text node in Stage 2a
        # if full_path in self.processed_images:
        #     print(f"    (Skipping) Image-only pipeline: {src} already covered in text analysis.")
        #     return (src, [])
        self.processed_images.add(full_path)
        
        node_id = node.get('node_id', 'img')

        if not full_path.exists():
            print(f"    Warning: Skipping missing image {src}")
            return (src, [])

        # Prompt specifically designed for standalone images
        prompt_text = f"""Analyze this scientific image (Figure/Table) specifically for Ionic Conductivity Data.

Image Filename: {src}
Node ID: {node_id}

INSTRUCTIONS:
1. If this is a Data Plot (e.g., Arrhenius plot, Conductivity vs Temperature):
   - Extract data points carefully.
   - If multiple lines exist, identify the material for each line.
   - Estimate values as precisely as possible.

2. If this is a Table:
   - Extract rows containing ionic conductivity.

3. If this is NOT related to ionic conductivity (e.g., SEM micrograph, XRD pattern, photo of a battery):
   - Return an empty list.

For EACH measurement found:
- source_sentence_id: "Derived from Plot {src}" or content of table cell.
- material_class: Ceramic, Polymer, Composite, or Other
- electrolyte_name: Name of the material (look at legends, labels)
- ionic_conductivity_S_per_cm: Numeric value with units (e.g. "1.2e-4 S cm^-1")
- measurement_temperature: Temperature (Look for x-axis labels like 1000/T or °C)
- confidence: "high" (clear text/table), "medium" (plot estimation)
- data_source: "primary"
- material_description: Any details in legends/labels
- processing_method: "N/A" or any relevant details visibly mentioned

Respond with JSON only."""

        try:
            image_bytes = full_path.read_bytes()
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Determine mime type
            mime = "image/png" if full_path.suffix.lower() == '.png' else "image/jpeg"

            async with semaphore:
                # print(f"    Processing Image: {src}")
                response = await _safe_llm_call_async(
                    self.client.models.generate_content,
                    model=self.model_vision, # MUST use vision model
                    contents=[
                        types.Content(
                            parts=[
                                types.Part(text=prompt_text),
                                types.Part(
                                    inline_data=types.Blob(mime_type=mime, data=image_b64),
                                    media_resolution={"level": "media_resolution_high"}
                                )
                            ]
                        )
                    ],
                    config=types.GenerateContentConfig(
                        temperature=0.2 if "2.5" in self.model_vision else 1.0, # Lower temperature for precise reading
                        thinking_config=types.ThinkingConfig(thinking_level=SCIENTIFIC_NORMALIZER_THINKING_LEVEL) if self.model_vision == 'gemini-3-flash-preview' else None,
                        max_output_tokens=4096,
                        response_mime_type="application/json",
                        response_json_schema=MaterialExtractionResponse.model_json_schema()
                    )
                )

                if not response or not response.text:
                    return (src, [])

                result = MaterialExtractionResponse.model_validate_json(response.text)

                print('88888888888888888888')
                print(">>> Extract from image: ", src, "\n", result)
                print('88888888888888888888')
                
                # Tag results
                materials = []
                for mat in result.materials:
                    mat_dict = mat.model_dump()
                    mat_dict['source_node'] = {
                        'node_id': node_id,
                        'title': src,
                        'section': 'Image-Only Extraction'
                    }
                    materials.append(mat_dict)
                
                return (src, materials)

        except Exception as e:
            print(f"    Error processing image {src}: {e}")
            return (src, [])
    # ========================================================================
    # Stage 3: Smart Deduplication
    # ========================================================================
    
    # def _deduplicate(self, materials: List[dict]) -> List[dict]:
    #     """Remove duplicate data points using normalized values and cross-references."""
    #     print(f"\n[Stage 3] Deduplicating {len(materials)} data points...")
        
    #     # Priority sorting: high confidence and primary data first
    #     materials.sort(key=lambda x: (
    #         0 if x.get('confidence') == 'high' else 1,
    #         0 if x.get('data_type') == 'primary' else 1
    #     ))

    #     # Remove duplicates
    #     unique = []
    #     duplicates_removed = 0

    #     for candidate in materials:
    #         is_dup = False
    #         c_cond = candidate.get('_norm_cond')
    #         c_temp = candidate.get('_norm_temp')
    #         # Normalize name for comparison (strip spaces/case)
    #         c_name = (candidate.get('electrolyte_name', {}).get('acronym') or 
    #                 candidate.get('electrolyte_name', {}).get('full_name') or "").lower().strip()

    #         for existing in unique:
    #             e_cond = existing.get('_norm_cond')
    #             e_temp = existing.get('_norm_temp')
    #             e_name = (existing.get('electrolyte_name', {}).get('acronym') or 
    #                     existing.get('electrolyte_name', {}).get('full_name') or "").lower().strip()

    #             # Logic: If names match and we have valid floats for both
    #             if c_name == e_name and c_cond is not None and e_cond is not None:
    #                 # 1. Conductivity: Use 5% relative tolerance
    #                 # 2. Temperature: Use 2°C absolute tolerance
    #                 cond_match = math.isclose(c_cond, e_cond, rel_tol=0.05)
                    
    #                 # Handle cases where temp might be None
    #                 temp_match = True
    #                 if c_temp is not None and e_temp is not None:
    #                     temp_match = abs(c_temp - e_temp) <= 2.0
                    
    #                 if cond_match and temp_match:
    #                     is_dup = True
    #                     break
            
    #         if is_dup:
    #             duplicates_removed += 1
    #         else:
    #             unique.append(candidate)
                
    #     print(f"  → Removed {duplicates_removed} duplicates, {len(unique)} unique points remain.")
    #     return unique

    def _deduplicate(self, materials: List[dict]) -> List[dict]:
        """Remove duplicate data points using normalized values and cross-references."""
        print(f"\n[Stage 3] Deduplicating {len(materials)} data points...")
        
        # --- FIX 1: Sort by Completeness ---
        # We want to keep the record that has the most data.
        # Priority: High Confidence > Primary Source > Has Valid Temperature > Has Valid Conductivity
        materials.sort(key=lambda x: (
            0 if x.get('confidence') == 'high' else 1,
            0 if x.get('data_type') == 'primary' else 1,
            0 if x.get('_norm_temp') is not None else 1,
            0 if x.get('_norm_cond') is not None else 1
        ))

        unique = []
        merged_count = 0

        for candidate in materials:
            is_dup = False
            
            # Get Candidate Key
            c_cond = candidate.get('_norm_cond')
            c_temp = candidate.get('_norm_temp')
            c_canon = candidate.get('canonical_formula')
            
            if c_canon:
                c_key = c_canon.strip().lower()
            else:
                c_name = candidate.get('electrolyte_name', {}).get('full_name', "")
                c_prop = candidate.get('electrolyte_name', {}).get('proportion', "")
                c_key = f"{c_name} {c_prop}".lower().strip().replace(" ", "")

            for existing in unique:
                # Get Existing Key
                e_cond = existing.get('_norm_cond')
                e_temp = existing.get('_norm_temp')
                e_canon = existing.get('canonical_formula')
                
                if e_canon:
                    e_key = e_canon.strip().lower()
                else:
                    e_name = existing.get('electrolyte_name', {}).get('full_name', "")
                    e_prop = existing.get('electrolyte_name', {}).get('proportion', "")
                    e_key = f"{e_name} {e_prop}".lower().strip().replace(" ", "")

                # MATCHING LOGIC
                if c_key == e_key and c_cond is not None and e_cond is not None:
                    # 1. Check Conductivity (5% tolerance)
                    if math.isclose(c_cond, e_cond, rel_tol=0.05):
                        # 2. Check Temperature (2.0 deg tolerance)
                        temp_match = True
                        if c_temp is not None and e_temp is not None:
                            temp_match = abs(c_temp - e_temp) <= 2.0
                        
                        if temp_match:
                            # IT IS A DUPLICATE!
                            # MERGE candidate info INTO existing record
                            self._merge_records(target=existing, source=candidate)
                            is_dup = True
                            merged_count += 1
                            break
            
            if not is_dup:
                unique.append(candidate)
                
        print(f"  → Merged {merged_count} duplicates, {len(unique)} unique points remain.")
        return unique

    def _merge_records(self, target: dict, source: dict):
        """
        Intelligently merges 'source' data into 'target' data.
        """
        # 1. Merge Processing Method (Text often has this, Images often miss it)
        t_proc = target.get('processing_method')
        s_proc = source.get('processing_method')
        
        if not t_proc or t_proc.lower() in ["n/a", "not specified", "unknown"]:
            if s_proc and s_proc.lower() not in ["n/a", "not specified"]:
                target['processing_method'] = s_proc
        elif s_proc and s_proc.lower() not in ["n/a", "not specified"]:
            if s_proc.lower() not in t_proc.lower():
                # Append if different and valid
                target['processing_method'] = f"{t_proc} | {s_proc}"

        # 2. Merge Description
        t_desc = target.get('material_description')
        s_desc = source.get('material_description')
        
        if not t_desc or t_desc.lower() in ["n/a", "not specified"]:
            if s_desc: target['material_description'] = s_desc
        elif s_desc and len(s_desc) > 10: # Only merge substantial descriptions
            if s_desc.lower() not in t_desc.lower():
                target['material_description'] = f"{t_desc} ; {s_desc}"

        # 3. Merge Source Sentence IDs (Traceability)
        # This lets you know this data point came from "Text AND Figure 5"
        t_src = str(target.get('source_sentence_id', ''))
        s_src = str(source.get('source_sentence_id', ''))
        if s_src and s_src not in t_src:
            target['source_sentence_id'] = f"{t_src}, {s_src}"
    # ========================================================================
    # Stage 4: Validation -- see DataValidator class for details
    # ========================================================================    
    def _final_sanity_check(self, materials):
        clean_materials = []
        for mat in materials:
            # 1. Check if "not specified" or empty
            val = mat.get('ionic_conductivity_S_per_cm', '').lower()
            if 'not specified' in val or 'unknown' in val:
                continue
                
            # 2. Check if it contains at least one digit
            if not any(char.isdigit() for char in val):
                # e.g. "highest conductivity"
                continue
                
            clean_materials.append(mat)
        return clean_materials
    
    # ========================================================================
    # Main Pipeline
    # ========================================================================
    async def _run_image_pipeline(self, image_nodes: List[dict], base_path: Path, batch_size: int = 5) -> List[dict]:
        """Orchestrator for the image-only extraction stage."""
        print(f"\n[Stage 2b] Processing {len(image_nodes)} images directly...")
        
        semaphore = asyncio.Semaphore(batch_size)
        tasks = [self._extract_from_image_only(node, base_path, semaphore) for node in image_nodes]
        
        all_materials = []
        completed = 0
        
        for coro in asyncio.as_completed(tasks):
            src, materials = await coro
            completed += 1
            if materials:
                print(f"  [{completed}/{len(image_nodes)}] ✓ {src} ({len(materials)} points)")
                all_materials.extend(materials)
            else:
                # Optional: verbose logging for empty images
                # print(f"  [{completed}/{len(image_nodes)}] ○ {src}")
                pass
                
        return all_materials

    def extract(self, structure: List[dict], base_path: Path, batch_size: int = 7) -> dict:
        """
        Run the full four-stage extraction pipeline.
        
        Returns:
            dict with 'materials' list and 'stats' summary
        """
        # track time
        start_time = time.time()

        # 0. Index Figures First
        self._index_figures(base_path, structure)
        self.processed_images = set()
        self._image_lock = threading.Lock()

        # Collect all nodes
        text_nodes = self._collect_all_nodes(structure)
        image_nodes = self._collect_image_nodes(structure)
        print(f"Found {len(text_nodes)} text nodes and {len(image_nodes)} image nodes.")
        
        # if not all_nodes:
        #     return {'materials': [], 'stats': {'total_nodes': 0}}
        
        # Stage 1: Filter relevant nodes
        # relevant_text_nodes = asyncio.run(self._filter_relevant_nodes(text_nodes, batch_size)) # skipping, this is redundant
        relevant_text_nodes = text_nodes

        # 3. Stage 2a: Extract from Text Nodes
        # (This uses your existing text extraction logic)
        text_materials = []
        if relevant_text_nodes:
            text_materials = asyncio.run(self._extract_from_nodes(relevant_text_nodes, batch_size))

        # 4. Stage 2b: Extract from Image Nodes (NEW)
        image_materials = []
        if image_nodes:
            image_materials = asyncio.run(self._run_image_pipeline(image_nodes, base_path, batch_size))

        # Merge Results
        print(f"\nMerging: {len(text_materials)} text-based + {len(image_materials)} image-based points.")
        combined_materials = text_materials + image_materials
        
        if not combined_materials:
            print("No relevant nodes found!")
            return {'materials': [], 'stats': {'total_nodes': len(combined_materials), 'relevant_nodes': 0}}
        
        

        # Stage 4: Validate
        nodes_map = {n['node_id']: n for n in relevant_text_nodes + image_nodes}
        validator = DataValidator(self.client, self.figure_index, model_name=self.model_text)
        validated_materials = validator.validate_all(combined_materials, nodes_map)
        
        # Stage 2.5: Document-Level Name Normalization
        doc_name = structure[0].get('title', 'Unknown') if structure else 'Unknown'
        name_normalizer = MaterialNameNormalizer(self.client, model_name=self.model_text)
        normalized_materials = asyncio.run(
            name_normalizer.normalize_document_names(validated_materials, doc_name)
        )

        # Stage 3: Normalize & Deduplicate
        normalizer = ScientificNormalizer(self.client, model_name=self.model_text)
        materials_with_floats = asyncio.run(normalizer.normalize_batch(normalized_materials))
        unique_materials = self._deduplicate(materials_with_floats)
        unique_materials = self._final_sanity_check(unique_materials) # clean non-numeric values
        # unique_materials = materials_with_floats # skip deduplication for now
        

        # Compile stats
        end_time = time.time() - start_time
        validated_materials = unique_materials # just we don't want to change all the variables below.
        stats = {
            'total_nodes': len(text_nodes) + len(image_nodes),
            # 'relevant_nodes': len(relevant_nodes),
            'raw_extracted': len(combined_materials),
            'after_dedup': len(unique_materials),
            'time_elapsed': end_time,
            'valid_count': sum(1 for m in validated_materials if m.get('_validation', {}).get('is_valid', True)),
            'invalid_count': sum(1 for m in validated_materials if not m.get('_validation', {}).get('is_valid', True)),
            'by_confidence': {
                'high': sum(1 for m in validated_materials if m.get('confidence') == 'high'),
                'medium': sum(1 for m in validated_materials if m.get('confidence') == 'medium'),
                'low': sum(1 for m in validated_materials if m.get('confidence') == 'low'),
            },
            'by_source': {
                'primary': sum(1 for m in validated_materials if m.get('data_source') == 'primary'),
                'cited': sum(1 for m in validated_materials if m.get('data_source') == 'cited'),
                'inferred': sum(1 for m in validated_materials if m.get('data_source') == 'inferred'),
            },
            'from_text_nodes': len(text_materials),
            'from_image_nodes': len(image_materials),
            'final_count': len(unique_materials)
        }
        

        return {'materials': validated_materials, 'stats': stats}

    ## Helpers

    def _load_image_content(self, text: str, base_path: Path):
        """Find markdown image tags, load the file, and return the image object."""
        
        # Regex to find standard markdown image links: ![](_path_to_image.jpg)
        # Adjust regex if your format is different (e.g. HTML img tags)
        match = re.search(r'!\[.*?\]\((.*?)\)', text)
        
        if match:
            image_rel_path = match.group(1)
            # You might need to adjust logic to resolve the absolute path
            # identifying where the images are stored relative to your script
            image_path = base_path / image_rel_path
            
            if image_path.exists():
                try:
                    return Image.open(image_path)
                except Exception as e:
                    print(f"Warning: Could not load image {image_path}: {e}")
                    return None
        return None

# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Extract ionic conductivity data from PageIndex JSON structure',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Stages:
  1. LLM Node Identification - Filter nodes likely containing conductivity data
  2. Full Extraction - Extract data with provenance tracking
  3. Smart Deduplication - Remove duplicates using normalized values
  4. Validation - Check data consistency and flag issues

Examples:
  python run_extraction.py results/paper_keywords_structure.json --asset_dir tests/markdowns/paper_md
  python run_extraction.py results/paper_keywords_structure.json --batch-size 10
        """
    )
    
    parser.add_argument('input_file', help='Path to PageIndex JSON file')
    parser.add_argument('--asset_dir', help='Path to original MD folder where assets like images are')
    parser.add_argument('--output', '-o', help='Output JSON file path')
    parser.add_argument('--batch-size', '-b', type=int, default=7, 
                        help='Max concurrent API calls (default: 7)')
    
    args = parser.parse_args()
    
    # Load input
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        return 1
    
    print(f"Loading structure from: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    structure = data.get('structure', [])
    doc_name = data.get('doc_name', input_path.stem)
    
    print(f"Document: {doc_name}")
    print("=" * 70)

    print("MODELS USED")
    print(f"SCIENTIFIC_NORMALIZER_MODEL: {SCIENTIFIC_NORMALIZER_MODEL}")
    print(f"DATA_VALIDATOR_MODEL: {DATA_VALIDATOR_MODEL}")
    print(f"MATERIAL_NAME_NORMALIZER_MODEL: {MATERIAL_NAME_NORMALIZER_MODEL}")
    print(f"EXTRACTOR_TEXT_MODEL: {EXTRACTOR_TEXT_MODEL}")
    print(f"EXTRACTOR_VISION_MODEL: {EXTRACTOR_VISION_MODEL}")    
    # Run extraction pipeline
    base_path = Path(args.asset_dir)

    extractor = MaterialExtractor(model_text=EXTRACTOR_TEXT_MODEL, model_vision=EXTRACTOR_VISION_MODEL)
    result = extractor.extract(structure, base_path=base_path, batch_size=args.batch_size)
    
    materials = result['materials']
    stats = result['stats']
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_materials.json"
    

    # Print cost summary
    tracker.print_summary()

    # Save results
    output_data = {
        'doc_name': doc_name,
        'models_used': {
            'SCIENTIFIC_NORMALIZER_MODEL': SCIENTIFIC_NORMALIZER_MODEL,
            'DATA_VALIDATOR_MODEL': DATA_VALIDATOR_MODEL,
            'MATERIAL_NAME_NORMALIZER_MODEL': MATERIAL_NAME_NORMALIZER_MODEL,
            'EXTRACTOR_TEXT_MODEL': EXTRACTOR_TEXT_MODEL,
            'EXTRACTOR_VISION_MODEL': EXTRACTOR_VISION_MODEL
        },
        'cost_summary': {
            'total_input_tokens': tracker.total_input_tokens,
            'total_output_tokens': tracker.total_output_tokens,
            'total_cost_usd': tracker.total_cost_usd,
            'call_counts': tracker.call_counts
        },
        'source_file': str(input_path),
        'extraction_stats': stats,
        'material_count': len(materials),
        'materials': materials
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"✓ Saved to: {output_path}")
    print(f"Time elapsed (s) ", stats.get('time_elapsed', 0))
    print(f"\nStatistics:")
    print(f"  Nodes analyzed: {stats.get('total_nodes', -1)}")
    # print(f"  Relevant nodes: {stats.get('relevant_nodes', -1)}")
    print(f"  Raw data points: {stats.get('raw_extracted', -1)}")
    print(f"  After dedup: {stats.get('after_dedup', -1)}")
    print(f"  Valid: {stats.get('valid_count', -1)}, Invalid: {stats.get('invalid_count', -1)}")
    print(f"\nBy Confidence:")
    for level, count in stats.get('by_confidence', {}).items():
        print(f"  {level}: {count}")
    print(f"\nBy Data Source:")
    for source, count in stats.get('by_source', {}).items():
        print(f"  {source}: {count}")
    
    # Show sample materials
    print(f"\nSample data points:")
    for mat in materials[:5]:
        name = mat.get('electrolyte_name', {}).get('acronym') or mat.get('electrolyte_name', {}).get('full_name', 'Unknown')
        cond = mat.get('ionic_conductivity_S_per_cm', 'N/A')
        temp = mat.get('measurement_temperature', 'N/A')
        conf = mat.get('confidence', 'N/A')
        valid = "✓" if mat.get('_validation', {}).get('is_valid', True) else "✗"
        print(f"  {valid} {name}: {cond} @ {temp} [{conf}]")
    
    if len(materials) > 5:
        print(f"  ... and {len(materials) - 5} more")
    
    # Force exit to prevent hanging on thread cleanup
    import os
    os._exit(0)


if __name__ == '__main__':
    exit(main())
