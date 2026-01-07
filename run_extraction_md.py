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

import argparse
import asyncio
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

load_dotenv()


# ============================================================================
# Pydantic Models for Structured Output
# ============================================================================

class ElectrolyteName(BaseModel):
    """A structured representation of the electrolyte's name."""
    full_name: str = Field(
        ...,
        description="The complete, formal name of the electrolyte (e.g., Lithium Lanthanum Zirconate Oxide)."
    )
    acronym: Optional[str] = Field(
        None,
        description="The abbreviation, chemical formula, or common name (e.g., LLZO, PEO)."
    )
    proportion: Optional[str] = Field(
        None,
        description="Stoichiometric ratios, molar ratios, or concentrations (e.g., 90:10 wt%)."
    )


class IonicConductivityDataPoint(BaseModel):
    """Represents a single extracted data point for ionic conductivity."""
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
        description="Ionic conductivity value in S cm⁻¹ (e.g., '1.2 × 10⁻⁴', '~10⁻⁵')."
    )
    measurement_temperature: str = Field(
        ...,
        description="Temperature of measurement (e.g., '25°C', 'RT', 'room temperature')."
    )
    # Provenance tracking
    confidence: str = Field(
        ...,
        description="Confidence level: 'high' (primary data from this study), 'medium' (clearly stated cited data), 'low' (ambiguous or inferred)."
    )
    data_source: str = Field(
        ...,
        description="Data origin: 'primary' (this paper's measurement), 'cited' (from a reference), 'inferred' (calculated or estimated)."
    )
    exact_quote: str = Field(
        ...,
        description="The exact sentence or phrase containing this measurement."
    )
    specific_source_location: str = Field(
        ...,
        description="Location in document (e.g., 'Figure 3', 'Table 2', 'main text paragraph 5')."
    )
    # Cross-reference detection
    refers_to_figure: Optional[str] = Field(
        None,
        description="If this data refers to a figure (e.g., 'Figure 3', 'Fig. 2a')."
    )
    refers_to_table: Optional[str] = Field(
        None,
        description="If this data refers to a table (e.g., 'Table 2', 'Table S1')."
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


class ValidationResult(BaseModel):
    """Result of validation checks."""
    is_valid: bool = Field(..., description="Whether the data point passed validation.")
    issues: List[str] = Field(default_factory=list, description="List of validation issues found.")
    suggestions: List[str] = Field(default_factory=list, description="Suggestions for fixing issues.")


class ValidationVerdict(BaseModel):
    """The auditor's verdict on a single data point."""
    data_point_index: int = Field(..., description="The index of the data point in the provided list.")
    reason: str = Field(..., description="Explanation of why it is valid or invalid (cite the text).")
    correction: Optional[str] = Field(None, description="If invalid, provide the correct value/unit from text.")
    is_supported: bool = Field(..., description="True if the text explicitly supports this extracted value.")

class BatchValidationResponse(BaseModel):
    """Response containing verdicts for a batch of data points."""
    verdicts: List[ValidationVerdict] = Field(default_factory=list)


# ============================================================================
# UTILITIES for Robust LLM Execution
# ============================================================================
async def _safe_llm_call_async(func, *args, retries=3, timeout=60, default=None, **kwargs):
    """
    Executes a blocking LLM call (func) inside a DETACHED thread executor.
    1. Async Timeout (to kill blocking calls that hang)
    2. Retries (to handle transient failures or timeouts)
    3. Thread Isolation (Uses a fresh executor per call so hung threads don't starve the global pool)
    """
    import concurrent.futures
    loop = asyncio.get_running_loop()
    
    for attempt in range(retries):
        executor = None
        try:
            # Create a FRESH executor for this attempt only.
            # If the thread hangs, we abandon the executor (leak the thread) rather than filling a global pool.
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            
            result = await asyncio.wait_for(
                loop.run_in_executor(executor, lambda: func(*args, **kwargs)), 
                timeout=timeout
            )
            
            # Clean up the executor if successful
            executor.shutdown(wait=False)
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

def calculate_standard_units(
    cond_value: float, 
    cond_unit: str, 
    temp_value: Optional[float], 
    temp_unit: Optional[str]
) -> dict:
    """Deterministic conversion logic."""
    # Unit mapping to S/cm
    # 1 S/m = 0.01 S/cm
    # 1 mS/cm = 0.001 S/cm
    unit_multipliers = {
        "s/cm": 1.0, "s·cm⁻¹": 1.0, "scm-1": 1.0,
        "ms/cm": 1e-3, "ms·cm⁻¹": 1e-3,
        "μs/cm": 1e-6, "us/cm": 1e-6, "µs/cm": 1e-6,
        "s/m": 0.01, "s·m⁻¹": 0.01
    }
    
    # Normalize conductivity
    multiplier = unit_multipliers.get(cond_unit.lower().strip(), 1.0)
    norm_cond = cond_value * multiplier
    
    # Normalize temperature to Celsius
    norm_temp = temp_value
    if temp_unit:
        u = temp_unit.upper().strip()
        if u == "K":
            norm_temp = temp_value - 273.15
        elif u in ["RT", "ROOM", "ROOM TEMPERATURE"]:
            norm_temp = 25.0
            
    return {"norm_cond": norm_cond, "norm_temp": norm_temp}

class ScientificNormalizer:
    def __init__(self, client, model_name="gemini-2.0-flash-lite"):
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
                f"IMPORTANT: If no unit is explicitly written (e.g., just '10^-4'), "
                f"ASSUME the unit is 'S/cm' and set cond_unit='S/cm'.\n\n"
                f"for temperature, make an educated guess based on the value.\n\n"
                f"Conductivity: {cond_raw}\n"
                f"Temperature: {temp_raw}"
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
                            results = calculate_standard_units(**args)
                            
                            mat['_norm_cond'] = results['norm_cond']
                            mat['_norm_temp'] = results['norm_temp']
                except Exception as e:
                    print(f"    Warning: Could not normalize {mat.get('electrolyte_name')}: {e}")
                    mat['_norm_cond'] = None
                    mat['_norm_temp'] = None

        # Create tasks for all items
        tasks = [_norm_item(mat) for mat in materials]
        await asyncio.gather(*tasks)

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
    def __init__(self, client, figure_index: Dict[str, List[Path]], model_name: str = "gemini-2.5-flash"):
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
        if cond and (cond > 5.0 or cond < 1e-12):
            issues.append(f"Physical Improbability: Conductivity {cond:.2e} S/cm is outside typical bounds.")

        temp = mat.get('_norm_temp')
        if temp and (temp < -50 or temp > 1000):
            issues.append(f"Physical Improbability: Temperature {temp}°C is outside typical bounds.")

        # Metadata Checks
        if not mat.get('exact_quote'):
            issues.append("Metadata Error: Missing source quote.")

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
                f"Material='{item['electrolyte_name']['full_name']}', "
                f"Conductivity='{item['ionic_conductivity_S_per_cm']}', "
                f"Temp='{item['measurement_temperature']}'\n"
            )

        text_content = node.get('text', '')

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
        
        prompt_text = f"""You are a Scientific Data Auditor. Verify these extracted values against the provided text.

SOURCE TEXT:
"{text_content}"

CLAIMS TO VERIFY:
{claims_text}

INSTRUCTIONS:
1. For each claim, check if the Source Text *explicitly* supports the Conductivity and Temperature values.
2. Watch out for unit errors (e.g., text says "mS/cm" but claim is treated as "S/cm").
3. Watch out for hallucinations (claims not in text).
4. Ignore minor formatting differences.
5. Attach the images to the text."

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
                        response_mime_type="application/json",
                        response_json_schema=BatchValidationResponse.model_json_schema()
                    )
                )
                print('...>>>', text_content)
                print('>>>', response.text)
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
                        
                        if not verdict.is_supported:
                            mat['_validation']['is_valid'] = False
                            mat['_validation']['issues'].append(f"LLM Audit Failed: {verdict.reason}")
                            if verdict.correction:
                                mat['_validation']['suggested_correction'] = verdict.correction
                            
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
    
    def __init__(self, model_text: str = "gemini-2.5-flash-lite", model_vision: str = 'gemini-3-flash-preview'):
        self.model_text = model_text
        self.model_vision = model_vision
        
        # Initialize Gemini client
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        self.client = client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})

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
        
        # def find_images(nodes):
        #     for node in nodes:
        #         text = node.get('text', '')
        #         matches = re.findall(r'!\[.*?\]\((.*?)\)', text)
                
        #         for image_rel_path in matches:
        #             # Regex looks for "Fig" or "Figure" followed by a number
        #             fig_match = re.search(r'(?:fig|figure)[-_]?(\d+)', image_rel_path, re.IGNORECASE)
                    
        #             if fig_match:
        #                 fig_num = fig_match.group(1) # e.g., "2"
        #                 full_path = base_path / image_rel_path
                        
        #                 if full_path.exists():
        #                     if fig_num not in self.figure_index:
        #                         self.figure_index[fig_num] = []
                            
        #                     # Avoid adding the exact same file path twice
        #                     if full_path not in self.figure_index[fig_num]:
        #                         self.figure_index[fig_num].append(full_path)
        #                         print(f"    Indexed Figure {fig_num} -> {image_rel_path}")
                
        #         if 'nodes' in node:
        #             find_images(node['nodes'])

        # find_images(structure)
        
        # # Stats
        # total_images = sum(len(paths) for paths in self.figure_index.values())
        # print(f"  Total figures indexed: {total_images} across {len(self.figure_index)} distinct figure numbers.")
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
                    
                    nodes.append({
                        'node_id': node.get('node_id', ''),
                        'title': title,
                        'text': text,
                        'summary': node.get('summary', ''),
                        'section_title': current_section,
                        'parent_title': parent_title,
                        'keywords': keywords,
                        'metadata': node.get('metadata', {})
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



        async with semaphore:
            try:
                # print(prompt)
                # raise BaseException()
                loop = asyncio.get_event_loop()
                response = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: self.client.models.generate_content(
                            model=self.model_text,
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                temperature=0.7,
                                max_output_tokens=4096,
                                response_mime_type="application/json",
                                response_json_schema=NodeRelevanceResponse.model_json_schema()
                            )
                        )
                    ),
                    timeout=timeout
                )
                
                # print(">>>", prompt)
                # print(">>>", response.text)

                result = NodeRelevanceResponse.model_validate_json(response.text)
                return (node, result.is_relevant, result.relevance_reason)
            
            except asyncio.TimeoutError:
                # On timeout, include the node (be permissive)
                return (node, True, "Timeout - including by default")
            except Exception as e:
                # On error, include the node (be permissive)
                return (node, True, f"Error checking relevance: {e}")
    
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
    # Stage 2: Full Extraction with Provenance
    # ========================================================================
    
    def _build_extraction_prompt(self, node: dict) -> str:
        """Build extraction prompt with full context."""
        keywords_str = ', '.join([kw['term'] for kw in node.get('keywords', [])]) or 'None'
        
        return f"""Extract ALL ionic conductivity measurements from this text.

Section: {node.get('section_title', 'Unknown')}
Title: {node.get('title', 'Unknown')}
Keywords: {keywords_str}

Text:
{node.get('text', '')}

INSTRUCTIONS:
1. Analyze the provided text text for ionic conductivity data.
2. If an image is provided along with this text, analyze it as well. 
   - If the image is a data plot (e.g., Arrhenius plot), extract the specific conductivity values from the data points in the plot.
   - If the image is a table, extract the values from the table rows.

For EACH ionic conductivity measurement, extract:
1. material_class: Ceramic, Polymer, Composite, or Other
2. electrolyte_name: full_name, acronym, proportion
3. ionic_conductivity_S_per_cm: The value (e.g., "1.2 × 10⁻⁴")
4. measurement_temperature: Temperature (e.g., "25°C", "RT")
5. confidence: "high" (primary data), "medium" (cited clearly), "low" (ambiguous)
6. data_source: "primary" (this paper), "cited" (from reference), "inferred"
7. exact_quote: The EXACT sentence containing this measurement
8. specific_source_location: Where in document (e.g., "Figure 3", "Table 2", "main text")
9. refers_to_figure: If references a figure (e.g., "Figure 3")
10. refers_to_table: If references a table (e.g., "Table 2")
11. material_description: Properties, or "N/A (Cited Work)"
12. processing_method: Synthesis, or "N/A (Cited Work)"

IMPORTANT:
- Extract EVERY measurement, even from cited references
- Include the EXACT quote for each measurement
- Note cross-references to figures/tables
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
        image_paths = self._resolve_relevant_images(text_content)

        # Record that we are about to process these images
        for path in image_paths:
            self.processed_images.add(path)

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
            print('>>> Vision needed for ', node_title)
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
                                max_output_tokens=8192,
                                response_mime_type="application/json",
                                response_json_schema=MaterialExtractionResponse.model_json_schema()
                            )
                        )
                    ),
                    timeout=timeout
                )
                
                result = MaterialExtractionResponse.model_validate_json(response.text)
                
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
    # Stage 2.5: Process leftover images
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
        node_id = node.get('node_id', 'img')

        if not full_path.exists():
            print(f"    Warning: Skipping missing image {src}")
            return (src, [])

        # Prompt specifically designed for standalone images
        prompt_text = f"""Analyze this scientific image (Figure/Table) specifically for Ionic Conductivity Data.

Image Filename: {src}

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
- material_class: Ceramic, Polymer, Composite, or Other
- electrolyte_name: Name of the material (look at legends, labels)
- ionic_conductivity_S_per_cm: Numeric value (e.g. "1.2e-4")
- measurement_temperature: Temperature (Look for x-axis labels like 1000/T or °C)
- confidence: "high" (clear text/table), "medium" (plot estimation)
- data_source: "primary"
- exact_quote: "Derived from Plot {src}" or content of table cell.
- specific_source_location: "{src}"
- refers_to_figure: "{src}"
- material_description: Any details in legends/labels
- processing_method: "N/A"

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
                        temperature=0.2, # Lower temperature for precise reading
                        max_output_tokens=4096,
                        response_mime_type="application/json",
                        response_json_schema=MaterialExtractionResponse.model_json_schema()
                    )
                )

                if not response or not response.text:
                    return (src, [])

                result = MaterialExtractionResponse.model_validate_json(response.text)
                
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
    
    def _unicode_superscript_to_int(self, superscript: str) -> int:
        """Convert Unicode superscript numbers to regular integers."""
        superscript_map = {
            '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
            '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9'
        }
        regular = ''.join(superscript_map.get(c, c) for c in superscript)
        return int(regular)

    def _normalize_conductivity(self, value: str) -> Optional[float]:
        """Normalize conductivity string to float in S/cm."""
        import re
        if not value:
            return None
        
        value = value.strip().lower()
        
        # Handle scientific notation variants including LaTeX and Unicode
        # "1.2 × 10⁻⁴", "1.2e-4", "1.2 x 10^-4", "10⁻⁴", "$10^{-6}$", "10^{-7}", "10⁻⁶"
        patterns = [
            r'([\d.]+)\s*[×x]\s*10[⁻\-^](\d+)',           # 1.2 × 10⁻⁴
            r'([\d.]+)e[⁻\-]?(\d+)',                       # 1.2e-4
            r'[\$]?10[\^]?[\{]?[⁻\-](\d+)[\}]?[\$]?',     # $10^{-6}$, 10^{-7}, $10^-6$
            r'10[⁻\-^](\d+)',                             # 10⁻⁴ (coefficient = 1)
        ]
        
        for i, pattern in enumerate(patterns):
            match = re.search(pattern, value)
            if match:
                if i < 2:
                    coef = float(match.group(1))
                    exp = int(match.group(2))
                else:  # Single 10^-x format
                    coef = 1.0
                    exp = int(match.group(1))
                return coef * (10 ** -exp)
        
        # Handle ranges like "10^{-6} – 10^{-7}" or "10⁻⁶ – 10⁻⁷" - take the better (higher) value
        range_patterns = [
            r'[\$]?10[\^]?[\{]?[⁻\-](\d+)[\}]?[\$]?\s*[–\-]\s*[\$]?10[\^]?[\{]?[⁻\-](\d+)[\}]?[\$]?',  # LaTeX ranges
            r'10[⁻\-][⁰¹²³⁴⁵⁶⁷⁸⁹]+\s*[–\-]\s*10[⁻\-][⁰¹²³⁴⁵⁶⁷⁸⁹]+',  # Unicode superscript ranges like "10⁻⁶ – 10⁻⁷"
        ]
        
        for i, range_pattern in enumerate(range_patterns):
            range_match = re.search(range_pattern, value)
            if range_match:
                if i == 0:  # LaTeX format
                    exp1, exp2 = int(range_match.group(1)), int(range_match.group(2))
                else:  # Unicode superscript format
                    # Extract the superscript parts
                    unicode_match = re.search(r'10[⁻\-]([⁰¹²³⁴⁵⁶⁷⁸⁹]+)\s*[–\-]\s*10[⁻\-]([⁰¹²³⁴⁵⁶⁷⁸⁹]+)', value)
                    if unicode_match:
                        exp1 = self._unicode_superscript_to_int(unicode_match.group(1))
                        exp2 = self._unicode_superscript_to_int(unicode_match.group(2))
                    else:
                        continue
                # Take the smaller exponent (higher conductivity)
                exp = min(exp1, exp2)
                return 1.0 * (10 ** -exp)
        
        # Try direct float
        try:
            # Remove ~ and other qualifiers
            clean = re.sub(r'[~≈<>${}]', '', value)
            clean = re.sub(r's/?cm.*', '', clean).strip()
            return float(clean)
        except:
            return None
    
    def _normalize_temperature(self, value: str) -> Optional[float]:
        """Normalize temperature string to Celsius."""
        import re
        if not value:
            return None
        
        value = value.strip().lower()
        
        # Room temperature
        if 'rt' in value or 'room' in value:
            return 25.0
        
        # Kelvin
        match = re.search(r'(\d+)\s*k\b', value)
        if match:
            return float(match.group(1)) - 273.15
        
        # Celsius
        match = re.search(r'(\d+)\s*°?c', value)
        if match:
            return float(match.group(1))
        
        # Just a number (assume Celsius)
        match = re.search(r'(\d+)', value)
        if match:
            return float(match.group(1))
        
        return None
    
    def _deduplicate(self, materials: List[dict]) -> List[dict]:
        """Remove duplicate data points using normalized values and cross-references."""
        print(f"\n[Stage 3] Deduplicating {len(materials)} data points...")
        
        # Priority sorting: high confidence and primary data first
        materials.sort(key=lambda x: (
            0 if x.get('confidence') == 'high' else 1,
            0 if x.get('data_type') == 'primary' else 1
        ))

        # Remove duplicates
        unique = []
        duplicates_removed = 0

        for candidate in materials:
            is_dup = False
            c_cond = candidate.get('_norm_cond')
            c_temp = candidate.get('_norm_temp')
            # Normalize name for comparison (strip spaces/case)
            c_name = (candidate.get('electrolyte_name', {}).get('acronym') or 
                    candidate.get('electrolyte_name', {}).get('full_name') or "").lower().strip()

            for existing in unique:
                e_cond = existing.get('_norm_cond')
                e_temp = existing.get('_norm_temp')
                e_name = (existing.get('electrolyte_name', {}).get('acronym') or 
                        existing.get('electrolyte_name', {}).get('full_name') or "").lower().strip()

                # Logic: If names match and we have valid floats for both
                if c_name == e_name and c_cond is not None and e_cond is not None:
                    # 1. Conductivity: Use 5% relative tolerance
                    # 2. Temperature: Use 2°C absolute tolerance
                    cond_match = math.isclose(c_cond, e_cond, rel_tol=0.05)
                    
                    # Handle cases where temp might be None
                    temp_match = True
                    if c_temp is not None and e_temp is not None:
                        temp_match = abs(c_temp - e_temp) <= 2.0
                    
                    if cond_match and temp_match:
                        is_dup = True
                        break
            
            if is_dup:
                duplicates_removed += 1
            else:
                unique.append(candidate)
                
        print(f"  → Removed {duplicates_removed} duplicates, {len(unique)} unique points remain.")
        return unique

    # ========================================================================
    # Stage 4: Validation
    # ========================================================================
    
    def _validate_data_points(self, materials: List[dict]) -> List[dict]:
        """Validate extracted data points and flag issues."""
        print(f"\n[Stage 4] Validating {len(materials)} data points...")
        
        issues_found = 0
        
        for mat in materials:
            issues = []
            
            # Check conductivity value
            cond = self._normalize_conductivity(mat.get('ionic_conductivity_S_per_cm', ''))
            if cond is None:
                issues.append("Could not parse conductivity value")
            elif cond > 1:
                issues.append(f"Conductivity unusually high: {cond} S/cm")
            elif cond < 1e-12:
                issues.append(f"Conductivity unusually low: {cond} S/cm")
            
            # Check temperature
            temp = self._normalize_temperature(mat.get('measurement_temperature', ''))
            if temp is None and 'not specified' not in mat.get('measurement_temperature', '').lower():
                issues.append("Could not parse temperature")
            elif temp is not None and (temp < -50 or temp > 500):
                issues.append(f"Temperature out of typical range: {temp}°C")
            
            # Check for missing exact quote
            if not mat.get('exact_quote') or len(mat.get('exact_quote', '')) < 10:
                issues.append("Missing or too short exact quote")
            
            # Check confidence vs data_source consistency
            if mat.get('data_source') == 'primary' and mat.get('confidence') == 'low':
                issues.append("Primary data should not have low confidence")
            
            # Check cross-reference consistency
            quote = mat.get('exact_quote', '').lower()
            if 'figure' in quote and not mat.get('refers_to_figure'):
                issues.append("Quote mentions figure but refers_to_figure not set")
            if 'table' in quote and not mat.get('refers_to_table'):
                issues.append("Quote mentions table but refers_to_table not set")
            
            # Store validation results
            mat['_validation'] = {
                'is_valid': len(issues) == 0,
                'issues': issues
            }
            
            if issues:
                issues_found += 1
        
        print(f"  → {issues_found} data points have validation issues")
        return materials
    
    def _cross_reference_check(self, materials: List[dict]) -> List[dict]:
        """Check for cross-reference consistency across all data points."""
        print("  Checking cross-references...")
        
        # Group by figure/table references
        figure_refs: Dict[str, List[dict]] = {}
        table_refs: Dict[str, List[dict]] = {}
        
        for mat in materials:
            if mat.get('refers_to_figure'):
                fig = mat['refers_to_figure'].lower()
                if fig not in figure_refs:
                    figure_refs[fig] = []
                figure_refs[fig].append(mat)
            
            if mat.get('refers_to_table'):
                tbl = mat['refers_to_table'].lower()
                if tbl not in table_refs:
                    table_refs[tbl] = []
                table_refs[tbl].append(mat)
        
        # Check for inconsistencies within same figure/table
        for ref_type, refs in [('figure', figure_refs), ('table', table_refs)]:
            for ref_name, mats in refs.items():
                if len(mats) > 1:
                    # Check if same material has different values
                    by_material: Dict[str, List[dict]] = {}
                    for m in mats:
                        name = (m.get('electrolyte_name', {}).get('acronym') or 
                               m.get('electrolyte_name', {}).get('full_name') or 'unknown').lower()
                        if name not in by_material:
                            by_material[name] = []
                        by_material[name].append(m)
                    
                    for name, same_mat in by_material.items():
                        if len(same_mat) > 1:
                            conds = [self._normalize_conductivity(m.get('ionic_conductivity_S_per_cm', '')) for m in same_mat]
                            conds = [c for c in conds if c is not None]
                            if conds and max(conds) / min(conds) > 2:
                                for m in same_mat:
                                    m['_validation']['issues'].append(
                                        f"Inconsistent values for {name} in {ref_type} {ref_name}"
                                    )
                                    m['_validation']['is_valid'] = False
        
        return materials

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
        # 0. Index Figures First
        self._index_figures(base_path, structure)
        self.processed_images = set()

        # Collect all nodes
        text_nodes = self._collect_all_nodes(structure)
        image_nodes = self._collect_image_nodes(structure)
        print(f"Found {len(text_nodes)} text nodes and {len(image_nodes)} image nodes.")
        
        # if not all_nodes:
        #     return {'materials': [], 'stats': {'total_nodes': 0}}
        
        # Stage 1: Filter relevant nodes
        relevant_text_nodes = asyncio.run(self._filter_relevant_nodes(text_nodes, batch_size))

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
            return {'materials': [], 'stats': {'total_nodes': len(all_nodes), 'relevant_nodes': 0}}
        
        # # Stage 2: Extract from relevant nodes
        # materials = asyncio.run(self._extract_from_nodes(relevant_nodes, batch_size))

        # # Stage 2.5: Extract from leftover images
        # orphan_materials = asyncio.run(self._process_leftover_images(batch_size))
        # materials.extend(orphan_materials)
        
        # if not materials:
        #     print("No data points extracted!")
        #     return {'materials': [], 'stats': {
        #         'total_nodes': len(all_nodes),
        #         'relevant_nodes': len(relevant_nodes),
        #         'extracted': 0
        #     }}
        
        # Stage 3: Normalize & Deduplicate
        normalizer = ScientificNormalizer(self.client, model_name=self.model_text)
        materials_with_floats = asyncio.run(normalizer.normalize_batch(combined_materials))
        unique_materials = self._deduplicate(materials_with_floats)
        
        # Stage 4: Validate
        nodes_map = {n['node_id']: n for n in relevant_text_nodes + image_nodes}
        validator = DataValidator(self.client, self.figure_index, model_name=self.model_text)
        validated_materials = validator.validate_all(unique_materials, nodes_map)
        
        # Compile stats
        stats = {
            # 'total_nodes': len(all_nodes),
            # 'relevant_nodes': len(relevant_nodes),
            'raw_extracted': len(combined_materials),
            'after_dedup': len(unique_materials),
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
    parser.add_argument('--model', default='gemini-2.5-flash-lite', help='LLM model to use')
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
    
    # Run extraction pipeline
    base_path = Path(args.asset_dir)
    print(base_path)
    extractor = MaterialExtractor(model_text=args.model, model_vision="gemini-3-flash-preview")
    result = extractor.extract(structure, base_path=base_path, batch_size=args.batch_size)
    
    materials = result['materials']
    stats = result['stats']
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_materials.json"
    
    # Save results
    output_data = {
        'doc_name': doc_name,
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
    print(f"\nStatistics:")
    print(f"  Nodes analyzed: {stats.get('total_nodes', 0)}")
    print(f"  Relevant nodes: {stats.get('relevant_nodes', 0)}")
    print(f"  Raw data points: {stats.get('raw_extracted', 0)}")
    print(f"  After dedup: {stats.get('after_dedup', 0)}")
    print(f"  Valid: {stats.get('valid_count', 0)}, Invalid: {stats.get('invalid_count', 0)}")
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
