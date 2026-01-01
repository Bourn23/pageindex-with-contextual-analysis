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
        
        def find_images(nodes):
            for node in nodes:
                text = node.get('text', '')
                matches = re.findall(r'!\[.*?\]\((.*?)\)', text)
                
                for image_rel_path in matches:
                    # Regex looks for "Fig" or "Figure" followed by a number
                    fig_match = re.search(r'(?:fig|figure)[-_]?(\d+)', image_rel_path, re.IGNORECASE)
                    
                    if fig_match:
                        fig_num = fig_match.group(1) # e.g., "2"
                        full_path = base_path / image_rel_path
                        
                        if full_path.exists():
                            if fig_num not in self.figure_index:
                                self.figure_index[fig_num] = []
                            
                            # Avoid adding the exact same file path twice
                            if full_path not in self.figure_index[fig_num]:
                                self.figure_index[fig_num].append(full_path)
                                print(f"    Indexed Figure {fig_num} -> {image_rel_path}")
                
                if 'nodes' in node:
                    find_images(node['nodes'])

        find_images(structure)
        
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
                            model=self.model,
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                temperature=0,
                                max_output_tokens=4096,
                                response_mime_type="application/json",
                                response_json_schema=NodeRelevanceResponse.model_json_schema()
                            )
                        )
                    ),
                    timeout=timeout
                )
                
                result = NodeRelevanceResponse.model_validate_json(response.text)
                return (node, result.is_relevant, result.relevance_reason)
            
            except asyncio.TimeoutError:
                # On timeout, include the node (be permissive)
                return (node, True, "Timeout - including by default")
            except Exception as e:
                # On error, include the node (be permissive)
                return (node, True, f"Error checking relevance: {e}")
    
    async def _filter_relevant_nodes(self, nodes: List[dict], batch_size: int = 10) -> List[dict]:
        """Filter nodes to only those likely containing ionic conductivity data."""
        print(f"\n[Stage 1] Filtering {len(nodes)} nodes for relevance...")
        
        semaphore = asyncio.Semaphore(batch_size)
        tasks = [self._check_node_relevance(node, semaphore) for node in nodes]
        
        relevant_nodes = []
        completed = 0
        
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
        # 2. Build content payload
        prompt_text = self._build_extraction_prompt(node)
        
        # raw_contents = [
        #     types.Content(
        #         parts=[
        #             types.Part(text=prompt_text)
        #         ]
        #     )
        # ]

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
        
        # Add normalized values
        for mat in materials:
            mat['_norm_cond'] = self._normalize_conductivity(mat.get('ionic_conductivity_S_per_cm', ''))
            mat['_norm_temp'] = self._normalize_temperature(mat.get('measurement_temperature', ''))
        
        # Group by material name
        groups: Dict[str, List[dict]] = {}
        for mat in materials:
            electrolyte = mat.get('electrolyte_name', {})
            key = (electrolyte.get('acronym') or electrolyte.get('full_name') or 'unknown').lower()
            if key not in groups:
                groups[key] = []
            groups[key].append(mat)
        
        unique = []
        duplicates_removed = 0
        
        for material_name, candidates in groups.items():
            if len(candidates) == 1:
                unique.append(candidates[0])
                continue
            
            # Find duplicates within this material group
            seen = set()
            for mat in candidates:
                # Create dedup key from normalized values
                cond = mat['_norm_cond']
                temp = mat['_norm_temp']
                
                # Round for fuzzy matching (within 5% for conductivity, 2°C for temp)
                cond_key = round(cond, 6) if cond else mat.get('ionic_conductivity_S_per_cm', '')
                temp_key = round(temp / 2) * 2 if temp else mat.get('measurement_temperature', '')
                
                key = (cond_key, temp_key)
                
                if key not in seen:
                    seen.add(key)
                    unique.append(mat)
                else:
                    duplicates_removed += 1
        
        # Clean up temp fields
        for mat in unique:
            mat.pop('_norm_cond', None)
            mat.pop('_norm_temp', None)
        
        # Sort by material name
        unique.sort(key=lambda x: (
            x.get('electrolyte_name', {}).get('acronym') or 
            x.get('electrolyte_name', {}).get('full_name') or ''
        ).lower())
        
        print(f"  → Removed {duplicates_removed} duplicates, {len(unique)} unique data points remain")
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
    
    def extract(self, structure: List[dict], base_path: Path, batch_size: int = 7) -> dict:
        """
        Run the full four-stage extraction pipeline.
        
        Returns:
            dict with 'materials' list and 'stats' summary
        """
        # 0. Index Figures First
        self._index_figures(base_path, structure)

        # Collect all nodes
        all_nodes = self._collect_all_nodes(structure)
        print(f"Found {len(all_nodes)} semantic_unit nodes in structure")
        
        if not all_nodes:
            return {'materials': [], 'stats': {'total_nodes': 0}}
        
        # Stage 1: Filter relevant nodes
        relevant_nodes = asyncio.run(self._filter_relevant_nodes(all_nodes, batch_size))
        
        if not relevant_nodes:
            print("No relevant nodes found!")
            return {'materials': [], 'stats': {'total_nodes': len(all_nodes), 'relevant_nodes': 0}}
        
        # Stage 2: Extract from relevant nodes
        materials = asyncio.run(self._extract_from_nodes(relevant_nodes, batch_size))
        
        if not materials:
            print("No data points extracted!")
            return {'materials': [], 'stats': {
                'total_nodes': len(all_nodes),
                'relevant_nodes': len(relevant_nodes),
                'extracted': 0
            }}
        
        # Stage 3: Deduplicate
        unique_materials = self._deduplicate(materials)
        
        # Stage 4: Validate
        validated_materials = self._validate_data_points(unique_materials)
        validated_materials = self._cross_reference_check(validated_materials)
        
        # Compile stats
        stats = {
            'total_nodes': len(all_nodes),
            'relevant_nodes': len(relevant_nodes),
            'raw_extracted': len(materials),
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
            }
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
    
    return 0


if __name__ == '__main__':
    exit(main())
