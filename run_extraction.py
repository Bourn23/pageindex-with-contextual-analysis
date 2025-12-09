#!/usr/bin/env python3
"""
Material extraction from PageIndex JSON structures.

Extracts materials, their full names, processing methods, and source nodes
using LLM-assisted extraction with hierarchical context windows.

Usage:
    python run_extraction.py results/paper_keywords_structure.json
    python run_extraction.py results/paper_keywords_structure.json --output materials.json
"""

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import List, Dict
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


# ============================================================================
# Pydantic Models for Structured Output
# ============================================================================

# class MaterialInfo(BaseModel):
#     """Individual material extracted from text."""
#     abbreviation: str = Field(..., description="Short name or abbreviation (e.g., 'PEO', 'LLZO')")
#     full_name: str = Field(default="", description="Full chemical/material name if mentioned")
#     material_type: str = Field(default="", description="Type: polymer, ceramic, composite, salt, etc.")
#     composition: str = Field(default="", description="Compositional details if mentioned (e.g., '90:10 wt%')")
#     processing_method: str = Field(default="", description="How the material was prepared/processed")



# class MaterialExtractionResponse(BaseModel):
#     """Response containing materials extracted from a node."""
#     materials: List[MaterialInfo] = Field(default_factory=list)

from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Type

class ElectrolyteName(BaseModel):
    """A structured representation of the electrolyte's name."""
    full_name: str = Field(
        ...,
        description="The complete, formal name of the electrolyte or its major components (e.g., Lithium Lanthanum Zirconate Oxide)."
    )
    acronym: Optional[str] = Field(
        None,
        description="The associated abbreviation, chemical formula, or common name (e.g., LLZO, PEO)."
    )
    proportion: Optional[str] = Field(
        None,
        description="The relative stoichiometric ratios, molar ratios, or concentrations of components."
    )

class IonicConductivityDataPoint(BaseModel):
    """Represents a single extracted data point for ionic conductivity."""
    material_class: str = Field(
        ...,
        description="The primary functional class of the material: Ceramic, Polymer, Composite, or Other."
    )
    electrolyte_name: ElectrolyteName = Field(
        ...,
        description="Structured name of the electrolyte, including its full name, acronym, and component proportions."
    )
    ionic_conductivity_S_per_cm: str = Field(
        ...,
        description="Ionic conductivity value in S cm⁻¹. May include qualitative notes."
    )
    measurement_temperature: str = Field(
        ...,
        description="The temperature at which the measurement was taken (°C, K, or 'RT')."
    )
    specific_source_location: str = Field(
        ...,
        description="Precise location within the source document where the data was found."
    )
    material_description: str = Field(
        ...,
        description="Description of the material's properties, or 'N/A (Cited Work)' if not the primary material of the study."
    )
    processing_method: str = Field(
        ...,
        description="Synthesis and fabrication steps, or 'N/A (Cited Work)' if not the primary material of the study."
    )    

class MaterialExtractionResponse(BaseModel):
    """Response containing materials extracted from a node."""
    materials: List[IonicConductivityDataPoint] = Field(default_factory=list)


class NormalizedValue(BaseModel):
    """Normalized temperature or conductivity value."""
    value: float = Field(..., description="Numeric value")
    unit: str = Field(..., description="Unit (e.g., 'celsius', 'S/cm')")
    original: str = Field(..., description="Original string representation")
    confidence: str = Field(..., description="high, medium, or low")


class NormalizationResponse(BaseModel):
    """Response containing normalized values."""
    temperature_celsius: Optional[float] = Field(None, description="Temperature in Celsius, or null if not specified")
    conductivity_s_per_cm: Optional[float] = Field(None, description="Conductivity in S/cm, or null if invalid")
    notes: str = Field(default="", description="Any parsing notes or warnings")

# ============================================================================
# Material Extractor
# ============================================================================

class MaterialExtractor:
    """Extracts material information from PageIndex tree structures."""
    
    def __init__(self, model: str = "gemini-2.5-flash-lite"):
        self.model = model
        
        # Initialize Gemini client
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        self.client = genai.Client(api_key=api_key)
        
        self.extraction_prompt = self._build_extraction_prompt()
        self.normalization_prompt = self._build_normalization_prompt()
    
    def _build_extraction_prompt(self) -> str:
        return """Extract ionic conductivity data points from this text from a scientific paper about battery electrolytes.

For each ionic conductivity measurement mentioned, extract:

1. material_class: The primary functional class - one of: Ceramic, Polymer, Composite, or Other
2. electrolyte_name: Structured name with three parts:
   - full_name: Complete formal name (e.g., "Lithium Lanthanum Zirconate Oxide")
   - acronym: Abbreviation or formula (e.g., "LLZO", "PEO")
   - proportion: Ratios or concentrations if mentioned (e.g., "90:10 wt%", "0.1 M")
3. ionic_conductivity_S_per_cm: The conductivity value in S cm⁻¹ (e.g., "1.2 × 10⁻⁴", "~10⁻⁵")
4. measurement_temperature: Temperature of measurement (e.g., "25°C", "RT", "room temperature")
5. specific_source_location: Where in the document this data appears (e.g., "Figure 3", "Table 2", "main text")
6. material_description: Properties and characteristics of the material, or "N/A (Cited Work)" if from a reference
7. processing_method: How the material was prepared, or "N/A (Cited Work)" if from a reference

Context - Section: {section_title}
Parent Context: {parent_context}

Text to analyze:
{text}

Rules:
- Only extract data points explicitly mentioned in the text
- For materials from the current study, provide detailed descriptions and processing methods
- For materials from cited references, use "N/A (Cited Work)" for description and processing
- If temperature is not stated, use "Not specified"
- Be precise with conductivity values - include scientific notation and units
- If a material has multiple measurements at different temperatures, create separate data points

Respond with JSON only."""
    
    def _build_normalization_prompt(self) -> str:
        return """Normalize temperature and conductivity values to standard units.

Given:
- Temperature: {temperature}
- Conductivity: {conductivity}

Convert to:
1. temperature_celsius: Temperature in °C (float), or null if "Not specified" or invalid
   - "RT" or "room temperature" → 25.0
   - "25°C" or "25 °C" → 25.0
   - "298K" → 25.0 (convert from Kelvin)
   - "Not specified" → null

2. conductivity_s_per_cm: Conductivity in S/cm (float), or null if invalid
   - "1.2 × 10⁻⁴" → 0.00012
   - "1.2e-4" → 0.00012
   - "~10⁻⁵" → 0.00001
   - "10^-4" → 0.0001
   - Handle ranges by taking midpoint: "10⁻⁵ - 10⁻⁸" → 0.0000055

3. notes: Any warnings or parsing issues

Respond with JSON only."""

    def _collect_nodes_for_extraction(self, structure: List[dict]) -> List[dict]:
        """
        Collect nodes suitable for material extraction.
        
        Focuses on semantic_unit and keyword nodes which have focused content.
        """
        nodes_to_process = []
        
        def traverse(nodes: List[dict], parent_title: str = "", section_title: str = ""):
            for node in nodes:
                node_type = node.get('node_type', 'section')
                title = node.get('title', '')
                text = node.get('text', '')
                
                # Track section context
                current_section = section_title
                if node_type == 'section' or (not section_title and title):
                    current_section = title
                
                # Collect semantic_unit nodes (best for extraction)
                if node_type == 'semantic_unit' and text:
                    
                    # Optional [uncomment]: Add filter for getting pre-keyword nodes
                    # child_nodes = node.get('node', [])
                    # has_keywords = any(n.get('node_type') == 'keyword' for n in child_nodes)
                    # if has_keywords:

                    nodes_to_process.append({
                        'node_id': node.get('node_id', ''),
                        'title': title,
                        'text': text,
                        'node_type': node_type,
                        'section_title': current_section,
                        'parent_title': parent_title,
                        'summary': node.get('summary', ''),
                        'metadata': node.get('metadata', {})
                    })
                
                # Also collect section nodes with substantial text (no children or leaf sections)
                elif node_type == 'section' and text and len(text) > 200:
                    child_nodes = node.get('nodes', [])
                    # Only if it doesn't have semantic_unit children
                    has_semantic_children = any(
                        n.get('node_type') == 'semantic_unit' for n in child_nodes
                    )
                    if not has_semantic_children:
                        nodes_to_process.append({
                            'node_id': node.get('node_id', ''),
                            'title': title,
                            'text': text[:5000],  # Limit text length
                            'node_type': node_type,
                            'section_title': current_section,
                            'parent_title': parent_title,
                            'summary': node.get('summary', ''),
                            'metadata': node.get('metadata', {})
                        })
                
                # Recurse into children
                if 'nodes' in node and node['nodes']:
                    traverse(node['nodes'], title, current_section)
        
        traverse(structure)
        return nodes_to_process

    async def _extract_from_node_async(self, node: dict, semaphore: asyncio.Semaphore) -> tuple[str, List[dict]]:
        """Extract materials from a single node using LLM (async with rate limiting)."""
        from google.genai import types
        
        prompt = self.extraction_prompt.format(
            section_title=node.get('section_title', 'Unknown'),
            parent_context=node.get('parent_title', ''),
            text=node.get('text', '')
        )
        
        node_title = node.get('title', 'Unknown')[:50]
        
        async with semaphore:  # Rate limiting
            try:
                # Run blocking API call in thread pool
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.client.models.generate_content(
                        model=self.model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=0,
                            response_mime_type="application/json",
                            response_json_schema=MaterialExtractionResponse.model_json_schema()
                        )
                    )
                )
                
                result = MaterialExtractionResponse.model_validate_json(response.text)
                
                # Add source node info to each material
                materials = []
                for mat in result.materials:
                    mat_dict = mat.model_dump()
                    mat_dict['source_node'] = {
                        'node_id': node.get('node_id', ''),
                        'title': node.get('title', ''),
                        'node_type': node.get('node_type', ''),
                        'section': node.get('section_title', '')
                    }
                    materials.append(mat_dict)
                
                return (node_title, materials)
                
            except Exception as e:
                return (node_title, [])
    
    async def _normalize_values_async(self, materials: List[dict], semaphore: asyncio.Semaphore) -> List[dict]:
        """Normalize temperature and conductivity values using LLM."""
        from google.genai import types
        
        if not materials:
            return materials
        
        print(f"\nNormalizing {len(materials)} data points...")
        
        async def normalize_single(mat: dict) -> dict:
            async with semaphore:
                try:
                    prompt = self.normalization_prompt.format(
                        temperature=mat.get('measurement_temperature', 'Not specified'),
                        conductivity=mat.get('ionic_conductivity_S_per_cm', '')
                    )
                    
                    loop = asyncio.get_event_loop()
                    response = await loop.run_in_executor(
                        None,
                        lambda: self.client.models.generate_content(
                            model=self.model,
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                temperature=0,
                                response_mime_type="application/json",
                                response_json_schema=NormalizationResponse.model_json_schema()
                            )
                        )
                    )
                    
                    result = NormalizationResponse.model_validate_json(response.text)
                    
                    # Add normalized values to material
                    mat['normalized'] = {
                        'temperature_celsius': result.temperature_celsius,
                        'conductivity_s_per_cm': result.conductivity_s_per_cm,
                        'notes': result.notes
                    }
                    
                    return mat
                    
                except Exception as e:
                    # Keep original if normalization fails
                    mat['normalized'] = {
                        'temperature_celsius': None,
                        'conductivity_s_per_cm': None,
                        'notes': f"Normalization failed: {str(e)}"
                    }
                    return mat
        
        # Normalize all materials concurrently
        tasks = [normalize_single(mat) for mat in materials]
        normalized = []
        
        for coro in asyncio.as_completed(tasks):
            result = await coro
            normalized.append(result)
        
        return normalized



    def _merge_materials(self, all_materials: List[dict]) -> List[dict]:
        """
        Process ionic conductivity data points.
        
        Uses normalized values for deduplication when available.
        """
        # Remove exact duplicates based on normalized values (or original if normalization failed)
        seen = set()
        unique_materials = []
        
        for mat in all_materials:
            electrolyte = mat.get('electrolyte_name', {})
            normalized = mat.get('normalized', {})
            
            # Use normalized values for deduplication if available
            temp = normalized.get('temperature_celsius')
            cond = normalized.get('conductivity_s_per_cm')
            
            # Fallback to original strings if normalization failed
            if temp is None:
                temp = mat.get('measurement_temperature', '')
            if cond is None:
                cond = mat.get('ionic_conductivity_S_per_cm', '')
            
            key = (
                electrolyte.get('acronym', ''),
                electrolyte.get('full_name', ''),
                str(cond),  # Convert to string for hashing
                str(temp),
                mat.get('source_node', {}).get('node_id', '')
            )
            
            if key not in seen:
                seen.add(key)
                unique_materials.append(mat)
        
        # Sort by material acronym/name
        unique_materials.sort(key=lambda x: (
            x.get('electrolyte_name', {}).get('acronym', '') or 
            x.get('electrolyte_name', {}).get('full_name', '')
        ).lower())
        
        return unique_materials

    async def _extract_batch_async(self, nodes: List[dict], batch_size: int = 10) -> List[dict]:
        """
        Extract materials from multiple nodes concurrently.
        
        Args:
            nodes: List of nodes to process
            batch_size: Max concurrent API calls
            
        Returns:
            List of all extracted materials
        """
        semaphore = asyncio.Semaphore(batch_size)
        total = len(nodes)
        
        print(f"Processing {total} nodes with batch size {batch_size}...")
        
        # Create tasks for all nodes
        tasks = [self._extract_from_node_async(node, semaphore) for node in nodes]
        
        # Process with progress tracking
        all_materials = []
        completed = 0
        
        for coro in asyncio.as_completed(tasks):
            node_title, materials = await coro
            completed += 1
            all_materials.extend(materials)
            
            status = "✓" if materials else "○"
            print(f"  [{completed}/{total}] {status} {node_title}... ({len(materials)} data points)")
        
        return all_materials
    
    def extract(self, structure: List[dict], batch_size: int = 10) -> List[dict]:
        """
        Extract materials from the entire tree structure.
        
        Args:
            structure: PageIndex tree structure
            batch_size: Max concurrent API calls (default: 10)
            
        Returns:
            List of merged material dictionaries
        """
        # Collect nodes to process
        nodes = self._collect_nodes_for_extraction(structure)
        print(f"Found {len(nodes)} nodes to analyze for materials")
        
        if not nodes:
            return []
        
        # Process nodes concurrently
        all_materials = asyncio.run(self._extract_batch_async(nodes, batch_size))
        
        print(f"\nFound {len(all_materials)} ionic conductivity data points across all nodes")
        
        # Normalize values
        semaphore = asyncio.Semaphore(batch_size)
        normalized_materials = asyncio.run(self._normalize_values_async(all_materials, semaphore))
        
        # Remove duplicates (using normalized values)
        unique = self._merge_materials(normalized_materials)
        print(f"Consolidated to {len(unique)} unique data points")
        
        return unique


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Extract materials from PageIndex JSON structure'
    )
    parser.add_argument('input_file', help='Path to PageIndex JSON file')
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
    
    # Extract materials
    extractor = MaterialExtractor(model=args.model)
    materials = extractor.extract(structure, batch_size=args.batch_size)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_materials.json"
    
    # Save results
    output_data = {
        'doc_name': doc_name,
        'source_file': str(input_path),
        'material_count': len(materials),
        'materials': materials
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print("=" * 70)
    print(f"✓ Extracted {len(materials)} materials")
    print(f"✓ Saved to: {output_path}")
    
    # Print summary
    print("\nMaterials found:")
    for mat in materials[:15]:  # Show first 15
        full = f" ({mat['electrolyte_name']['full_name']})" if mat['electrolyte_name']['full_name'] else ""
        mtype = f" [{mat['material_class']}]" if mat['material_class'] else ""
        print(f"  - {mat['electrolyte_name']['acronym']}{full}{mtype}")
    
    if len(materials) > 15:
        print(f"  ... and {len(materials) - 15} more")
    
    return 0


if __name__ == '__main__':
    exit(main())
