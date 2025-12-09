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

class MaterialInfo(BaseModel):
    """Individual material extracted from text."""
    abbreviation: str = Field(..., description="Short name or abbreviation (e.g., 'PEO', 'LLZO')")
    full_name: str = Field(default="", description="Full chemical/material name if mentioned")
    material_type: str = Field(default="", description="Type: polymer, ceramic, composite, salt, etc.")
    composition: str = Field(default="", description="Compositional details if mentioned (e.g., '90:10 wt%')")
    processing_method: str = Field(default="", description="How the material was prepared/processed")
    


class MaterialExtractionResponse(BaseModel):
    """Response containing materials extracted from a node."""
    materials: List[MaterialInfo] = Field(default_factory=list)


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
    
    def _build_extraction_prompt(self) -> str:
        return """Extract all materials mentioned in this text from a scientific paper about battery electrolytes.

For each material, identify:
1. abbreviation: Short name (e.g., "PEO", "LLZO", "LiTFSI")
2. full_name: Full chemical name if stated (e.g., "poly(ethylene oxide)")
3. material_type: Category - one of: polymer, ceramic, composite, salt, filler, additive, solvent, other
4. composition: Any compositional details (ratios, percentages, formulations)
5. processing_method: How it was prepared (e.g., "solution casting", "hot pressing", "ball milling")

Context - Section: {section_title}
Parent Context: {parent_context}

Text to analyze:
{text}

Rules:
- Only extract materials explicitly mentioned in the text
- If full_name is not stated, leave it empty
- If processing_method is not mentioned for a material, leave it empty
- Include composite formulations as separate materials (e.g., "LLZO-PEO" is distinct from "LLZO" and "PEO")
- Be precise with abbreviations - don't guess

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

    def _extract_from_node_sync(self, node: dict) -> List[dict]:
        """Extract materials from a single node using LLM (synchronous)."""
        from google.genai import types
        
        prompt = self.extraction_prompt.format(
            section_title=node.get('section_title', 'Unknown'),
            parent_context=node.get('parent_title', ''),
            text=node.get('text', '')[:4000]  # Limit text length
        )
        
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                    response_json_schema=MaterialExtractionResponse.model_json_schema()
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
            
            return materials
            
        except Exception as e:
            print(f"    Error: {e}")
            return []



    def _merge_materials(self, all_materials: List[dict]) -> List[dict]:
        """
        Merge duplicate materials from different nodes.
        
        Groups by abbreviation and consolidates information.
        """
        material_map: Dict[str, dict] = {}
        
        for mat in all_materials:
            abbrev = mat.get('abbreviation', '').strip()
            if not abbrev:
                continue
            
            # Normalize key (case-insensitive)
            key = abbrev.lower()
            
            if key not in material_map:
                material_map[key] = {
                    'abbreviation': abbrev,
                    'full_name': mat.get('full_name', ''),
                    'material_type': mat.get('material_type', ''),
                    'compositions': set(),
                    'processing_methods': set(),
                    'source_nodes': []
                }
            
            entry = material_map[key]
            
            # Update full_name if we have a better one
            if mat.get('full_name') and not entry['full_name']:
                entry['full_name'] = mat['full_name']
            
            # Update material_type if we have one
            if mat.get('material_type') and not entry['material_type']:
                entry['material_type'] = mat['material_type']
            
            # Collect compositions
            if mat.get('composition'):
                entry['compositions'].add(mat['composition'])
            
            # Collect processing methods
            if mat.get('processing_method'):
                entry['processing_methods'].add(mat['processing_method'])
            
            # Track source nodes
            if mat.get('source_node'):
                # Avoid duplicate source nodes
                node_id = mat['source_node'].get('node_id', '')
                existing_ids = [n.get('node_id') for n in entry['source_nodes']]
                if node_id and node_id not in existing_ids:
                    entry['source_nodes'].append(mat['source_node'])
        
        # Convert sets to lists for JSON serialization
        merged = []
        for entry in material_map.values():
            merged.append({
                'abbreviation': entry['abbreviation'],
                'full_name': entry['full_name'],
                'material_type': entry['material_type'],
                'compositions': list(entry['compositions']),
                'processing_methods': list(entry['processing_methods']),
                'source_nodes': entry['source_nodes']
            })
        
        # Sort by abbreviation
        merged.sort(key=lambda x: x['abbreviation'].lower())
        
        return merged

    def extract(self, structure: List[dict]) -> List[dict]:
        """
        Extract materials from the entire tree structure.
        
        Args:
            structure: PageIndex tree structure
            
        Returns:
            List of merged material dictionaries
        """
        # Collect nodes to process
        nodes = self._collect_nodes_for_extraction(structure)
        print(f"Found {len(nodes)} nodes to analyze for materials")
        
        if not nodes:
            return []
        
        # Process nodes sequentially with progress
        all_materials = []
        total = len(nodes)
        
        for i, node in enumerate(nodes):
            print(f"  [{i+1}/{total}] {node.get('title', 'Unknown')[:50]}...", end=" ", flush=True)
            try:
                materials = self._extract_from_node_sync(node)
                all_materials.extend(materials)
                print(f"✓ {len(materials)} materials")
            except Exception as e:
                print(f"✗ {e}")
        
        print(f"\nFound {len(all_materials)} material mentions across all nodes")
        
        # Merge duplicates
        merged = self._merge_materials(all_materials)
        print(f"Consolidated to {len(merged)} unique materials")
        
        return merged


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
    materials = extractor.extract(structure)
    
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
        full = f" ({mat['full_name']})" if mat['full_name'] else ""
        mtype = f" [{mat['material_type']}]" if mat['material_type'] else ""
        print(f"  - {mat['abbreviation']}{full}{mtype}")
    
    if len(materials) > 15:
        print(f"  ... and {len(materials) - 15} more")
    
    return 0


if __name__ == '__main__':
    exit(main())
