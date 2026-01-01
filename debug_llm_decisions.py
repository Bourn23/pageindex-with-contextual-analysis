#!/usr/bin/env python3
"""
Debug script to show exactly what the LLM sees and how it decides.

This script demonstrates:
1. What input is fed to the LLM for relevance checking
2. What the LLM responds with
3. What input is fed for data extraction
4. What structured data comes back

Usage:
    python debug_llm_decisions.py results/test-results2.json
"""

import json
import os
import asyncio
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

# Import the extraction classes
from run_extraction_md import MaterialExtractor, NodeRelevanceResponse, MaterialExtractionResponse

class LLMDecisionDebugger:
    """Shows exactly what the LLM sees and decides."""
    
    def __init__(self):
        self.extractor = MaterialExtractor()
    
    def collect_sample_nodes(self, structure: List[dict], max_nodes: int = 3) -> List[dict]:
        """Collect a few sample nodes for debugging - mix of relevant and non-relevant."""
        nodes = self.extractor._collect_all_nodes(structure)
        
        # Try to get a good mix
        sample_nodes = []
        
        # Look for obviously relevant ones (results, EIS, conductivity in title)
        relevant_candidates = [n for n in nodes if any(keyword in n.get('title', '').lower() 
                              for keyword in ['eis', 'impedance', 'electrochemical'])]
        
        # Look for obviously non-relevant ones (introduction, conclusion)
        non_relevant_candidates = [n for n in nodes if any(keyword in n.get('title', '').lower() 
                                  for keyword in ['introduction', 'conclusion', 'abstract'])]
        
        # Add the EIS node first (most likely to have data)
        eis_node = next((n for n in nodes if 'electrochemical impedance' in n.get('title', '').lower()), None)
        if eis_node:
            sample_nodes.append(eis_node)
        
        # Add one non-relevant if available
        if non_relevant_candidates and len(sample_nodes) < max_nodes:
            sample_nodes.append(non_relevant_candidates[0])
        
        # Fill remaining slots with any nodes
        remaining_slots = max_nodes - len(sample_nodes)
        for node in nodes:
            if node not in sample_nodes and remaining_slots > 0:
                sample_nodes.append(node)
                remaining_slots -= 1
        
        return sample_nodes[:max_nodes]
    
    async def debug_relevance_decision(self, node: dict) -> dict:
        """Show the relevance checking process step by step."""
        print("=" * 80)
        print("🔍 STAGE 1: RELEVANCE CHECKING")
        print("=" * 80)
        
        # Show what we're feeding to the LLM
        keyword_list = [kw['term'] for kw in node.get('keywords', [])]
        keywords_str = ', '.join(keyword_list) if keyword_list else 'None'
        
        print(f"📍 Node ID: {node.get('node_id', 'Unknown')}")
        print(f"📍 Section: {node.get('section_title', 'Unknown')}")
        print(f"📍 Title: {node.get('title', 'Unknown')}")
        print(f"📍 Text Length: {len(node.get('text', ''))} characters")
        print(f"📍 Keywords: {keywords_str}")
        print()
        
        # Build the exact prompt that goes to the LLM
        prompt = f"""Determine if this section from a scientific paper contains ionic conductivity measurements with numerical values.

Section Title: {node.get('title', 'Unknown')}
Section Summary: {node.get('summary', 'No summary')}
Keywords in this section: {keywords_str}

Answer these questions:
1. is_relevant: Does this section likely contain ionic conductivity measurements (numerical values in S/cm or similar units)?
2. relevance_reason: Brief explanation (1 sentence)
3. expected_data_points: How many distinct measurements might be in this section? (0 if not relevant)

Consider:
- Sections about "results", "conductivity", "impedance", "EIS" are likely relevant
- Sections about "introduction", "motivation", "references" are usually NOT relevant (unless they cite specific values)
- Look for keywords like: conductivity, S/cm, impedance, measurement, temperature

Respond with JSON only."""

        print("📤 PROMPT SENT TO LLM:")
        print("-" * 40)
        print(prompt)
        print("-" * 40)
        print()
        
        # Get the LLM response
        try:
            from google.genai import types
            
            response = self.extractor.client.models.generate_content(
                model=self.extractor.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=4096,
                    response_mime_type="application/json",
                    response_json_schema=NodeRelevanceResponse.model_json_schema()
                )
            )
            
            print("📥 RAW LLM RESPONSE:")
            print("-" * 40)
            print(response.text)
            print("-" * 40)
            print()
            
            # Parse the response
            result = NodeRelevanceResponse.model_validate_json(response.text)
            
            print("🧠 LLM DECISION:")
            print(f"  ✅ Relevant: {result.is_relevant}")
            print(f"  💭 Reasoning: {result.relevance_reason}")
            print(f"  📊 Expected Data Points: {result.expected_data_points}")
            print()
            
            return {
                'node': node,
                'prompt': prompt,
                'raw_response': response.text,
                'parsed_decision': result.model_dump(),
                'is_relevant': result.is_relevant
            }
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return {
                'node': node,
                'prompt': prompt,
                'error': str(e),
                'is_relevant': False
            }
    
    async def debug_extraction_decision(self, node: dict) -> dict:
        """Show the data extraction process step by step."""
        print("=" * 80)
        print("🔬 STAGE 2: DATA EXTRACTION")
        print("=" * 80)
        
        # Show the text being analyzed
        text = node.get('text', '')
        print(f"📍 Analyzing Node: {node.get('title', 'Unknown')}")
        print(f"📍 Text Length: {len(text)} characters")
        print()
        
        print("📄 FULL TEXT BEING ANALYZED:")
        print("-" * 40)
        print(text[:1000] + ("..." if len(text) > 1000 else ""))
        print("-" * 40)
        print()
        
        # Build extraction prompt
        keywords_str = ', '.join([kw['term'] for kw in node.get('keywords', [])]) or 'None'
        
        prompt = f"""Extract ALL ionic conductivity measurements from this text.

Section: {node.get('section_title', 'Unknown')}
Title: {node.get('title', 'Unknown')}
Keywords: {keywords_str}

Text:
{text[:6000]}

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

        print("📤 EXTRACTION PROMPT SENT TO LLM:")
        print("-" * 40)
        print(prompt[:800] + "..." if len(prompt) > 800 else prompt)
        print("-" * 40)
        print()
        
        # Get extraction response
        try:
            from google.genai import types
            
            response = self.extractor.client.models.generate_content(
                model=self.extractor.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=8192,
                    response_mime_type="application/json",
                    response_json_schema=MaterialExtractionResponse.model_json_schema()
                )
            )
            
            print("📥 RAW EXTRACTION RESPONSE:")
            print("-" * 40)
            print(response.text)
            print("-" * 40)
            print()
            
            # Parse the response
            result = MaterialExtractionResponse.model_validate_json(response.text)
            
            print("🔬 EXTRACTED DATA POINTS:")
            for i, material in enumerate(result.materials, 1):
                print(f"  📊 Data Point {i}:")
                print(f"    Material: {material.electrolyte_name.acronym or material.electrolyte_name.full_name}")
                print(f"    Conductivity: {material.ionic_conductivity_S_per_cm}")
                print(f"    Temperature: {material.measurement_temperature}")
                print(f"    Confidence: {material.confidence}")
                print(f"    Quote: \"{material.exact_quote[:100]}...\"")
                print()
            
            return {
                'node': node,
                'prompt': prompt,
                'raw_response': response.text,
                'extracted_materials': [m.model_dump() for m in result.materials],
                'count': len(result.materials)
            }
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return {
                'node': node,
                'prompt': prompt,
                'error': str(e),
                'count': 0
            }
    
    async def debug_full_pipeline(self, structure_file: str):
        """Run the full debugging pipeline."""
        print("🚀 LLM DECISION DEBUGGING PIPELINE")
        print("=" * 80)
        
        # Load structure
        with open(structure_file, 'r') as f:
            data = json.load(f)
        
        structure = data.get('structure', [])
        doc_name = data.get('doc_name', 'Unknown')
        
        print(f"📄 Document: {doc_name}")
        print(f"📄 Structure File: {structure_file}")
        print()
        
        # Collect sample nodes
        sample_nodes = self.collect_sample_nodes(structure, max_nodes=2)
        print(f"🔍 Analyzing {len(sample_nodes)} sample nodes...")
        print()
        
        results = []
        
        for i, node in enumerate(sample_nodes, 1):
            print(f"🎯 NODE {i}/{len(sample_nodes)}")
            
            # Stage 1: Relevance check
            relevance_result = await self.debug_relevance_decision(node)
            results.append(relevance_result)
            
            # Stage 2: If relevant, do extraction
            if relevance_result.get('is_relevant', False):
                extraction_result = await self.debug_extraction_decision(node)
                relevance_result['extraction'] = extraction_result
            else:
                print("⏭️  Skipping extraction (not relevant)")
                print()
        
        # Summary
        print("=" * 80)
        print("📊 SUMMARY")
        print("=" * 80)
        
        relevant_count = sum(1 for r in results if r.get('is_relevant', False))
        total_extractions = sum(r.get('extraction', {}).get('count', 0) for r in results)
        
        print(f"📈 Nodes analyzed: {len(results)}")
        print(f"📈 Relevant nodes: {relevant_count}")
        print(f"📈 Total data points extracted: {total_extractions}")
        print()
        
        for i, result in enumerate(results, 1):
            node_title = result['node'].get('title', 'Unknown')[:50]
            is_relevant = "✅" if result.get('is_relevant') else "❌"
            extraction_count = result.get('extraction', {}).get('count', 0)
            print(f"  {i}. {is_relevant} {node_title} → {extraction_count} data points")
        
        return results


async def main():
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python debug_llm_decisions.py <structure_file.json>")
        print("Example: python debug_llm_decisions.py results/test-results2.json")
        return
    
    structure_file = sys.argv[1]
    
    if not Path(structure_file).exists():
        print(f"Error: File not found: {structure_file}")
        return
    
    debugger = LLMDecisionDebugger()
    await debugger.debug_full_pipeline(structure_file)


if __name__ == '__main__':
    asyncio.run(main())