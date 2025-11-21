#!/usr/bin/env python3
"""
Example: Using PageIndex with Keywords Granularity

This shows how to use the PageIndex library directly in your code.
"""

from pageindex import page_index_main
from pageindex.utils import ConfigLoader
import json
from pathlib import Path


def process_pdf_with_keywords(pdf_path, output_dir='results'):
    """
    Process a PDF with keywords granularity.
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save results
        
    Returns:
        Dictionary with structure and metadata
    """
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load configuration
    config_loader = ConfigLoader()
    opt = config_loader.load({
        'model': 'gemini-2.5-flash-lite',
        'granularity': 'keywords',  # Options: coarse, medium, fine, keywords
        'enable_figure_detection': True,
        'enable_table_detection': True,
        'enable_semantic_subdivision': True,
        'semantic_min_pages': 0.5,
        'if_add_node_id': 'yes',
        'if_add_node_summary': 'no',  # Set to 'yes' to generate summaries
        'if_add_doc_description': 'no',
        'if_add_node_text': 'yes',  # Must be 'yes' for text to be included
    })
    
    print(f"Processing: {pdf_path}")
    print(f"Granularity: {opt.granularity}")
    print(f"Features: figures={opt.enable_figure_detection}, tables={opt.enable_table_detection}")
    print()
    
    # Process the PDF
    result = page_index_main(pdf_path, opt)
    
    # Extract structure
    if isinstance(result, dict):
        doc_name = result.get('doc_name', 'document')
        structure = result.get('structure', [])
    else:
        doc_name = Path(pdf_path).stem
        structure = result
    
    # Save to JSON
    output_file = Path(output_dir) / f"{doc_name}_keywords_structure.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(structure, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved structure to: {output_file}")
    
    # Print statistics
    print_statistics(structure)
    
    return result


def print_statistics(structure):
    """Print statistics about the generated structure."""
    def count_nodes_by_type(nodes, counts=None):
        if counts is None:
            counts = {}
        
        for node in nodes:
            node_type = node.get('node_type', 'section')
            counts[node_type] = counts.get(node_type, 0) + 1
            
            if 'nodes' in node and node['nodes']:
                count_nodes_by_type(node['nodes'], counts)
        
        return counts
    
    def get_max_depth(nodes, current_depth=0):
        if not nodes:
            return current_depth
        
        max_d = current_depth
        for node in nodes:
            if 'nodes' in node and node['nodes']:
                d = get_max_depth(node['nodes'], current_depth + 1)
                max_d = max(max_d, d)
        
        return max_d
    
    counts = count_nodes_by_type(structure)
    max_depth = get_max_depth(structure)
    
    print("\nStatistics:")
    print(f"  Total nodes: {sum(counts.values())}")
    print(f"  Max depth: {max_depth}")
    print(f"  Node types:")
    for node_type, count in sorted(counts.items()):
        print(f"    {node_type}: {count}")


def extract_keywords_from_structure(structure):
    """Extract all keywords from the structure."""
    keywords = []
    
    def collect_keywords(nodes):
        for node in nodes:
            if node.get('node_type') == 'keyword':
                keywords.append({
                    'term': node.get('title'),
                    'context': node.get('metadata', {}).get('context', ''),
                    'relevance': node.get('metadata', {}).get('relevance', ''),
                    'parent': node.get('metadata', {}).get('parent_title', ''),
                })
            
            if 'nodes' in node and node['nodes']:
                collect_keywords(node['nodes'])
    
    collect_keywords(structure)
    return keywords


# Example usage
if __name__ == '__main__':
    import sys
    
    # Get PDF path from command line or use default
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = 'tests/pdfs/adefowoke-ojokoh-et-al-2009-automated-document-metadata-extraction.pdf'
    
    # Check if file exists
    if not Path(pdf_path).exists():
        print(f"Error: File not found: {pdf_path}")
        print("Usage: python example_keywords_usage.py <path_to_pdf>")
        sys.exit(1)
    
    # Process the PDF
    result = process_pdf_with_keywords(pdf_path)
    
    # Extract and display keywords
    if isinstance(result, dict):
        structure = result.get('structure', result)
    else:
        structure = result
    
    keywords = extract_keywords_from_structure(structure)
    
    print(f"\nExtracted {len(keywords)} keywords:")
    for i, kw in enumerate(keywords[:10], 1):  # Show first 10
        print(f"  {i}. {kw['term']}")
        print(f"     Context: {kw['context'][:80]}...")
        print(f"     Parent: {kw['parent']}")
    
    if len(keywords) > 10:
        print(f"  ... and {len(keywords) - 10} more")
