#!/usr/bin/env python3
"""
Process markdown file and save the PageIndex structure.
"""

import json
import sys
from pathlib import Path
from pageindex import markdown_page_index


def main():
    # Your markdown file
    base_path = "tests/markdowns/New Insights into the Compositional Dependence of Li-Ion Transport in polymer-ceramic composite electrolytes"
    md_file = f"{base_path}/New Insights into the Compositional Dependence of Li-Ion Transport in polymer-ceramic composite electrolytes.md"
    meta_file = f"{base_path}/New Insights into the Compositional Dependence of Li-Ion Transport in polymer-ceramic composite electrolytes_meta.json"
    
    print(f"Processing markdown file...")
    print(f"  Input: {md_file}")
    
    # Process with keywords granularity
    result = markdown_page_index(
        md_file,
        metadata_path=meta_file,
        granularity='keywords',
        if_add_node_text='yes',
        if_add_node_summary='no'
    )
    
    # Save to results directory
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    doc_name = result['doc_name']
    output_file = output_dir / f"{doc_name}_markdown_keywords_structure.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Structure saved to: {output_file}")
    
    # Print statistics
    def count_nodes(nodes, counts=None):
        if counts is None:
            counts = {}
        for node in nodes:
            node_type = node.get('node_type', 'section')
            counts[node_type] = counts.get(node_type, 0) + 1
            if 'nodes' in node and node['nodes']:
                count_nodes(node['nodes'], counts)
        return counts
    
    counts = count_nodes(result['structure'])
    
    print(f"\nStatistics:")
    print(f"  Total nodes: {sum(counts.values())}")
    print(f"  Node types:")
    for node_type, count in sorted(counts.items()):
        print(f"    {node_type}: {count}")
    
    # Also create visualization
    try:
        from visualize_structure import create_html_visualization
        html_file = output_dir / f"{doc_name}_markdown_keywords_structure.html"
        create_html_visualization(result['structure'], str(html_file), doc_name)
        print(f"\n✓ Visualization saved to: {html_file}")
        print(f"  Open in browser: file://{html_file.absolute()}")
    except Exception as e:
        print(f"\nNote: Could not create visualization: {e}")
    
    return str(output_file)


if __name__ == '__main__':
    output_file = main()
    print(f"\nYou can now inspect: {output_file}")
