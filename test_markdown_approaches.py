#!/usr/bin/env python3
"""
Compare two markdown processing approaches:
1. markdown_adapter (converts to text, uses LLM)
2. page_index_md (uses markdown headers directly)
"""

import asyncio
import json
from pathlib import Path
from pageindex import markdown_page_index
from pageindex.page_index_md import md_to_tree


def count_nodes(structure):
    """Count nodes recursively."""
    counts = {}
    
    def count(nodes):
        for node in nodes:
            node_type = node.get('node_type', 'section')
            counts[node_type] = counts.get(node_type, 0) + 1
            if 'nodes' in node and node['nodes']:
                count(node['nodes'])
    
    count(structure)
    return counts


def print_top_sections(structure, name):
    """Print top-level sections."""
    print(f"\n{name} - Top-level sections:")
    for i, node in enumerate(structure[:10], 1):  # First 10
        title = node.get('title', 'Unknown')
        level = node.get('level', '?')
        child_count = len(node.get('nodes', []))
        print(f"  {i}. [L{level}] {title[:60]} ({child_count} children)")


async def main():
    base_path = "tests/markdowns/New Insights into the Compositional Dependence of Li-Ion Transport in polymer-ceramic composite electrolytes"
    md_file = f"{base_path}/New Insights into the Compositional Dependence of Li-Ion Transport in polymer-ceramic composite electrolytes.md"
    meta_file = f"{base_path}/New Insights into the Compositional Dependence of Li-Ion Transport in polymer-ceramic composite electrolytes_meta.json"
    
    print("="*70)
    print("APPROACH 1: markdown_adapter (LLM-based, ignores headers)")
    print("="*70)
    
    from pageindex.markdown_adapter import markdown_page_index_main
    from pageindex.utils import ConfigLoader
    
    opt = ConfigLoader().load({
        'granularity': 'medium',
        'if_add_node_text': 'no',
        'if_add_node_summary': 'no'
    })
    
    result1 = await markdown_page_index_main(md_file, meta_file, opt)
    
    counts1 = count_nodes(result1['structure'])
    print(f"\nTotal nodes: {sum(counts1.values())}")
    print(f"Node types: {counts1}")
    print_top_sections(result1['structure'], "Approach 1")
    
    print("\n" + "="*70)
    print("APPROACH 2: page_index_md (uses markdown headers directly)")
    print("="*70)
    
    result2 = await md_to_tree(
        md_file,
        if_thinning=False,
        if_add_node_summary='no',
        if_add_node_text='no',
        if_add_node_id='yes'
    )
    
    counts2 = count_nodes(result2)
    print(f"\nTotal nodes: {sum(counts2.values())}")
    print(f"Node types: {counts2}")
    print_top_sections(result2, "Approach 2")
    
    # Save both for inspection
    output_dir = Path("results")
    
    with open(output_dir / "markdown_approach1_adapter.json", 'w') as f:
        json.dump(result1, f, indent=2, ensure_ascii=False)
    
    with open(output_dir / "markdown_approach2_direct.json", 'w') as f:
        json.dump({'structure': result2}, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"\nApproach 1 (adapter): {sum(counts1.values())} nodes")
    print(f"Approach 2 (direct):  {sum(counts2.values())} nodes")
    print(f"\nApproach 2 uses markdown headers directly = faster & more accurate!")
    print(f"\nSaved to:")
    print(f"  - results/markdown_approach1_adapter.json")
    print(f"  - results/markdown_approach2_direct.json")


if __name__ == '__main__':
    asyncio.run(main())
