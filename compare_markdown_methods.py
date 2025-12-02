#!/usr/bin/env python3
"""
Simple comparison of markdown processing methods.
"""

import asyncio
import json
from pathlib import Path


def count_all_nodes(obj):
    """Count all nodes recursively."""
    if isinstance(obj, dict):
        count = 1 if 'title' in obj else 0
        if 'nodes' in obj:
            for child in obj['nodes']:
                count += count_all_nodes(child)
        return count
    elif isinstance(obj, list):
        return sum(count_all_nodes(item) for item in obj)
    return 0


def extract_titles(obj, level=0, max_level=2):
    """Extract titles up to max_level."""
    titles = []
    if isinstance(obj, dict):
        if 'title' in obj and level <= max_level:
            titles.append(('  ' * level) + f"- {obj['title'][:70]}")
        if 'nodes' in obj and level < max_level:
            for child in obj['nodes']:
                titles.extend(extract_titles(child, level + 1, max_level))
    elif isinstance(obj, list):
        for item in obj:
            titles.extend(extract_titles(item, level, max_level))
    return titles


async def main():
    md_file = "tests/markdowns/New Insights into the Compositional Dependence of Li-Ion Transport in polymer-ceramic composite electrolytes/New Insights into the Compositional Dependence of Li-Ion Transport in polymer-ceramic composite electrolytes.md"
    
    print("="*70)
    print("METHOD 1: page_index_md (uses markdown ## headers directly)")
    print("="*70)
    
    from pageindex.page_index_md import md_to_tree
    
    result1 = await md_to_tree(
        md_file,
        if_add_node_id='yes',
        if_add_node_text='no',
        if_add_node_summary='no'
    )
    
    # Result has doc_name and structure keys
    structure1 = result1.get('structure', result1)
    count1 = count_all_nodes(structure1)
    print(f"\nTotal nodes: {count1}")
    print(f"\nStructure (first 2 levels):")
    for title in extract_titles(structure1, max_level=1):
        print(title)
    
    print("\n" + "="*70)
    print("METHOD 2: markdown_adapter (converts to text, LLM rediscovers)")
    print("="*70)
    
    from pageindex.markdown_adapter import markdown_page_index_main
    from pageindex.utils import ConfigLoader
    
    opt = ConfigLoader().load({
        'granularity': 'coarse',  # Just sections, no semantic units
        'if_add_node_text': 'no',
        'if_add_node_summary': 'no'
    })
    
    result2 = await markdown_page_index_main(md_file, None, opt)
    
    count2 = count_all_nodes(result2['structure'])
    print(f"\nTotal nodes: {count2}")
    print(f"\nStructure (first 2 levels):")
    for title in extract_titles(result2['structure'], max_level=1):
        print(title)
    
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    print(f"""
Method 1 (page_index_md):
  - Uses markdown headers (## INTRODUCTION) directly
  - Fast: No LLM calls for structure
  - Accurate: Preserves exact markdown structure
  - Total nodes: {count1}

Method 2 (markdown_adapter):
  - Ignores markdown headers
  - Slow: LLM must rediscover structure from text
  - Less accurate: LLM might miss or misinterpret sections
  - Total nodes: {count2}

RECOMMENDATION: Use page_index_md.md_to_tree() for markdown files!
It's faster and more accurate since it uses the existing structure.
""")
    
    # Save for inspection
    Path("results").mkdir(exist_ok=True)
    with open("results/method1_direct_headers.json", 'w') as f:
        json.dump(result1, f, indent=2, ensure_ascii=False)
    with open("results/method2_llm_rediscover.json", 'w') as f:
        json.dump(result2, f, indent=2, ensure_ascii=False)
    
    print("Saved to:")
    print("  - results/method1_direct_headers.json")
    print("  - results/method2_llm_rediscover.json")


if __name__ == '__main__':
    asyncio.run(main())
