#!/usr/bin/env python3
"""
Simple approach: Use markdown_adapter but with better page splitting.

The key insight: markdown_adapter already applies granular features,
we just need to give it better page boundaries.
"""

import argparse
import json
from pathlib import Path
from pageindex.markdown_adapter import markdown_page_index


def main():
    parser = argparse.ArgumentParser(description='Process markdown with granular features')
    parser.add_argument('md_file', help='Path to markdown file')
    parser.add_argument('--metadata', help='Path to metadata JSON')
    parser.add_argument('--granularity', '-g', 
                       choices=['coarse', 'medium', 'fine', 'keywords'],
                       default='keywords')
    parser.add_argument('--visualize', action='store_true')
    
    args = parser.parse_args()
    
    md_path = Path(args.md_file)
    if not md_path.exists():
        print(f"Error: File not found: {md_path}")
        return 1
    
    print("="*70)
    print("Processing Markdown with Granular Features")
    print("="*70)
    print(f"Input:       {md_path.name}")
    print(f"Granularity: {args.granularity}")
    print("="*70)
    print()
    
    # Use markdown_page_index (it already applies granular features!)
    result = markdown_page_index(
        str(md_path),
        metadata_path=args.metadata,
        granularity=args.granularity,
        if_add_node_text='yes',
        if_add_node_summary='no'
    )
    
    # Save result
    output_path = Path('results') / f"{md_path.stem}_markdown_{args.granularity}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved to: {output_path}")
    
    # Print statistics
    def count_by_type(nodes, counts=None):
        if counts is None:
            counts = {}
        for n in nodes:
            node_type = n.get('node_type', 'section')
            counts[node_type] = counts.get(node_type, 0) + 1
            if 'nodes' in n:
                count_by_type(n['nodes'], counts)
        return counts
    
    counts = count_by_type(result['structure'])
    print("\nStatistics:")
    print(f"  Total nodes: {sum(counts.values())}")
    print(f"  Node types:")
    for node_type, count in sorted(counts.items()):
        print(f"    {node_type}: {count}")
    
    # Visualize if requested
    if args.visualize:
        import subprocess
        subprocess.run(['python', 'visualize_structure.py', str(output_path)])
        html_path = output_path.with_suffix('.html')
        print(f"\n✓ Visualization: {html_path}")
    
    return 0


if __name__ == '__main__':
    exit(main())
