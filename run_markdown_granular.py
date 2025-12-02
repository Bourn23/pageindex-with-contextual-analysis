#!/usr/bin/env python3
"""
Process markdown with full granular features using the V2 processor.

This script uses the robust markdown-it-py based processor to:
1. Parse markdown structure
2. Extract figures and tables (with local path resolution)
3. Apply semantic subdivision (medium/fine granularity)
4. Extract keywords (keywords granularity)
"""

import asyncio
import argparse
import json
import os
import sys
import logging
from pathlib import Path
import time

# Import V2 processor
from pageindex.markdown_processor_v2 import process_markdown_v2
from pageindex.llm_client import get_llm_client
from pageindex.utils import write_node_id

async def main():
    parser = argparse.ArgumentParser(
        description='Process markdown with full granular features (V2)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('md_file', help='Path to markdown file')
    parser.add_argument(
        "--granularity",
        type=str,
        default="medium",
        choices=["coarse", "medium", "fine", "keywords"],
        help="Granularity of the output structure."
    )
    parser.add_argument(
        "--keyword-level",
        type=str,
        default="fine",
        choices=["section", "medium", "fine"],
        help="Level at which to extract keywords (only used if granularity is 'keywords'). Default: fine."
    )
    parser.add_argument('--model', default='gemini-2.5-flash-lite', help='LLM model to use')
    parser.add_argument('--visualize', action='store_true', help='Generate HTML visualization')
    
    args = parser.parse_args()
    
    md_path = Path(args.md_file)
    if not md_path.exists():
        print(f"Error: File not found: {md_path}")
        return 1
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%H:%M:%S',
        force=True,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # Suppress noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("google_genai").setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    
    print("="*70)
    print("Markdown Processing V2 (markdown-it-py)")
    print("="*70)
    print(f"Input:       {md_path.name}")
    print(f"Granularity: {args.granularity}")
    print(f"Model:       {args.model}")
    print("="*70)
    print()
    
    # Initialize LLM client if needed
    llm_client = None
    if args.granularity != 'coarse':
        try:
            llm_client = get_llm_client(provider='gemini', model=args.model)
            print(f"✓ LLM Client initialized ({llm_client.provider})")
        except Exception as e:
            print(f"Warning: Could not initialize LLM client: {e}")
            print("Granular features will be skipped.")
    
    # Process
    print(f"Processing {md_path.name}...")
    # Process markdown
    try:
        result = await process_markdown_v2(
            args.md_file,
            granularity=args.granularity,
            llm_client=llm_client,
            keyword_level=args.keyword_level
        )
    except Exception as e:
        print(f"Error processing markdown: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    structure = result.get('structure', [])
    doc_name = result.get('doc_name', md_path.stem)
    
    # Save result
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / f"{doc_name}_markdown_{args.granularity}_{time.time()}.json"
    output_data = {
        'doc_name': doc_name,
        'structure': structure,
        'source': 'markdown',
        'granularity': args.granularity,
        'processor': 'v2'
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
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
    
    counts = count_by_type(structure)
    print("\nStatistics:")
    print(f"  Total nodes: {sum(counts.values())}")
    print(f"  Node types:")
    for node_type, count in sorted(counts.items()):
        print(f"    {node_type}: {count}")
        
    # Visualize if requested
    if args.visualize:
        print("\nGenerating visualization...")
        try:
            import subprocess
            subprocess.run(['python', 'visualize_structure.py', str(output_path)], check=True)
            html_path = output_path.with_suffix('.html')
            print(f"✓ Visualization: {html_path}")
            print(f"  Open: file://{html_path.absolute()}")
        except Exception as e:
            print(f"Warning: Visualization failed: {e}")
    
    return 0

if __name__ == '__main__':
    exit(asyncio.run(main()))
