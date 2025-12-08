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
import time as t

# Import V2 processor
from pageindex.markdown_processor_v2 import process_markdown_v2
from pageindex.llm_client import get_llm_client
from pageindex.utils import write_node_id, ConfigLoader

async def main():
    parser = argparse.ArgumentParser(
        description='Process markdown with full granular features (V2)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Granularity Levels:
  coarse    - Sections only (fastest)
  medium    - Sections + semantic units
  fine      - Sections + semantic units + fine semantic units
  keywords  - All of the above + keyword extraction (slowest, most detailed)

Examples:
  python run_markdown_granular.py document.md --granularity keywords
  python run_markdown_granular.py document.md --granularity fine --no-figures --no-tables
  python run_markdown_granular.py document.md --granularity medium --summaries
        """
    )
    
    # Required arguments
    parser.add_argument('md_file', help='Path to markdown file')
    
    # Granularity options
    parser.add_argument(
        '--granularity', '-g',
        choices=['coarse', 'medium', 'fine', 'keywords'],
        default='keywords',
        help='Granularity level (default: keywords)'
    )
    parser.add_argument(
        "--keyword-level",
        type=str,
        default="fine",
        choices=["section", "medium", "fine"],
        help="Level at which to extract keywords: 'section' (from sections), 'medium' (from semantic units), 'fine' (from sentence-level units). Default: fine."
    )
    
    # Feature flags
    parser.add_argument('--figures', action='store_true', default=True, help='Enable figure detection (default: on)')
    parser.add_argument('--no-figures', action='store_false', dest='figures', help='Disable figure detection')
    
    parser.add_argument('--tables', action='store_true', default=True, help='Enable table detection (default: on)')
    parser.add_argument('--no-tables', action='store_false', dest='tables', help='Disable table detection')
    
    parser.add_argument('--summaries', action='store_true', help='Generate summaries for nodes')
    parser.add_argument('--doc-description', action='store_true', help='Generate document description')
    
    # Model options
    parser.add_argument('--model', default='gemini-2.5-flash-lite', help='LLM model to use')
    
    # Output options
    parser.add_argument('--output', '-o', help='Output JSON file path')
    parser.add_argument('--visualize', action='store_true', help='Generate HTML visualization')
    
    # Advanced options
    parser.add_argument('--semantic-min-pages', type=float, default=0.5, 
                       help='Minimum pages for semantic subdivision (default: 0.5)')
    parser.add_argument('--validate-coverage', action='store_true',
                       help='Validate that all source text is covered in the output tree')
    parser.add_argument('--no-gap-fill', action='store_true',
                       help='Disable automatic gap filling for uncovered paragraphs')
    
    args = parser.parse_args()
    
    # Validate input
    md_path = Path(args.md_file)
    if not md_path.exists():
        print(f"Error: File not found: {md_path}")
        return 1
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path('results') / f"{md_path.stem}_{args.granularity}_structure.json"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Setup logging FIRST (before any other operations)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Suppress noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("google.genai").setLevel(logging.WARNING)
    logging.getLogger("markdown_it").setLevel(logging.WARNING)  # Critical: suppress markdown-it verbose logging
    
    # Print configuration
    print("=" * 70)
    print(f"Markdown Processing (markdown-it-py)")
    print("=" * 70)
    print(f"Input:       {md_path}")
    print(f"Output:      {output_path}")
    print(f"Granularity: {args.granularity}")
    print(f"Model:       {args.model}")
    print(f"Features:    figures={args.figures}, tables={args.tables}, summaries={args.summaries}")
    print("=" * 70)
    print()
    
    # Initialize LLM client if needed
    llm_client = None
    if args.granularity != 'coarse':
        try:
            llm_client = get_llm_client(model=args.model)
            print(f"✓ LLM Client initialized ({llm_client.provider})")
        except Exception as e:
            print(f"Warning: Could not initialize LLM client: {e}")
            print("Granular features will be skipped.")
    
    # Process markdown
    try:
        print(f"Processing {md_path.name}...")
        print(f"  - Reading file...")
        result = await process_markdown_v2(
            str(args.md_file),
            granularity=args.granularity,
            llm_client=llm_client,
            keyword_level=args.keyword_level,
        )
        print(f"  - Processing complete")
    except Exception as e:
        print(f"Error processing markdown: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Extract structure
    if isinstance(result, dict):
        structure = result.get('structure', [])
        doc_name = result.get('doc_name', md_path.stem)
        doc_description = result.get('doc_description')
    else:
        structure = result
        doc_name = md_path.stem
        doc_description = None
    
    # Save to JSON
    output_data = {
        'doc_name': doc_name,
        'structure': structure
    }
    if doc_description:
        output_data['doc_description'] = doc_description
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Structure saved to: {output_path}")
    
    # Print statistics
    print_statistics(structure)
    
    # Validate coverage if requested
    if args.validate_coverage:
        print("\nValidating coverage...")
        from pageindex.coverage_validator import validate_tree_coverage, validate_full_tree_coverage
        
        # Read original source text
        with open(md_path, 'r', encoding='utf-8') as f:
            source_text = f.read()
        
        # Validate overall coverage
        report = validate_tree_coverage(structure, source_text)
        print(str(report))
        
        # Validate per-node coverage
        node_reports = validate_full_tree_coverage(structure)
        if node_reports:
            print(f"\n⚠ {len(node_reports)} nodes have incomplete coverage")
            for path, node_report in list(node_reports.items())[:5]:
                print(f"  - {path}: {node_report.coverage_percentage:.1f}%")
        else:
            print("✓ All nodes have complete coverage")
    
    # Generate visualization if requested
    if args.visualize:
        html_path = output_path.with_suffix('.html')
        import subprocess
        subprocess.run(['python', 'visualize_structure.py', str(output_path)])
        print(f"✓ Visualization saved to: {html_path}")
        print(f"  Open in browser: file://{html_path.absolute()}")
    
    return 0


def print_statistics(structure):
    """Print statistics about the structure."""
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
    print()
    
    return 0

if __name__ == '__main__':
    import sys
    import os
    
    try:
        # Run the async main function
        exit_code = asyncio.run(main())
        
        # Flush output before exit
        sys.stdout.flush()
        sys.stderr.flush()
        
        # Force immediate exit without waiting for thread cleanup
        # This is necessary because asyncio.to_thread() doesn't clean up properly
        os._exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.stdout.flush()
        os._exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        sys.stdout.flush()
        os._exit(1)
