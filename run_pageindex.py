#!/usr/bin/env python3
"""
Command-line interface for PageIndex with all granularity levels.
Supports both PDF and Markdown files.

Usage:
    python run_pageindex.py <file_path> [options]

Examples:
    # PDF processing
    python run_pageindex.py paper.pdf --granularity keywords
    
    # Markdown processing (uses header structure)
    python run_pageindex.py document.md --granularity keywords
    
    # With all features
    python run_pageindex.py paper.pdf --granularity keywords --figures --tables --summaries
    
    # Fine granularity without keywords
    python run_pageindex.py paper.pdf --granularity fine
    
    # Coarse (fastest)
    python run_pageindex.py paper.pdf --granularity coarse
"""

import argparse
import asyncio
import json
from pathlib import Path
from pageindex.utils import ConfigLoader


def main():
    parser = argparse.ArgumentParser(
        description='Process PDF or Markdown documents with PageIndex',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Granularity Levels:
  coarse    - Sections only (fastest)
  medium    - Sections + semantic units
  fine      - Sections + semantic units + fine semantic units
  keywords  - All of the above + keyword extraction (slowest, most detailed)

Examples:
  python run_pageindex.py paper.pdf --granularity keywords
  python run_pageindex.py document.md --granularity fine
  python run_pageindex.py paper.pdf --granularity fine --no-figures --no-tables
  python run_pageindex.py paper.pdf --granularity medium --summaries
        """
    )
    
    # Required arguments
    parser.add_argument('file_path', help='Path to PDF or Markdown file')
    
    # Granularity options
    parser.add_argument(
        '--granularity', '-g',
        choices=['coarse', 'medium', 'fine', 'keywords'],
        default='keywords',
        help='Granularity level (default: keywords)'
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
    
    args = parser.parse_args()
    
    # Validate input
    file_path = Path(args.file_path)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return 1
    
    # Determine file type
    suffix = file_path.suffix.lower()
    if suffix == '.pdf':
        file_type = 'pdf'
    elif suffix in ['.md', '.markdown']:
        file_type = 'markdown'
    else:
        print(f"Error: Unsupported file type: {suffix}")
        print("Supported types: .pdf, .md, .markdown")
        return 1
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path('results') / f"{file_path.stem}_{args.granularity}_structure.json"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Print configuration
    print("=" * 70)
    print("PageIndex Processing")
    print("=" * 70)
    print(f"Input:       {file_path}")
    print(f"File type:   {file_type}")
    print(f"Output:      {output_path}")
    print(f"Granularity: {args.granularity}")
    print(f"Model:       {args.model}")
    print(f"Features:    figures={args.figures}, tables={args.tables}, summaries={args.summaries}")
    print("=" * 70)
    print()
    
    # Process based on file type
    try:
        if file_type == 'pdf':
            result = process_pdf(file_path, args)
        else:
            result = asyncio.run(process_markdown(file_path, args))
        
        # Extract structure
        if isinstance(result, dict):
            structure = result.get('structure', [])
            doc_name = result.get('doc_name', file_path.stem)
            doc_description = result.get('doc_description')
        else:
            structure = result
            doc_name = file_path.stem
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
        
        # Generate visualization if requested
        if args.visualize:
            html_path = output_path.with_suffix('.html')
            import subprocess
            subprocess.run(['python', 'visualize_structure.py', str(output_path)])
            print(f"✓ Visualization saved to: {html_path}")
            print(f"  Open in browser: file://{html_path.absolute()}")
        
        return 0
        
    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()
        return 1


def process_pdf(file_path, args):
    """Process a PDF file using the existing page_index_main."""
    from pageindex import page_index_main
    
    opt = ConfigLoader().load({
        'model': args.model,
        'granularity': args.granularity,
        'enable_figure_detection': args.figures,
        'enable_table_detection': args.tables,
        'enable_semantic_subdivision': args.granularity in ['medium', 'fine', 'keywords'],
        'semantic_min_pages': args.semantic_min_pages,
        'if_add_node_id': 'yes',
        'if_add_node_summary': 'yes' if args.summaries else 'no',
        'if_add_doc_description': 'yes' if args.doc_description else 'no',
        'if_add_node_text': 'yes',
    })
    
    return page_index_main(str(file_path), opt)


async def process_markdown(file_path, args):
    """
    Process a Markdown file using the same pipeline as PDFs.
    
    This mirrors the PDF processing flow:
    1. Parse markdown to extract structure from headers
    2. Build tree structure (like TOC extraction for PDFs)
    3. Apply granular features (semantic subdivision, keywords)
    """
    from pageindex.page_index_markdown import markdown_index_main
    
    opt = ConfigLoader().load({
        'model': args.model,
        'granularity': args.granularity,
        'enable_figure_detection': False,  # Not applicable for markdown
        'enable_table_detection': False,   # Not applicable for markdown
        'enable_semantic_subdivision': args.granularity in ['medium', 'fine', 'keywords'],
        'semantic_min_pages': 0,  # No page minimum for markdown
        'if_add_node_id': 'yes',
        'if_add_node_summary': 'yes' if args.summaries else 'no',
        'if_add_doc_description': 'yes' if args.doc_description else 'no',
        'if_add_node_text': 'yes',
    })
    
    return await markdown_index_main(str(file_path), opt)


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


if __name__ == '__main__':
    exit(main())
