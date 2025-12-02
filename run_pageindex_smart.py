#!/usr/bin/env python3
"""
Smart PageIndex CLI that auto-detects file type (PDF or Markdown).

Usage:
    python run_pageindex_smart.py <file_path> [options]

Examples:
    # Process PDF
    python run_pageindex_smart.py paper.pdf --granularity keywords
    
    # Process Markdown (auto-detects and uses md_to_tree)
    python run_pageindex_smart.py paper.md --granularity keywords
    
    # With metadata
    python run_pageindex_smart.py paper.md --metadata paper_meta.json
"""

import argparse
import asyncio
import json
from pathlib import Path
from pageindex import page_index_main
from pageindex.page_index_md import md_to_tree
from pageindex.utils import ConfigLoader


async def process_markdown(md_path, metadata_path=None, opt=None):
    """Process markdown file using md_to_tree (uses headers directly)."""
    print(f"Processing markdown file (using headers directly)...")
    
    # Step 1: Extract structure from markdown headers
    result = await md_to_tree(
        str(md_path),
        if_thinning=False,
        if_add_node_summary='no',  # We'll do this later if needed
        if_add_node_text='yes',  # Need text for granular processing
        if_add_node_id='yes',
        model=opt.model
    )
    
    # Extract structure
    if isinstance(result, dict):
        structure = result.get('structure', [])
        doc_name = result.get('doc_name', Path(md_path).stem)
    else:
        structure = result
        doc_name = Path(md_path).stem
    
    # Step 2: Apply granular features if requested
    granularity = getattr(opt, 'granularity', 'coarse')
    
    if granularity in ['medium', 'fine', 'keywords']:
        print(f"Applying granular features (granularity: {granularity})...")
        
        # Convert markdown to page_list format for granular processing
        from pageindex.markdown_adapter import markdown_to_page_list
        page_list = markdown_to_page_list(str(md_path), metadata_path, opt.model)
        
        # Apply semantic subdivision
        if opt.enable_semantic_subdivision:
            from pageindex.granular.integration import apply_semantic_subdivision
            from pageindex.utils import JsonLogger
            
            logger = JsonLogger(str(md_path))
            await apply_semantic_subdivision(structure, page_list, opt, logger)
            print(f"✓ Semantic subdivision complete")
        
        # Apply figure/table detection if enabled
        # Note: This would need markdown-specific implementation
        # For now, we skip it for markdown files
    
    # Step 3: Generate summaries if requested
    if opt.if_add_node_summary == 'yes':
        print("Generating summaries...")
        from pageindex.utils import generate_summaries_for_structure
        await generate_summaries_for_structure(structure, model=opt.model)
    
    return {
        'doc_name': doc_name,
        'structure': structure
    }


def process_pdf(pdf_path, opt):
    """Process PDF file using standard pipeline."""
    print(f"Processing PDF file...")
    return page_index_main(str(pdf_path), opt)


def main():
    parser = argparse.ArgumentParser(
        description='Process PDF or Markdown documents with PageIndex',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
File Types:
  .pdf  - Uses standard PDF processing pipeline
  .md   - Uses markdown header extraction (faster, more accurate)

Granularity Levels:
  coarse    - Sections only (fastest)
  medium    - Sections + semantic units
  fine      - Sections + semantic units + fine semantic units
  keywords  - All of the above + keyword extraction (slowest, most detailed)

Examples:
  python run_pageindex_smart.py paper.pdf --granularity keywords
  python run_pageindex_smart.py paper.md --granularity keywords
  python run_pageindex_smart.py paper.md --metadata paper_meta.json
        """
    )
    
    # Required arguments
    parser.add_argument('file_path', help='Path to PDF or Markdown file')
    
    # Markdown-specific options
    parser.add_argument('--metadata', help='Path to metadata JSON (for markdown)')
    
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
    
    # Detect file type
    file_ext = file_path.suffix.lower()
    is_markdown = file_ext in ['.md', '.markdown']
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        suffix = 'markdown' if is_markdown else args.granularity
        output_path = Path('results') / f"{file_path.stem}_{suffix}_structure.json"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure PageIndex
    config_loader = ConfigLoader()
    opt = config_loader.load({
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
    
    # Print configuration
    print("=" * 70)
    print(f"PageIndex Processing")
    print("=" * 70)
    print(f"Input:       {file_path}")
    print(f"Type:        {'Markdown' if is_markdown else 'PDF'}")
    print(f"Output:      {output_path}")
    if is_markdown:
        print(f"Method:      Direct header extraction (fast)")
        if args.metadata:
            print(f"Metadata:    {args.metadata}")
    else:
        print(f"Granularity: {args.granularity}")
    print(f"Model:       {args.model}")
    print(f"Features:    figures={args.figures}, tables={args.tables}, summaries={args.summaries}")
    print("=" * 70)
    print()
    
    # Process file
    try:
        if is_markdown:
            # Use md_to_tree for markdown (faster, uses headers directly)
            result = asyncio.run(process_markdown(file_path, args.metadata, opt))
        else:
            # Use standard PDF pipeline
            result = process_pdf(file_path, opt)
        
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
            'structure': structure,
            'source': 'markdown' if is_markdown else 'pdf'
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
