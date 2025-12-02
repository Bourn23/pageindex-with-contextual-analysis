"""
Compare PDF vs Markdown processing for the same document.
"""

import json
from pageindex import markdown_page_index
from pathlib import Path


def count_nodes_by_type(structure):
    """Count nodes by type recursively."""
    counts = {}
    
    def count(nodes):
        for node in nodes:
            node_type = node.get('node_type', 'section')
            counts[node_type] = counts.get(node_type, 0) + 1
            if 'nodes' in node and node['nodes']:
                count(node['nodes'])
    
    count(structure)
    return counts


def get_max_depth(structure):
    """Get maximum tree depth."""
    def depth(nodes, current=0):
        if not nodes:
            return current
        max_d = current
        for node in nodes:
            if 'nodes' in node and node['nodes']:
                d = depth(node['nodes'], current + 1)
                max_d = max(max_d, d)
        return max_d
    
    return depth(structure)


def print_structure_summary(structure, source_name):
    """Print summary of structure."""
    counts = count_nodes_by_type(structure)
    max_depth = get_max_depth(structure)
    
    print(f"\n{'='*70}")
    print(f"{source_name} Processing Results")
    print(f"{'='*70}")
    print(f"Total nodes: {sum(counts.values())}")
    print(f"Max depth: {max_depth}")
    print(f"\nNode types:")
    for node_type, count in sorted(counts.items()):
        print(f"  {node_type:20s}: {count:4d}")


def print_top_level_sections(structure, source_name):
    """Print top-level section titles."""
    print(f"\n{source_name} - Top-level sections:")
    for i, node in enumerate(structure, 1):
        title = node.get('title', 'Unknown')
        node_type = node.get('node_type', 'section')
        child_count = len(node.get('nodes', []))
        print(f"  {i}. [{node_type}] {title} ({child_count} children)")


def main():
    base_path = "tests/markdowns/New Insights into the Compositional Dependence of Li-Ion Transport in polymer-ceramic composite electrolytes"
    
    # Process markdown
    print("Processing markdown file...")
    md_result = markdown_page_index(
        f'{base_path}/New Insights into the Compositional Dependence of Li-Ion Transport in polymer-ceramic composite electrolytes.md',
        metadata_path=f'{base_path}/New Insights into the Compositional Dependence of Li-Ion Transport in polymer-ceramic composite electrolytes_meta.json',
        granularity='keywords',
        if_add_node_text='yes',
        if_add_node_summary='no'
    )
    
    # Load PDF result
    print("Loading PDF processing result...")
    pdf_result_path = "results/New Insights into the Compositional Dependence of Li-Ion Transport in polymer-ceramic composite electrolytes_keywords_structure.json"
    
    if Path(pdf_result_path).exists():
        with open(pdf_result_path, 'r') as f:
            pdf_result = json.load(f)
        
        # Compare
        print_structure_summary(md_result['structure'], "MARKDOWN")
        print_structure_summary(pdf_result['structure'], "PDF")
        
        print_top_level_sections(md_result['structure'], "MARKDOWN")
        print_top_level_sections(pdf_result['structure'], "PDF")
        
        # Analysis
        md_counts = count_nodes_by_type(md_result['structure'])
        pdf_counts = count_nodes_by_type(pdf_result['structure'])
        
        print(f"\n{'='*70}")
        print("Comparison Analysis")
        print(f"{'='*70}")
        
        all_types = set(md_counts.keys()) | set(pdf_counts.keys())
        for node_type in sorted(all_types):
            md_count = md_counts.get(node_type, 0)
            pdf_count = pdf_counts.get(node_type, 0)
            diff = md_count - pdf_count
            diff_str = f"{diff:+d}" if diff != 0 else "0"
            print(f"  {node_type:20s}: MD={md_count:4d}  PDF={pdf_count:4d}  Diff={diff_str}")
        
        print(f"\nTotal: MD={sum(md_counts.values())}  PDF={sum(pdf_counts.values())}")
        
    else:
        print(f"\nPDF result not found at: {pdf_result_path}")
        print("Run the PDF processing first:")
        print('  python run_pageindex.py "tests/pdfs/..." --granularity keywords')
        
        print_structure_summary(md_result['structure'], "MARKDOWN")
        print_top_level_sections(md_result['structure'], "MARKDOWN")


if __name__ == '__main__':
    main()
