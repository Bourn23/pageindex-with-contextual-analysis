"""Test keywords granularity level."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from pageindex import page_index_main
from pageindex.utils import ConfigLoader
import json


def test_keywords_granularity(pdf_path: str):
    """Test with keywords granularity."""
    print(f"\n{'='*70}")
    print(f"Testing KEYWORDS granularity on: {Path(pdf_path).name}")
    print(f"{'='*70}")
    
    # Configure with keywords granularity
    config_loader = ConfigLoader()
    opt = config_loader.load({
        'model': 'gemini-2.5-flash-lite',
        'granularity': 'keywords',
        'enable_figure_detection': False,  # Disable for faster testing
        'enable_table_detection': False,
        'enable_semantic_subdivision': True,
        'semantic_min_pages': 0.5,
        'if_add_node_id': 'yes',
        'if_add_node_summary': 'no',
        'if_add_doc_description': 'no'
    })
    
    # Process PDF
    print("Processing PDF with keywords granularity...")
    result = page_index_main(pdf_path, opt)
    
    # Extract tree structure from result
    if isinstance(result, dict) and 'structure' in result:
        tree = result['structure']
    else:
        tree = result
    
    # Analyze results
    print_tree_summary(tree, "Keywords Granularity Results")
    
    # Save to file for inspection
    output_path = f"results/{Path(pdf_path).stem}_keywords_structure.json"
    Path("results").mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(tree, f, indent=2)
    
    print(f"\n✓ Structure saved to: {output_path}")
    
    return tree


def print_tree_summary(tree, title="Tree Summary"):
    """Print a summary of the tree structure."""
    print(f"\n{title}")
    print("=" * 70)
    
    def count_by_type(nodes, depth=0):
        """Count nodes by type."""
        counts = {}
        max_depth = depth
        
        for node in nodes:
            node_type = node.get('node_type', 'section')
            counts[node_type] = counts.get(node_type, 0) + 1
            
            if 'nodes' in node and node['nodes']:
                child_counts, child_depth = count_by_type(node['nodes'], depth + 1)
                for k, v in child_counts.items():
                    counts[k] = counts.get(k, 0) + v
                max_depth = max(max_depth, child_depth)
        
        return counts, max_depth
    
    counts, max_depth = count_by_type(tree)
    
    print(f"Total nodes: {sum(counts.values())}")
    print(f"Max depth: {max_depth}")
    print(f"\nNode types:")
    for node_type, count in sorted(counts.items()):
        print(f"  {node_type}: {count}")
    
    # Show sample keywords
    print(f"\nSample keywords (first 10):")
    keyword_count = 0
    
    def find_keywords(nodes, depth=0):
        """Find keyword nodes."""
        nonlocal keyword_count
        for node in nodes:
            if node.get('node_type') == 'keyword' and keyword_count < 10:
                keyword_count += 1
                term = node.get('title', 'Unknown')
                context = node.get('summary', '')
                print(f"  {keyword_count}. {term}")
                print(f"     Context: {context[:80]}...")
            
            if 'nodes' in node and node['nodes']:
                find_keywords(node['nodes'], depth + 1)
    
    find_keywords(tree)
    
    if keyword_count == 0:
        print("  No keywords found in tree")
    
    print("=" * 70)


if __name__ == '__main__':
    # Test with a sample PDF
    import sys
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        # Default test PDF
        pdf_path = 'tests/pdfs/four-lectures.pdf'
    
    if not Path(pdf_path).exists():
        print(f"Error: PDF not found at {pdf_path}")
        print("Usage: python test_keywords.py <path_to_pdf>")
        sys.exit(1)
    
    test_keywords_granularity(pdf_path)
