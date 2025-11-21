"""
Compare different granularity levels to see the difference.
"""

import json
import sys
from pathlib import Path

def count_nodes(tree, depth=0):
    """Count nodes at each depth level."""
    counts = {}
    
    def count_recursive(nodes, current_depth):
        if current_depth not in counts:
            counts[current_depth] = 0
        counts[current_depth] += len(nodes)
        
        for node in nodes:
            if 'nodes' in node and node['nodes']:
                count_recursive(node['nodes'], current_depth + 1)
    
    count_recursive(tree, 0)
    return counts

def analyze_tree(filepath):
    """Analyze a PageIndex tree structure."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Handle both 'nodes' and 'structure' keys
    tree = data.get('nodes', data.get('structure', []))
    
    # Count total nodes
    def count_total(nodes):
        count = len(nodes)
        for node in nodes:
            if 'nodes' in node and node['nodes']:
                count += count_total(node['nodes'])
        return count
    
    total = count_total(tree)
    
    # Count by depth
    depth_counts = count_nodes(tree)
    
    # Find max depth
    max_depth = max(depth_counts.keys()) if depth_counts else 0
    
    # Count semantic units
    def count_semantic(nodes):
        count = 0
        for node in nodes:
            if node.get('node_type') == 'semantic_unit':
                count += 1
            if 'nodes' in node and node['nodes']:
                count += count_semantic(node['nodes'])
        return count
    
    semantic_count = count_semantic(tree)
    
    # Count figures and tables
    def count_by_type(nodes, node_type):
        count = 0
        for node in nodes:
            if node.get('node_type') == node_type:
                count += 1
            if 'nodes' in node and node['nodes']:
                count += count_by_type(node['nodes'], node_type)
        return count
    
    figure_count = count_by_type(tree, 'figure')
    table_count = count_by_type(tree, 'table')
    
    return {
        'total_nodes': total,
        'max_depth': max_depth,
        'depth_counts': depth_counts,
        'semantic_units': semantic_count,
        'figures': figure_count,
        'tables': table_count
    }

def main():
    """Compare granularity levels."""
    
    # Handle both running from PageIndex/ and from root
    if Path('results').exists():
        results_dir = Path('results')
    elif Path('PageIndex/results').exists():
        results_dir = Path('PageIndex/results')
    else:
        print("Error: Could not find results directory")
        print("Run this script from the PageIndex directory or the parent directory")
        sys.exit(1)
    
    # Look for structure files
    files = {
        'coarse': results_dir / 'earthmover_structure_coarse.json',
        'medium': results_dir / 'earthmover_structure_medium.json',
        'fine': results_dir / 'earthmover_structure_fine.json',
    }
    
    # Also check default name
    default_file = results_dir / 'earthmover_structure.json'
    if default_file.exists():
        print(f"Found default structure file: {default_file}")
        print("Analyzing...\n")
        stats = analyze_tree(default_file)
        print(f"Total nodes: {stats['total_nodes']}")
        print(f"Max depth: {stats['max_depth']}")
        print(f"Semantic units: {stats['semantic_units']}")
        print(f"Figures: {stats['figures']}")
        print(f"Tables: {stats['tables']}")
        print(f"\nNodes by depth:")
        for depth, count in sorted(stats['depth_counts'].items()):
            print(f"  Level {depth}: {count} nodes")
        print("\n" + "="*60 + "\n")
    
    print("Comparing Granularity Levels")
    print("="*60)
    
    for level, filepath in files.items():
        if filepath.exists():
            print(f"\n{level.upper()} Granularity:")
            print("-"*60)
            stats = analyze_tree(filepath)
            print(f"Total nodes: {stats['total_nodes']}")
            print(f"Max depth: {stats['max_depth']}")
            print(f"Semantic units: {stats['semantic_units']}")
            print(f"Figures: {stats['figures']}")
            print(f"Tables: {stats['tables']}")
            print(f"\nNodes by depth:")
            for depth, count in sorted(stats['depth_counts'].items()):
                print(f"  Level {depth}: {count} nodes")
        else:
            print(f"\n{level.upper()}: File not found ({filepath})")
    
    print("\n" + "="*60)
    print("\nTo generate comparison files, run:")
    print("  python run_pageindex.py --pdf_path tests/pdfs/earthmover.pdf --granularity coarse")
    print("  mv results/earthmover_structure.json results/earthmover_structure_coarse.json")
    print("")
    print("  python run_pageindex.py --pdf_path tests/pdfs/earthmover.pdf --granularity medium")
    print("  mv results/earthmover_structure.json results/earthmover_structure_medium.json")
    print("")
    print("  python run_pageindex.py --pdf_path tests/pdfs/earthmover.pdf --granularity fine")
    print("  mv results/earthmover_structure.json results/earthmover_structure_fine.json")

if __name__ == "__main__":
    main()
