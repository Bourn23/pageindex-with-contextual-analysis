#!/usr/bin/env python3
"""
Test the smart summary generation logic.
"""

import json
import sys
from pathlib import Path

# Add pageindex to path
sys.path.insert(0, str(Path(__file__).parent))

from pageindex.utils import should_generate_summary


def analyze_structure(structure_file):
    """Analyze which nodes would get summaries in smart mode."""
    
    with open(structure_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    structure = data.get('structure', [])
    
    def analyze_node(node, path="", depth=0):
        """Recursively analyze nodes."""
        title = node.get('title', 'Unknown')
        current_path = f"{path}/{title}" if path else title
        
        node_type = node.get('node_type', 'section')
        has_summary = bool(node.get('summary'))
        summary_len = len(node.get('summary', ''))
        text_len = len(node.get('text', ''))
        children_count = len(node.get('nodes', []))
        
        # Check if summary would be generated
        would_generate = should_generate_summary(node)
        
        indent = "  " * depth
        status = "🔄 GENERATE" if would_generate else "✓ SKIP"
        
        print(f"{indent}{status} | {title[:50]:50} | type={node_type:15} | "
              f"summary={summary_len:4}ch | text={text_len:5}ch | children={children_count}")
        
        # Recursively analyze children
        for child in node.get('nodes', []):
            analyze_node(child, current_path, depth + 1)
        
        return would_generate
    
    print("=" * 120)
    print("Smart Summary Analysis")
    print("=" * 120)
    print(f"{'Status':<15} | {'Title':<50} | {'Type':<15} | {'Summary':<10} | {'Text':<10} | Children")
    print("-" * 120)
    
    total_nodes = 0
    nodes_to_generate = 0
    
    def count_nodes(node):
        nonlocal total_nodes, nodes_to_generate
        total_nodes += 1
        if should_generate_summary(node):
            nodes_to_generate += 1
        for child in node.get('nodes', []):
            count_nodes(child)
    
    for node in structure:
        analyze_node(node)
        count_nodes(node)
    
    print("=" * 120)
    print(f"Summary: {nodes_to_generate}/{total_nodes} nodes would generate summaries in smart mode")
    print(f"Savings: {total_nodes - nodes_to_generate} LLM calls avoided ({(1 - nodes_to_generate/total_nodes)*100:.1f}%)")
    print("=" * 120)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python test_smart_summaries.py <structure.json>")
        sys.exit(1)
    
    structure_file = sys.argv[1]
    
    if not Path(structure_file).exists():
        print(f"Error: File not found: {structure_file}")
        sys.exit(1)
    
    analyze_structure(structure_file)
