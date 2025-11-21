"""
Test semantic subdivision to see if it's working.
"""

import json
import sys

def check_semantic_units(filepath):
    """Check if a structure file has semantic units."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    tree = data.get('nodes', data.get('structure', []))
    
    def find_semantic_units(nodes, path=""):
        """Recursively find semantic units."""
        units = []
        for i, node in enumerate(nodes):
            node_path = f"{path}/{node.get('title', 'Unknown')}"
            node_type = node.get('node_type', 'section')
            
            if node_type == 'semantic_unit':
                units.append({
                    'path': node_path,
                    'title': node.get('title'),
                    'pages': f"{node.get('start_index')}-{node.get('end_index')}",
                    'semantic_type': node.get('metadata', {}).get('semantic_type', 'unknown')
                })
            
            if 'nodes' in node and node['nodes']:
                units.extend(find_semantic_units(node['nodes'], node_path))
        
        return units
    
    units = find_semantic_units(tree)
    
    print(f"\nAnalyzing: {filepath}")
    print("="*60)
    
    if units:
        print(f"Found {len(units)} semantic units:\n")
        for unit in units:
            print(f"  {unit['path']}")
            print(f"    Type: {unit['semantic_type']}")
            print(f"    Pages: {unit['pages']}")
            print()
    else:
        print("No semantic units found!")
        print("\nPossible reasons:")
        print("  1. Sections too short (< min_pages threshold)")
        print("  2. Semantic subdivision not enabled")
        print("  3. LLM didn't identify semantic boundaries")
        print("  4. Text content missing from nodes")
        
        # Check if nodes have text
        def check_text(nodes):
            has_text = []
            for node in nodes:
                if 'text' in node and node['text']:
                    has_text.append(node.get('title', 'Unknown'))
                if 'nodes' in node and node['nodes']:
                    has_text.extend(check_text(node['nodes']))
            return has_text
        
        nodes_with_text = check_text(tree)
        print(f"\n  Nodes with text content: {len(nodes_with_text)}")
        if nodes_with_text:
            print(f"    {', '.join(nodes_with_text[:5])}")

if __name__ == "__main__":
    files = [
        'PageIndex/results/earthmover_structure_medium.json',
        'PageIndex/results/earthmover_structure_fine.json',
    ]
    
    for filepath in files:
        try:
            check_semantic_units(filepath)
        except FileNotFoundError:
            print(f"\nFile not found: {filepath}")
        except Exception as e:
            print(f"\nError analyzing {filepath}: {e}")
