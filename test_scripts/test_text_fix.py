#!/usr/bin/env python3
"""
Test script to verify that semantic unit text extraction is working correctly.
"""

import json
from pathlib import Path

def check_text_uniqueness(structure_file):
    """Check if child nodes have unique text content."""
    
    with open(structure_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    structure = data.get('structure', [])
    
    def check_node(node, path=""):
        """Recursively check nodes for duplicate text."""
        title = node.get('title', 'Unknown')
        current_path = f"{path}/{title}" if path else title
        
        children = node.get('nodes', [])
        if not children:
            return
        
        # Get text from children
        child_texts = []
        for child in children:
            text = child.get('text', '')
            if text:
                child_texts.append({
                    'title': child.get('title', 'Unknown'),
                    'text': text,
                    'length': len(text),
                    'node_type': child.get('node_type', 'unknown')
                })
        
        if len(child_texts) > 1:
            # Check if all texts are identical
            first_text = child_texts[0]['text']
            all_same = all(c['text'] == first_text for c in child_texts)
            
            if all_same:
                print(f"\n⚠️  DUPLICATE TEXT FOUND in: {current_path}")
                print(f"   All {len(child_texts)} children have identical text ({len(first_text)} chars)")
                print(f"   Children:")
                for c in child_texts:
                    print(f"     - {c['title']} (type: {c['node_type']})")
            else:
                # Check for unique texts
                unique_texts = len(set(c['text'] for c in child_texts))
                print(f"\n✓ UNIQUE TEXT in: {current_path}")
                print(f"   {len(child_texts)} children with {unique_texts} unique texts")
                print(f"   Text lengths: {[c['length'] for c in child_texts]}")
        
        # Recursively check children
        for child in children:
            check_node(child, current_path)
    
    print("=" * 70)
    print("Checking text uniqueness in structure...")
    print("=" * 70)
    
    for node in structure:
        check_node(node)
    
    print("\n" + "=" * 70)
    print("Check complete!")
    print("=" * 70)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python test_text_fix.py <structure.json>")
        sys.exit(1)
    
    structure_file = sys.argv[1]
    
    if not Path(structure_file).exists():
        print(f"Error: File not found: {structure_file}")
        sys.exit(1)
    
    check_text_uniqueness(structure_file)
