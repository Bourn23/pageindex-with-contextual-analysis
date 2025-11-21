#!/usr/bin/env python3
"""
Patch an existing structure file to fix duplicate text in semantic units.
This simulates what the fix would do during generation.
"""

import json
import sys
from pathlib import Path


def split_into_paragraphs(text):
    """Split text into paragraphs."""
    paragraphs = []
    for p in text.split('\n\n'):
        p = p.strip()
        if p:
            paragraphs.append(p)
    
    if len(paragraphs) <= 1:
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    
    return paragraphs


def patch_node(node, parent_text=None):
    """Recursively patch nodes to fix semantic unit text."""
    
    # If this is a semantic unit with metadata, extract proper text
    if node.get('node_type') == 'semantic_unit' and parent_text:
        metadata = node.get('metadata', {})
        start_para = metadata.get('start_paragraph')
        end_para = metadata.get('end_paragraph')
        
        if start_para is not None and end_para is not None:
            paragraphs = split_into_paragraphs(parent_text)
            
            if start_para < len(paragraphs) and end_para < len(paragraphs):
                unit_paragraphs = paragraphs[start_para:end_para + 1]
                unit_text = '\n\n'.join(unit_paragraphs)
                
                old_len = len(node.get('text', ''))
                node['text'] = unit_text
                new_len = len(unit_text)
                
                print(f"  ✓ Patched '{node['title'][:50]}': {old_len} → {new_len} chars (para {start_para}-{end_para})")
    
    # Process children
    children = node.get('nodes', [])
    if children:
        # Use this node's text as parent text for children
        current_text = node.get('text', '')
        for child in children:
            patch_node(child, current_text)


def patch_structure(structure_file, output_file=None):
    """Patch structure file to fix duplicate text."""
    
    print(f"Loading structure from: {structure_file}")
    with open(structure_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    structure = data.get('structure', [])
    
    print("\nPatching semantic units...")
    print("=" * 70)
    
    for node in structure:
        patch_node(node)
    
    print("=" * 70)
    print("Patching complete!")
    
    # Save patched structure
    if output_file is None:
        input_path = Path(structure_file)
        output_file = input_path.parent / f"{input_path.stem}_patched{input_path.suffix}"
    
    print(f"\nSaving patched structure to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print("✓ Done!")
    return output_file


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python patch_existing_structure.py <structure.json> [output.json]")
        sys.exit(1)
    
    structure_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(structure_file).exists():
        print(f"Error: File not found: {structure_file}")
        sys.exit(1)
    
    patched_file = patch_structure(structure_file, output_file)
    
    print(f"\nYou can now visualize the patched structure:")
    print(f"  python PageIndex/visualize_structure.py {patched_file}")
