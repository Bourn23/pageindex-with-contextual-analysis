#!/usr/bin/env python3
"""
Tests for markdown processing functionality.
"""

import json
from pathlib import Path

from pageindex.markdown_processor import MarkdownProcessor, process_markdown_to_tree
from pageindex.markdown_integration import markdown_page_index


def test_markdown_processor_initialization():
    """Test MarkdownProcessor initialization."""
    print("\n[TEST] MarkdownProcessor initialization...")
    
    md_path = "PageIndex/tests/markdowns/Influence of the LLZO-PEO interface on the micro- and macro-scale properties of composite polymer electrolytes for solid-state batteries/Influence of the LLZO-PEO interface on the micro- and macro-scale properties of composite polymer electrolytes for solid-state batteries.md"
    meta_path = "PageIndex/tests/markdowns/Influence of the LLZO-PEO interface on the micro- and macro-scale properties of composite polymer electrolytes for solid-state batteries/Influence of the LLZO-PEO interface on the micro- and macro-scale properties of composite polymer electrolytes for solid-state batteries_meta.json"
    
    if not Path(md_path).exists():
        print("  ⚠️  SKIP: Test file not found")
        return False
    
    processor = MarkdownProcessor(md_path, meta_path)
    
    assert processor.markdown_content is not None, "Markdown content not loaded"
    assert processor.metadata is not None, "Metadata not loaded"
    assert len(processor.lines) > 0, "No lines extracted"
    
    print("  ✓ PASS")
    return True


def test_structure_extraction():
    """Test structure extraction from markdown."""
    print("\n[TEST] Structure extraction...")
    
    md_path = "PageIndex/tests/markdowns/Influence of the LLZO-PEO interface on the micro- and macro-scale properties of composite polymer electrolytes for solid-state batteries/Influence of the LLZO-PEO interface on the micro- and macro-scale properties of composite polymer electrolytes for solid-state batteries.md"
    meta_path = "PageIndex/tests/markdowns/Influence of the LLZO-PEO interface on the micro- and macro-scale properties of composite polymer electrolytes for solid-state batteries/Influence of the LLZO-PEO interface on the micro- and macro-scale properties of composite polymer electrolytes for solid-state batteries_meta.json"
    
    if not Path(md_path).exists():
        print("  ⚠️  SKIP: Test file not found")
        return False
    
    result = process_markdown_to_tree(md_path, meta_path)
    
    assert 'tree' in result, "No tree in result"
    assert 'page_count' in result, "No page count"
    assert len(result['tree']) > 0, "Empty tree"
    
    print(f"  Found {len(result['tree'])} top-level sections")
    print(f"  Total pages: {result['page_count']}")
    print("  ✓ PASS")
    return True


def test_figure_extraction():
    """Test figure extraction from markdown."""
    print("\n[TEST] Figure extraction...")
    
    md_path = "PageIndex/tests/markdowns/Influence of the LLZO-PEO interface on the micro- and macro-scale properties of composite polymer electrolytes for solid-state batteries/Influence of the LLZO-PEO interface on the micro- and macro-scale properties of composite polymer electrolytes for solid-state batteries.md"
    meta_path = "PageIndex/tests/markdowns/Influence of the LLZO-PEO interface on the micro- and macro-scale properties of composite polymer electrolytes for solid-state batteries/Influence of the LLZO-PEO interface on the micro- and macro-scale properties of composite polymer electrolytes for solid-state batteries_meta.json"
    
    if not Path(md_path).exists():
        print("  ⚠️  SKIP: Test file not found")
        return False
    
    processor = MarkdownProcessor(md_path, meta_path)
    figures, tables = processor.extract_figures_and_tables()
    
    assert len(figures) > 0, "No figures found"
    assert all('page' in fig for fig in figures), "Figures missing page info"
    
    print(f"  Found {len(figures)} figures")
    print(f"  Found {len(tables)} tables")
    print("  ✓ PASS")
    return True


def test_pageindex_integration():
    """Test integration with PageIndex."""
    print("\n[TEST] PageIndex integration...")
    
    md_path = "PageIndex/tests/markdowns/Influence of the LLZO-PEO interface on the micro- and macro-scale properties of composite polymer electrolytes for solid-state batteries/Influence of the LLZO-PEO interface on the micro- and macro-scale properties of composite polymer electrolytes for solid-state batteries.md"
    meta_path = "PageIndex/tests/markdowns/Influence of the LLZO-PEO interface on the micro- and macro-scale properties of composite polymer electrolytes for solid-state batteries/Influence of the LLZO-PEO interface on the micro- and macro-scale properties of composite polymer electrolytes for solid-state batteries_meta.json"
    
    if not Path(md_path).exists():
        print("  ⚠️  SKIP: Test file not found")
        return False
    
    structure = markdown_page_index(
        markdown_path=md_path,
        metadata_path=meta_path,
        opt={'if_add_node_text': 'no'}
    )
    
    assert 'tree' in structure, "No tree in structure"
    assert 'page_count' in structure, "No page count"
    assert structure['source'] == 'markdown', "Wrong source type"
    
    # Check node IDs
    first_node = structure['tree'][0]
    assert 'node_id' in first_node, "No node ID"
    
    print(f"  Structure has {len(structure['tree'])} sections")
    print(f"  First node ID: {first_node['node_id']}")
    print("  ✓ PASS")
    return True


def test_node_hierarchy():
    """Test hierarchical node structure."""
    print("\n[TEST] Node hierarchy...")
    
    md_path = "PageIndex/tests/markdowns/Influence of the LLZO-PEO interface on the micro- and macro-scale properties of composite polymer electrolytes for solid-state batteries/Influence of the LLZO-PEO interface on the micro- and macro-scale properties of composite polymer electrolytes for solid-state batteries.md"
    meta_path = "PageIndex/tests/markdowns/Influence of the LLZO-PEO interface on the micro- and macro-scale properties of composite polymer electrolytes for solid-state batteries/Influence of the LLZO-PEO interface on the micro- and macro-scale properties of composite polymer electrolytes for solid-state batteries_meta.json"
    
    if not Path(md_path).exists():
        print("  ⚠️  SKIP: Test file not found")
        return False
    
    structure = markdown_page_index(md_path, meta_path)
    
    def check_hierarchy(nodes, parent_id=""):
        for i, node in enumerate(nodes):
            expected_id = f"{parent_id}{i+1}" if parent_id else str(i+1)
            assert node['node_id'] == expected_id, f"Wrong node ID: {node['node_id']} != {expected_id}"
            
            if 'children' in node and node['children']:
                check_hierarchy(node['children'], f"{expected_id}.")
    
    check_hierarchy(structure['tree'])
    print("  ✓ PASS")
    return True


def main():
    """Run all tests."""
    print("="*60)
    print("Markdown Processor Tests")
    print("="*60)
    
    tests = [
        test_markdown_processor_initialization,
        test_structure_extraction,
        test_figure_extraction,
        test_pageindex_integration,
        test_node_hierarchy
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test in tests:
        try:
            result = test()
            if result is False:
                skipped += 1
            else:
                passed += 1
        except AssertionError as e:
            print(f"  ✗ FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
