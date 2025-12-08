#!/usr/bin/env python3
"""
Test script for gap-filling functionality in semantic subdivision.

Tests that uncovered paragraphs are automatically filled with "Additional Content" nodes.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pageindex.granular.semantic_analyzer import SemanticAnalyzer, SemanticUnit
from pageindex.llm_client import get_llm_client


def test_gap_filling_basic():
    """Test basic gap filling between semantic units."""
    print("\n" + "=" * 60)
    print("Test: Basic gap filling")
    print("=" * 60)
    
    # Create a mock LLM client (we won't actually call it)
    try:
        llm_client = get_llm_client()
    except:
        print("Warning: No LLM client available, using None")
        llm_client = None
    
    analyzer = SemanticAnalyzer(llm_client)
    
    # Create semantic units with a gap (missing paragraph 2)
    semantic_units = [
        SemanticUnit(
            title="Unit 1",
            start_paragraph=0,
            end_paragraph=1,
            start_page=1,
            end_page=1,
            semantic_type="intro",
            summary="First unit"
        ),
        SemanticUnit(
            title="Unit 2",
            start_paragraph=3,
            end_paragraph=4,
            start_page=1,
            end_page=1,
            semantic_type="body",
            summary="Second unit"
        )
    ]
    
    # Section node with 5 paragraphs
    section_node = {
        'title': 'Test Section',
        'text': """Paragraph 0 content.

Paragraph 1 content.

Paragraph 2 content that will be in a gap.

Paragraph 3 content.

Paragraph 4 content.""",
        'start_index': 1,
        'end_index': 1
    }
    
    page_texts = [("Full page text", 100)]
    
    # Test with gap filling enabled (default)
    nodes = analyzer.create_nodes_from_semantic_units(
        semantic_units, section_node, page_texts, fill_gaps=True
    )
    
    print(f"Created {len(nodes)} nodes:")
    for node in nodes:
        is_gap = node.get('metadata', {}).get('is_gap_fill', False)
        marker = " [GAP FILL]" if is_gap else ""
        print(f"  - {node['title']}: paragraphs {node['metadata']['start_paragraph']}-{node['metadata']['end_paragraph']}{marker}")
    
    # Should have 3 nodes: Unit 1, Gap (paragraph 2), Unit 2
    assert len(nodes) == 3, f"Expected 3 nodes, got {len(nodes)}"
    
    # Check that the gap node exists
    gap_nodes = [n for n in nodes if n.get('metadata', {}).get('is_gap_fill')]
    assert len(gap_nodes) == 1, f"Expected 1 gap node, got {len(gap_nodes)}"
    
    gap_node = gap_nodes[0]
    assert gap_node['metadata']['start_paragraph'] == 2
    assert gap_node['metadata']['end_paragraph'] == 2
    assert "Paragraph 2 content" in gap_node['text']
    
    print("✓ Gap filling works correctly")


def test_gap_filling_trailing():
    """Test gap filling for trailing paragraphs."""
    print("\n" + "=" * 60)
    print("Test: Trailing gap filling")
    print("=" * 60)
    
    try:
        llm_client = get_llm_client()
    except:
        llm_client = None
    
    analyzer = SemanticAnalyzer(llm_client)
    
    # Semantic unit only covers first 2 paragraphs
    semantic_units = [
        SemanticUnit(
            title="Unit 1",
            start_paragraph=0,
            end_paragraph=1,
            start_page=1,
            end_page=1,
            semantic_type="intro",
            summary="First unit"
        )
    ]
    
    # Section has 4 paragraphs
    section_node = {
        'title': 'Test Section',
        'text': """Para 0.

Para 1.

Para 2 trailing.

Para 3 trailing.""",
        'start_index': 1,
        'end_index': 1
    }
    
    page_texts = [("text", 100)]
    
    nodes = analyzer.create_nodes_from_semantic_units(
        semantic_units, section_node, page_texts, fill_gaps=True
    )
    
    print(f"Created {len(nodes)} nodes:")
    for node in nodes:
        is_gap = node.get('metadata', {}).get('is_gap_fill', False)
        marker = " [GAP FILL]" if is_gap else ""
        print(f"  - {node['title']}: paragraphs {node['metadata']['start_paragraph']}-{node['metadata']['end_paragraph']}{marker}")
    
    # Should have 2 nodes: Unit 1, trailing gap
    assert len(nodes) == 2, f"Expected 2 nodes, got {len(nodes)}"
    
    gap_nodes = [n for n in nodes if n.get('metadata', {}).get('is_gap_fill')]
    assert len(gap_nodes) == 1, f"Expected 1 gap node, got {len(gap_nodes)}"
    
    gap_node = gap_nodes[0]
    assert gap_node['metadata']['start_paragraph'] == 2
    assert gap_node['metadata']['end_paragraph'] == 3
    
    print("✓ Trailing gap filling works correctly")


def test_gap_filling_leading():
    """Test gap filling for leading paragraphs."""
    print("\n" + "=" * 60)
    print("Test: Leading gap filling")
    print("=" * 60)
    
    try:
        llm_client = get_llm_client()
    except:
        llm_client = None
    
    analyzer = SemanticAnalyzer(llm_client)
    
    # Semantic unit starts at paragraph 2
    semantic_units = [
        SemanticUnit(
            title="Unit 1",
            start_paragraph=2,
            end_paragraph=3,
            start_page=1,
            end_page=1,
            semantic_type="body",
            summary="Main unit"
        )
    ]
    
    section_node = {
        'title': 'Test Section',
        'text': """Para 0 leading.

Para 1 leading.

Para 2.

Para 3.""",
        'start_index': 1,
        'end_index': 1
    }
    
    page_texts = [("text", 100)]
    
    nodes = analyzer.create_nodes_from_semantic_units(
        semantic_units, section_node, page_texts, fill_gaps=True
    )
    
    print(f"Created {len(nodes)} nodes:")
    for node in nodes:
        is_gap = node.get('metadata', {}).get('is_gap_fill', False)
        marker = " [GAP FILL]" if is_gap else ""
        print(f"  - {node['title']}: paragraphs {node['metadata']['start_paragraph']}-{node['metadata']['end_paragraph']}{marker}")
    
    # Should have 2 nodes: leading gap, Unit 1
    assert len(nodes) == 2, f"Expected 2 nodes, got {len(nodes)}"
    
    # First node should be the gap (sorted by start_paragraph)
    assert nodes[0]['metadata'].get('is_gap_fill'), "First node should be gap fill"
    assert nodes[0]['metadata']['start_paragraph'] == 0
    assert nodes[0]['metadata']['end_paragraph'] == 1
    
    print("✓ Leading gap filling works correctly")


def test_no_gap_filling():
    """Test that gap filling can be disabled."""
    print("\n" + "=" * 60)
    print("Test: Gap filling disabled")
    print("=" * 60)
    
    try:
        llm_client = get_llm_client()
    except:
        llm_client = None
    
    analyzer = SemanticAnalyzer(llm_client)
    
    semantic_units = [
        SemanticUnit(
            title="Unit 1",
            start_paragraph=0,
            end_paragraph=0,
            start_page=1,
            end_page=1,
            semantic_type="intro",
            summary="First"
        ),
        SemanticUnit(
            title="Unit 2",
            start_paragraph=2,
            end_paragraph=2,
            start_page=1,
            end_page=1,
            semantic_type="body",
            summary="Second"
        )
    ]
    
    section_node = {
        'title': 'Test',
        'text': """P0.

P1 gap.

P2.""",
        'start_index': 1,
        'end_index': 1
    }
    
    page_texts = [("text", 100)]
    
    # Disable gap filling
    nodes = analyzer.create_nodes_from_semantic_units(
        semantic_units, section_node, page_texts, fill_gaps=False
    )
    
    print(f"Created {len(nodes)} nodes (gap filling disabled):")
    for node in nodes:
        print(f"  - {node['title']}")
    
    # Should only have 2 nodes (no gap fill)
    assert len(nodes) == 2, f"Expected 2 nodes, got {len(nodes)}"
    
    gap_nodes = [n for n in nodes if n.get('metadata', {}).get('is_gap_fill')]
    assert len(gap_nodes) == 0, "Expected no gap nodes when fill_gaps=False"
    
    print("✓ Gap filling disabled correctly")


def test_full_coverage_no_gaps():
    """Test that no gap nodes are created when coverage is complete."""
    print("\n" + "=" * 60)
    print("Test: Full coverage (no gaps needed)")
    print("=" * 60)
    
    try:
        llm_client = get_llm_client()
    except:
        llm_client = None
    
    analyzer = SemanticAnalyzer(llm_client)
    
    # Contiguous semantic units covering all paragraphs
    semantic_units = [
        SemanticUnit(
            title="Unit 1",
            start_paragraph=0,
            end_paragraph=1,
            start_page=1,
            end_page=1,
            semantic_type="intro",
            summary="First"
        ),
        SemanticUnit(
            title="Unit 2",
            start_paragraph=2,
            end_paragraph=3,
            start_page=1,
            end_page=1,
            semantic_type="body",
            summary="Second"
        )
    ]
    
    section_node = {
        'title': 'Test',
        'text': """P0.

P1.

P2.

P3.""",
        'start_index': 1,
        'end_index': 1
    }
    
    page_texts = [("text", 100)]
    
    nodes = analyzer.create_nodes_from_semantic_units(
        semantic_units, section_node, page_texts, fill_gaps=True
    )
    
    print(f"Created {len(nodes)} nodes:")
    for node in nodes:
        is_gap = node.get('metadata', {}).get('is_gap_fill', False)
        marker = " [GAP FILL]" if is_gap else ""
        print(f"  - {node['title']}{marker}")
    
    # Should have exactly 2 nodes (no gaps)
    assert len(nodes) == 2, f"Expected 2 nodes, got {len(nodes)}"
    
    gap_nodes = [n for n in nodes if n.get('metadata', {}).get('is_gap_fill')]
    assert len(gap_nodes) == 0, "Expected no gap nodes for full coverage"
    
    print("✓ Full coverage handled correctly (no unnecessary gaps)")


def run_all_tests():
    """Run all gap filling tests."""
    print("\n" + "=" * 70)
    print("GAP FILLING TESTS")
    print("=" * 70)
    
    tests = [
        test_gap_filling_basic,
        test_gap_filling_trailing,
        test_gap_filling_leading,
        test_no_gap_filling,
        test_full_coverage_no_gaps,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
