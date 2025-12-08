#!/usr/bin/env python3
"""
Test script for coverage validation and gap-filling functionality.

Tests:
1. Gap-filling in semantic subdivision
2. Coverage validation for tree structures
3. Per-node coverage validation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pageindex.coverage_validator import (
    validate_tree_coverage,
    validate_node_coverage,
    validate_full_tree_coverage,
    CoverageReport
)


def test_coverage_report_str():
    """Test CoverageReport string representation."""
    print("\n" + "=" * 60)
    print("Test: CoverageReport string representation")
    print("=" * 60)
    
    report = CoverageReport(
        total_paragraphs=10,
        covered_paragraphs=8,
        uncovered_paragraphs=[3, 7],
        coverage_percentage=80.0,
        total_characters=1000,
        covered_characters=800,
        uncovered_characters=200,
        character_coverage_percentage=80.0,
        gaps=[{'start': 3, 'end': 3, 'char_count': 100, 'paragraph_count': 1}],
        warnings=["Large gap at paragraph 3"],
        is_complete=False
    )
    
    print(str(report))
    assert "80.0%" in str(report)
    assert "INCOMPLETE" in str(report)
    print("✓ CoverageReport string representation works")


def test_validate_tree_coverage_complete():
    """Test coverage validation with complete coverage."""
    print("\n" + "=" * 60)
    print("Test: Complete tree coverage")
    print("=" * 60)
    
    source_text = """First paragraph with some content.

Second paragraph with more content.

Third paragraph to complete the text."""
    
    # Tree that covers all paragraphs
    tree = [
        {
            'title': 'Section 1',
            'text': 'First paragraph with some content.',
            'nodes': []
        },
        {
            'title': 'Section 2', 
            'text': 'Second paragraph with more content.',
            'nodes': []
        },
        {
            'title': 'Section 3',
            'text': 'Third paragraph to complete the text.',
            'nodes': []
        }
    ]
    
    report = validate_tree_coverage(tree, source_text)
    print(str(report))
    
    assert report.coverage_percentage >= 99.0, f"Expected >= 99% coverage, got {report.coverage_percentage}%"
    assert report.is_complete, "Expected complete coverage"
    print("✓ Complete coverage validated correctly")


def test_validate_tree_coverage_incomplete():
    """Test coverage validation with missing paragraphs."""
    print("\n" + "=" * 60)
    print("Test: Incomplete tree coverage")
    print("=" * 60)
    
    source_text = """First paragraph.

Second paragraph that will be missing.

Third paragraph."""
    
    # Tree missing the second paragraph
    tree = [
        {
            'title': 'Section 1',
            'text': 'First paragraph.',
            'nodes': []
        },
        {
            'title': 'Section 3',
            'text': 'Third paragraph.',
            'nodes': []
        }
    ]
    
    report = validate_tree_coverage(tree, source_text)
    print(str(report))
    
    assert report.coverage_percentage < 100.0, "Expected incomplete coverage"
    assert len(report.uncovered_paragraphs) > 0, "Expected uncovered paragraphs"
    assert not report.is_complete, "Expected incomplete status"
    print("✓ Incomplete coverage detected correctly")


def test_validate_node_coverage():
    """Test per-node coverage validation."""
    print("\n" + "=" * 60)
    print("Test: Node coverage validation")
    print("=" * 60)
    
    # Parent node with text
    node = {
        'title': 'Parent Section',
        'text': """Para 1 content here.

Para 2 content here.

Para 3 content here.

Para 4 content here.""",
        'nodes': [
            {
                'title': 'Child 1',
                'text': 'Para 1 content here.',
                'metadata': {'start_paragraph': 0, 'end_paragraph': 0},
                'nodes': []
            },
            {
                'title': 'Child 2',
                'text': 'Para 3 content here.\n\nPara 4 content here.',
                'metadata': {'start_paragraph': 2, 'end_paragraph': 3},
                'nodes': []
            }
            # Note: Para 2 (index 1) is missing!
        ]
    }
    
    report = validate_node_coverage(node)
    print(str(report))
    
    assert 1 in report.uncovered_paragraphs, "Expected paragraph 1 to be uncovered"
    assert not report.is_complete, "Expected incomplete coverage"
    print("✓ Node coverage gap detected correctly")


def test_validate_node_coverage_with_metadata():
    """Test node coverage using metadata for precise tracking."""
    print("\n" + "=" * 60)
    print("Test: Node coverage with metadata")
    print("=" * 60)
    
    node = {
        'title': 'Full Coverage Section',
        'text': """Para 0.

Para 1.

Para 2.""",
        'nodes': [
            {
                'title': 'Unit 1',
                'text': 'Para 0.',
                'metadata': {'start_paragraph': 0, 'end_paragraph': 0},
                'nodes': []
            },
            {
                'title': 'Unit 2',
                'text': 'Para 1.\n\nPara 2.',
                'metadata': {'start_paragraph': 1, 'end_paragraph': 2},
                'nodes': []
            }
        ]
    }
    
    report = validate_node_coverage(node)
    print(str(report))
    
    assert report.is_complete, f"Expected complete coverage, got {report.coverage_percentage}%"
    assert len(report.uncovered_paragraphs) == 0, "Expected no uncovered paragraphs"
    print("✓ Full coverage with metadata validated correctly")


def test_gap_identification():
    """Test that gaps are correctly identified."""
    print("\n" + "=" * 60)
    print("Test: Gap identification")
    print("=" * 60)
    
    source_text = """P0.

P1.

P2.

P3.

P4.

P5."""
    
    # Tree with gaps at paragraphs 1-2 and 4
    tree = [
        {'title': 'A', 'text': 'P0.', 'nodes': []},
        {'title': 'B', 'text': 'P3.', 'nodes': []},
        {'title': 'C', 'text': 'P5.', 'nodes': []}
    ]
    
    report = validate_tree_coverage(tree, source_text)
    print(str(report))
    
    assert len(report.gaps) >= 1, "Expected at least one gap"
    print(f"Found {len(report.gaps)} gaps: {report.gaps}")
    print("✓ Gaps identified correctly")


def test_empty_tree():
    """Test coverage validation with empty tree."""
    print("\n" + "=" * 60)
    print("Test: Empty tree")
    print("=" * 60)
    
    report = validate_tree_coverage([], "Some source text.")
    print(str(report))
    
    assert not report.is_complete, "Empty tree should not be complete"
    print("✓ Empty tree handled correctly")


def test_empty_source():
    """Test coverage validation with empty source."""
    print("\n" + "=" * 60)
    print("Test: Empty source")
    print("=" * 60)
    
    report = validate_tree_coverage([{'title': 'Node', 'text': '', 'nodes': []}], "")
    print(str(report))
    
    assert report.is_complete, "Empty source should be complete"
    print("✓ Empty source handled correctly")


def run_all_tests():
    """Run all coverage validation tests."""
    print("\n" + "=" * 70)
    print("COVERAGE VALIDATION TESTS")
    print("=" * 70)
    
    tests = [
        test_coverage_report_str,
        test_validate_tree_coverage_complete,
        test_validate_tree_coverage_incomplete,
        test_validate_node_coverage,
        test_validate_node_coverage_with_metadata,
        test_gap_identification,
        test_empty_tree,
        test_empty_source,
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
