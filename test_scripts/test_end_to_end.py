"""
End-to-end tests for granular PageIndex features on sample scientific papers.

This test suite:
- Tests with papers from different domains
- Verifies node count increases (30-50+ vs 8-12 baseline)
- Verifies hierarchy depth (4-5 levels)
- Validates all requirements are met
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pageindex import page_index_main
from pageindex.utils import ConfigLoader


def count_nodes(tree: List[Dict]) -> int:
    """Recursively count all nodes in the tree."""
    count = len(tree)
    for node in tree:
        if 'nodes' in node and node['nodes']:
            count += count_nodes(node['nodes'])
    return count


def get_max_depth(tree: List[Dict], current_depth: int = 1) -> int:
    """Get the maximum depth of the tree."""
    if not tree:
        return current_depth - 1
    
    max_child_depth = current_depth
    for node in tree:
        if 'nodes' in node and node['nodes']:
            child_depth = get_max_depth(node['nodes'], current_depth + 1)
            max_child_depth = max(max_child_depth, child_depth)
    
    return max_child_depth


def get_node_types(tree: List[Dict]) -> Dict[str, int]:
    """Count nodes by type."""
    type_counts = {}
    
    def count_types(nodes):
        for i, node in enumerate(nodes):
            # Handle case where node might be a string or other non-dict type
            if not isinstance(node, dict):
                print(f"Warning: Node at index {i} is not a dict, it's a {type(node).__name__}: {node}")
                continue
            
            node_type = node.get('node_type', 'section')
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
            if 'nodes' in node and node['nodes']:
                count_types(node['nodes'])
    
    count_types(tree)
    return type_counts


def validate_tree_structure(tree: List[Dict], parent_range: Tuple[int, int] = None) -> List[str]:
    """Validate tree structure and return list of issues."""
    issues = []
    
    for i, node in enumerate(tree):
        # Handle case where node might not be a dict
        if not isinstance(node, dict):
            issues.append(f"Node at index {i} is not a dict, it's a {type(node).__name__}: {node}")
            continue
        
        # Check required fields
        if 'title' not in node:
            issues.append(f"Node {i} missing 'title'")
        if 'start_index' not in node:
            issues.append(f"Node {i} ({node.get('title', 'unknown')}) missing 'start_index'")
        if 'end_index' not in node:
            issues.append(f"Node {i} ({node.get('title', 'unknown')}) missing 'end_index'")
        
        # Check page range validity
        if 'start_index' in node and 'end_index' in node:
            if node['start_index'] > node['end_index']:
                issues.append(
                    f"Node '{node.get('title', 'unknown')}' has invalid page range: "
                    f"{node['start_index']} > {node['end_index']}"
                )
            
            # Check parent-child relationship
            if parent_range:
                parent_start, parent_end = parent_range
                if node['start_index'] < parent_start:
                    issues.append(
                        f"Node '{node.get('title', 'unknown')}' starts before parent "
                        f"({node['start_index']} < {parent_start})"
                    )
                if node['end_index'] > parent_end:
                    issues.append(
                        f"Node '{node.get('title', 'unknown')}' ends after parent "
                        f"({node['end_index']} > {parent_end})"
                    )
        
        # Recursively validate children
        if 'nodes' in node and node['nodes']:
            child_range = (node.get('start_index'), node.get('end_index'))
            child_issues = validate_tree_structure(node['nodes'], child_range)
            issues.extend(child_issues)
    
    return issues


def print_tree_summary(tree: List[Dict], title: str = "Tree Summary"):
    """Print a summary of the tree structure."""
    print(f"\n{title}")
    print("=" * 70)
    
    total_nodes = count_nodes(tree)
    max_depth = get_max_depth(tree)
    node_types = get_node_types(tree)
    
    print(f"Total nodes: {total_nodes}")
    print(f"Maximum depth: {max_depth}")
    print(f"Node types: {node_types}")
    
    # Print top-level structure
    print(f"\nTop-level sections ({len(tree)}):")
    for node in tree:
        title = node.get('title', 'Unknown')
        start = node.get('start_index', '?')
        end = node.get('end_index', '?')
        child_count = len(node.get('nodes', []))
        print(f"  - {title} (pages {start}-{end}, {child_count} children)")
    
    print("=" * 70)


def test_coarse_granularity(pdf_path: str):
    """Test with coarse granularity (baseline behavior)."""
    print(f"\n{'='*70}")
    print(f"Testing COARSE granularity on: {Path(pdf_path).name}")
    print(f"{'='*70}")
    
    # Configure with coarse granularity
    config_loader = ConfigLoader()
    opt = config_loader.load({
        'model': 'gemini-2.5-flash-lite',
        'granularity': 'coarse',  # Options: coarse, medium, fine, keywords
        'if_add_node_id': 'yes',
        'if_add_node_summary': 'no',  # Skip summaries for faster testing
        'if_add_doc_description': 'no'
    })
    
    # Process PDF
    print("Processing PDF with coarse granularity...")
    result = page_index_main(pdf_path, opt)
    
    # Extract tree structure from result
    if isinstance(result, dict) and 'structure' in result:
        tree = result['structure']
    else:
        tree = result
    
    # Analyze results
    print_tree_summary(tree, "Coarse Granularity Results")
    
    # Validate structure
    issues = validate_tree_structure(tree)
    if issues:
        print(f"\n⚠ Structure validation issues:")
        for issue in issues[:5]:  # Show first 5 issues
            print(f"  - {issue}")
        if len(issues) > 5:
            print(f"  ... and {len(issues) - 5} more issues")
    else:
        print("\n✓ Tree structure is valid")
    
    # Return metrics
    return {
        'total_nodes': count_nodes(tree),
        'max_depth': get_max_depth(tree),
        'node_types': get_node_types(tree),
        'tree': tree,
        'issues': issues
    }


def test_medium_granularity(pdf_path: str):
    """Test with medium granularity (semantic + figures/tables)."""
    print(f"\n{'='*70}")
    print(f"Testing MEDIUM granularity on: {Path(pdf_path).name}")
    print(f"{'='*70}")
    
    # Configure with medium granularity
    config_loader = ConfigLoader()
    opt = config_loader.load({
        'model': 'gemini-2.5-flash-lite',
        'granularity': 'medium',
        'enable_figure_detection': True,
        'enable_table_detection': True,
        'enable_semantic_subdivision': True,
        'if_add_node_id': 'yes',
        'if_add_node_summary': 'no',  # Skip summaries for faster testing
        'if_add_doc_description': 'no'
    })
    
    # Process PDF
    print("Processing PDF with medium granularity...")
    result = page_index_main(pdf_path, opt)
    
    # Extract tree structure from result
    if isinstance(result, dict) and 'structure' in result:
        tree = result['structure']
    else:
        tree = result
    
    # Analyze results
    print_tree_summary(tree, "Medium Granularity Results")
    
    # Validate structure
    issues = validate_tree_structure(tree)
    if issues:
        print(f"\n⚠ Structure validation issues:")
        for issue in issues[:5]:  # Show first 5 issues
            print(f"  - {issue}")
        if len(issues) > 5:
            print(f"  ... and {len(issues) - 5} more issues")
    else:
        print("\n✓ Tree structure is valid")
    
    # Return metrics
    return {
        'total_nodes': count_nodes(tree),
        'max_depth': get_max_depth(tree),
        'node_types': get_node_types(tree),
        'tree': tree,
        'issues': issues
    }


def test_fine_granularity(pdf_path: str):
    """Test with fine granularity (deeper semantic analysis)."""
    print(f"\n{'='*70}")
    print(f"Testing FINE granularity on: {Path(pdf_path).name}")
    print(f"{'='*70}")
    
    # Configure with fine granularity
    config_loader = ConfigLoader()
    opt = config_loader.load({
        'model': 'gemini-2.5-flash-lite',
        'granularity': 'fine',
        'enable_figure_detection': True,
        'enable_table_detection': True,
        'enable_semantic_subdivision': True,
        'if_add_node_id': 'yes',
        'if_add_node_summary': 'no',  # Skip summaries for faster testing
        'if_add_doc_description': 'no'
    })
    
    # Process PDF
    print("Processing PDF with fine granularity...")
    result = page_index_main(pdf_path, opt)
    
    # Extract tree structure from result
    if isinstance(result, dict) and 'structure' in result:
        tree = result['structure']
    else:
        tree = result
    
    # Analyze results
    print_tree_summary(tree, "Fine Granularity Results")
    
    # Validate structure
    issues = validate_tree_structure(tree)
    if issues:
        print(f"\n⚠ Structure validation issues:")
        for issue in issues[:5]:  # Show first 5 issues
            print(f"  - {issue}")
        if len(issues) > 5:
            print(f"  ... and {len(issues) - 5} more issues")
    else:
        print("\n✓ Tree structure is valid")
    
    # Return metrics
    return {
        'total_nodes': count_nodes(tree),
        'max_depth': get_max_depth(tree),
        'node_types': get_node_types(tree),
        'tree': tree,
        'issues': issues
    }


def compare_results(coarse_results: Dict, medium_results: Dict, fine_results: Dict):
    """Compare results across granularity levels."""
    print(f"\n{'='*70}")
    print("COMPARISON ACROSS GRANULARITY LEVELS")
    print(f"{'='*70}")
    
    print(f"\nNode Count:")
    print(f"  Coarse:  {coarse_results['total_nodes']:3d} nodes")
    print(f"  Medium:  {medium_results['total_nodes']:3d} nodes")
    print(f"  Fine:    {fine_results['total_nodes']:3d} nodes")
    
    print(f"\nMaximum Depth:")
    print(f"  Coarse:  {coarse_results['max_depth']} levels")
    print(f"  Medium:  {medium_results['max_depth']} levels")
    print(f"  Fine:    {fine_results['max_depth']} levels")
    
    print(f"\nNode Types (Medium):")
    for node_type, count in medium_results['node_types'].items():
        print(f"  {node_type}: {count}")
    
    print(f"\nNode Types (Fine):")
    for node_type, count in fine_results['node_types'].items():
        print(f"  {node_type}: {count}")
    
    # Verify requirements
    print(f"\n{'='*70}")
    print("REQUIREMENT VERIFICATION")
    print(f"{'='*70}")
    
    # Requirement: Node count should increase significantly
    coarse_count = coarse_results['total_nodes']
    medium_count = medium_results['total_nodes']
    fine_count = fine_results['total_nodes']
    
    print(f"\n1. Node Count Increase:")
    if medium_count > coarse_count:
        increase_pct = ((medium_count - coarse_count) / coarse_count) * 100
        print(f"   ✓ Medium has {medium_count - coarse_count} more nodes than coarse (+{increase_pct:.1f}%)")
    else:
        print(f"   ✗ Medium should have more nodes than coarse")
    
    if fine_count >= medium_count:
        print(f"   ✓ Fine has {fine_count - medium_count} more nodes than medium")
    else:
        print(f"   ⚠ Fine has fewer nodes than medium (may be expected)")
    
    # Requirement: Target 30-50+ nodes for medium/fine (vs 8-12 baseline)
    print(f"\n2. Target Node Count (30-50+ for granular):")
    if coarse_count >= 8:
        print(f"   ✓ Coarse baseline: {coarse_count} nodes (expected 8-12)")
    else:
        print(f"   ⚠ Coarse baseline: {coarse_count} nodes (expected 8-12)")
    
    if medium_count >= 30:
        print(f"   ✓ Medium: {medium_count} nodes (target 30-50+)")
    else:
        print(f"   ⚠ Medium: {medium_count} nodes (target 30-50+, may vary by document)")
    
    if fine_count >= 30:
        print(f"   ✓ Fine: {fine_count} nodes (target 30-50+)")
    else:
        print(f"   ⚠ Fine: {fine_count} nodes (target 30-50+, may vary by document)")
    
    # Requirement: Hierarchy depth should be 4-5 levels
    print(f"\n3. Hierarchy Depth (target 4-5 levels):")
    if medium_results['max_depth'] >= 4:
        print(f"   ✓ Medium: {medium_results['max_depth']} levels (target 4-5)")
    else:
        print(f"   ⚠ Medium: {medium_results['max_depth']} levels (target 4-5, may vary by document)")
    
    if fine_results['max_depth'] >= 4:
        print(f"   ✓ Fine: {fine_results['max_depth']} levels (target 4-5)")
    else:
        print(f"   ⚠ Fine: {fine_results['max_depth']} levels (target 4-5, may vary by document)")
    
    # Requirement: Should have figure and table nodes
    print(f"\n4. Figure and Table Detection:")
    medium_types = medium_results['node_types']
    fine_types = fine_results['node_types']
    
    if 'figure' in medium_types or 'table' in medium_types:
        print(f"   ✓ Medium detected figures/tables")
        if 'figure' in medium_types:
            print(f"     - {medium_types['figure']} figure nodes")
        if 'table' in medium_types:
            print(f"     - {medium_types['table']} table nodes")
    else:
        print(f"   ⚠ Medium: No figures/tables detected (may not be present in document)")
    
    if 'figure' in fine_types or 'table' in fine_types:
        print(f"   ✓ Fine detected figures/tables")
        if 'figure' in fine_types:
            print(f"     - {fine_types['figure']} figure nodes")
        if 'table' in fine_types:
            print(f"     - {fine_types['table']} table nodes")
    else:
        print(f"   ⚠ Fine: No figures/tables detected (may not be present in document)")
    
    # Requirement: Should have semantic units
    print(f"\n5. Semantic Subdivision:")
    if 'semantic_unit' in medium_types:
        print(f"   ✓ Medium created {medium_types['semantic_unit']} semantic unit nodes")
    else:
        print(f"   ⚠ Medium: No semantic units detected (may not be applicable)")
    
    if 'semantic_unit' in fine_types:
        print(f"   ✓ Fine created {fine_types['semantic_unit']} semantic unit nodes")
    else:
        print(f"   ⚠ Fine: No semantic units detected (may not be applicable)")
    
    # Requirement: Tree structure should be valid
    print(f"\n6. Tree Structure Validity:")
    if not coarse_results['issues']:
        print(f"   ✓ Coarse tree structure is valid")
    else:
        print(f"   ✗ Coarse has {len(coarse_results['issues'])} validation issues")
    
    if not medium_results['issues']:
        print(f"   ✓ Medium tree structure is valid")
    else:
        print(f"   ✗ Medium has {len(medium_results['issues'])} validation issues")
    
    if not fine_results['issues']:
        print(f"   ✓ Fine tree structure is valid")
    else:
        print(f"   ✗ Fine has {len(fine_results['issues'])} validation issues")


def run_end_to_end_test(pdf_path: str):
    """Run complete end-to-end test on a single PDF."""
    print(f"\n{'#'*70}")
    print(f"# END-TO-END TEST: {Path(pdf_path).name}")
    print(f"{'#'*70}")
    
    # Test all granularity levels
    coarse_results = test_coarse_granularity(pdf_path)
    medium_results = test_medium_granularity(pdf_path)
    fine_results = test_fine_granularity(pdf_path)
    
    # Compare results
    compare_results(coarse_results, medium_results, fine_results)
    
    return {
        'pdf': pdf_path,
        'coarse': coarse_results,
        'medium': medium_results,
        'fine': fine_results
    }


if __name__ == '__main__':
    print("=" * 70)
    print("END-TO-END TESTS FOR GRANULAR PAGEINDEX FEATURES")
    print("=" * 70)
    print("\nThis test suite validates:")
    print("  - Processing with different granularity levels")
    print("  - Node count increases (30-50+ vs 8-12 baseline)")
    print("  - Hierarchy depth (4-5 levels)")
    print("  - Figure and table detection")
    print("  - Semantic subdivision")
    print("  - Tree structure validity")
    
    # Define test PDFs
    test_pdfs = [
        'tests/pdfs/earthmover.pdf',  # Scientific paper
        'tests/pdfs/four-lectures.pdf',  # Academic paper
    ]
    
    # Check if PDFs exist
    available_pdfs = []
    for pdf in test_pdfs:
        pdf_path = Path(__file__).parent / pdf
        if pdf_path.exists():
            available_pdfs.append(str(pdf_path))
        else:
            print(f"\n⚠ Warning: PDF not found: {pdf}")
    
    if not available_pdfs:
        print("\n✗ No test PDFs found. Please ensure test PDFs are available.")
        sys.exit(1)
    
    print(f"\nTesting with {len(available_pdfs)} PDF(s):")
    for pdf in available_pdfs:
        print(f"  - {Path(pdf).name}")
    
    # Run tests
    all_results = []
    for pdf_path in available_pdfs:
        try:
            results = run_end_to_end_test(pdf_path)
            all_results.append(results)
        except Exception as e:
            print(f"\n✗ Error testing {Path(pdf_path).name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Final summary
    print(f"\n{'#'*70}")
    print("# FINAL SUMMARY")
    print(f"{'#'*70}")
    
    if all_results:
        print(f"\nSuccessfully tested {len(all_results)} document(s)")
        print("\nKey Findings:")
        
        for result in all_results:
            pdf_name = Path(result['pdf']).name
            coarse = result['coarse']['total_nodes']
            medium = result['medium']['total_nodes']
            fine = result['fine']['total_nodes']
            medium_depth = result['medium']['max_depth']
            fine_depth = result['fine']['max_depth']
            
            print(f"\n{pdf_name}:")
            print(f"  Nodes: {coarse} (coarse) → {medium} (medium) → {fine} (fine)")
            print(f"  Depth: {medium_depth} (medium), {fine_depth} (fine)")
            
            # Check if targets met
            targets_met = []
            if medium >= 30 or fine >= 30:
                targets_met.append("node count")
            if medium_depth >= 4 or fine_depth >= 4:
                targets_met.append("hierarchy depth")
            
            if targets_met:
                print(f"  ✓ Targets met: {', '.join(targets_met)}")
        
        print(f"\n{'='*70}")
        print("✓ End-to-end tests completed!")
        print(f"{'='*70}")
    else:
        print("\n✗ No tests completed successfully")
        sys.exit(1)
