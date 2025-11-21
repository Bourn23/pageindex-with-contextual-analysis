"""
Simple integration test for granular features.

This test verifies that the granular features can be enabled and that
the integration points work correctly.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pageindex.utils import ConfigLoader


def test_config_loading():
    """Test that granular config parameters load correctly."""
    print("Testing config loading...")
    
    # Test default config
    config_loader = ConfigLoader()
    opt = config_loader.load()
    
    assert hasattr(opt, 'granularity'), "Config missing 'granularity' attribute"
    assert hasattr(opt, 'enable_figure_detection'), "Config missing 'enable_figure_detection' attribute"
    assert hasattr(opt, 'enable_table_detection'), "Config missing 'enable_table_detection' attribute"
    assert hasattr(opt, 'enable_semantic_subdivision'), "Config missing 'enable_semantic_subdivision' attribute"
    assert hasattr(opt, 'semantic_min_pages'), "Config missing 'semantic_min_pages' attribute"
    
    print(f"  ✓ Default granularity: {opt.granularity}")
    print(f"  ✓ Figure detection: {opt.enable_figure_detection}")
    print(f"  ✓ Table detection: {opt.enable_table_detection}")
    print(f"  ✓ Semantic subdivision: {opt.enable_semantic_subdivision}")
    print(f"  ✓ Min pages: {opt.semantic_min_pages}")
    
    # Test custom config
    custom_opt = config_loader.load({
        'granularity': 'medium',
        'enable_figure_detection': False,
        'semantic_min_pages': 1.0
    })
    
    assert custom_opt.granularity == 'medium', "Custom granularity not applied"
    assert custom_opt.enable_figure_detection == False, "Custom figure detection not applied"
    assert custom_opt.semantic_min_pages == 1.0, "Custom min pages not applied"
    
    print("  ✓ Custom config overrides work correctly")
    print()


def test_integration_imports():
    """Test that granular integration modules can be imported."""
    print("Testing integration imports...")
    
    try:
        from pageindex.granular.integration import (
            apply_semantic_subdivision,
            detect_and_integrate_figures_tables,
            insert_node_into_tree
        )
        print("  ✓ Integration functions imported successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import integration functions: {e}")
        raise
    
    try:
        from pageindex.granular.semantic_analyzer import SemanticAnalyzer
        print("  ✓ SemanticAnalyzer imported successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import SemanticAnalyzer: {e}")
        raise
    
    try:
        from pageindex.granular.figure_detector import FigureDetector
        print("  ✓ FigureDetector imported successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import FigureDetector: {e}")
        raise
    
    try:
        from pageindex.granular.table_detector import TableDetector
        print("  ✓ TableDetector imported successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import TableDetector: {e}")
        raise
    
    print()


def test_tree_parser_import():
    """Test that tree_parser can be imported with granular features."""
    print("Testing tree_parser import...")
    
    try:
        from pageindex.page_index import tree_parser
        print("  ✓ tree_parser imported successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import tree_parser: {e}")
        raise
    
    print()


def test_backward_compatibility():
    """Test that coarse granularity maintains backward compatibility."""
    print("Testing backward compatibility...")
    
    config_loader = ConfigLoader()
    
    # Test with no granularity specified (should default to coarse)
    opt = config_loader.load({})
    assert opt.granularity == 'coarse', "Default granularity should be 'coarse'"
    print("  ✓ Default granularity is 'coarse' (backward compatible)")
    
    # Test that existing code patterns still work
    opt = config_loader.load({
        'model': 'gpt-4o-2024-11-20',
        'toc_check_page_num': 20,
        'if_add_node_id': 'yes'
    })
    assert opt.granularity == 'coarse', "Granularity should default to 'coarse' when not specified"
    print("  ✓ Existing config patterns work without modification")
    
    print()


if __name__ == '__main__':
    print("=" * 60)
    print("Granular Features Integration Test")
    print("=" * 60)
    print()
    
    try:
        test_config_loading()
        test_integration_imports()
        test_tree_parser_import()
        test_backward_compatibility()
        
        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print()
        print("=" * 60)
        print(f"✗ Test failed: {e}")
        print("=" * 60)
        sys.exit(1)
