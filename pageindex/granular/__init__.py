"""
Granular node generation module for PageIndex.

This module provides components for creating fine-grained nodes including:
- Figure detection and node creation
- Table detection and node creation
- Semantic section subdivision
- Tree integration utilities
"""

from .figure_detector import FigureDetector, FigureNode
from .table_detector import TableDetector, TableNode
from .semantic_analyzer import SemanticAnalyzer, SemanticUnit
from .integration import (
    apply_semantic_subdivision,
    detect_and_integrate_figures_tables,
    insert_node_into_tree,
    reassign_hierarchical_node_ids
)

__all__ = [
    'FigureDetector', 
    'FigureNode',
    'TableDetector',
    'TableNode',
    'SemanticAnalyzer',
    'SemanticUnit',
    'apply_semantic_subdivision',
    'detect_and_integrate_figures_tables',
    'insert_node_into_tree',
    'reassign_hierarchical_node_ids'
]
