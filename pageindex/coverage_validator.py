"""
Coverage validation utilities for PageIndex tree structures.

This module provides functions to validate that all source text is represented
in the parsed tree structure, ensuring no content is lost during processing.
"""

import logging
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field


@dataclass
class CoverageReport:
    """Report of text coverage analysis."""
    total_paragraphs: int = 0
    covered_paragraphs: int = 0
    uncovered_paragraphs: List[int] = field(default_factory=list)
    coverage_percentage: float = 0.0
    
    total_characters: int = 0
    covered_characters: int = 0
    uncovered_characters: int = 0
    character_coverage_percentage: float = 0.0
    
    gaps: List[Dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    is_complete: bool = False
    
    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "Coverage Validation Report",
            "=" * 60,
            f"Paragraph Coverage: {self.covered_paragraphs}/{self.total_paragraphs} ({self.coverage_percentage:.1f}%)",
            f"Character Coverage: {self.covered_characters}/{self.total_characters} ({self.character_coverage_percentage:.1f}%)",
        ]
        
        if self.gaps:
            lines.append(f"\nGaps Found: {len(self.gaps)}")
            for gap in self.gaps[:5]:  # Show first 5 gaps
                lines.append(f"  - Paragraphs {gap['start']}-{gap['end']}: {gap['char_count']} chars")
            if len(self.gaps) > 5:
                lines.append(f"  ... and {len(self.gaps) - 5} more gaps")
        
        if self.warnings:
            lines.append(f"\nWarnings: {len(self.warnings)}")
            for warning in self.warnings[:5]:
                lines.append(f"  - {warning}")
            if len(self.warnings) > 5:
                lines.append(f"  ... and {len(self.warnings) - 5} more warnings")
        
        lines.append("")
        lines.append(f"Status: {'✓ COMPLETE' if self.is_complete else '✗ INCOMPLETE'}")
        lines.append("=" * 60)
        
        return "\n".join(lines)


def validate_tree_coverage(
    tree: List[dict],
    source_text: str,
    logger: Optional[logging.Logger] = None
) -> CoverageReport:
    """
    Validate that the tree structure covers all source text.
    
    Analyzes the tree to ensure no paragraphs from the source text are missing.
    This is critical for RAG applications where missing content could lead to
    incomplete or incorrect answers.
    
    Args:
        tree: Root tree structure (list of top-level nodes)
        source_text: Original source text that was processed
        logger: Optional logger instance
        
    Returns:
        CoverageReport with detailed coverage statistics
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    report = CoverageReport()
    
    # Split source into paragraphs
    paragraphs = _split_into_paragraphs(source_text)
    report.total_paragraphs = len(paragraphs)
    report.total_characters = len(source_text)
    
    if not paragraphs:
        report.is_complete = True
        report.coverage_percentage = 100.0
        report.character_coverage_percentage = 100.0
        return report
    
    # Collect all text from leaf nodes
    leaf_texts = []
    _collect_leaf_texts(tree, leaf_texts)
    
    # Track which paragraphs are covered
    covered_indices = set()
    
    for para_idx, paragraph in enumerate(paragraphs):
        # Check if this paragraph appears in any leaf node
        para_normalized = _normalize_text(paragraph)
        
        for leaf_text in leaf_texts:
            leaf_normalized = _normalize_text(leaf_text)
            if para_normalized in leaf_normalized:
                covered_indices.add(para_idx)
                break
    
    # Calculate coverage
    report.covered_paragraphs = len(covered_indices)
    report.uncovered_paragraphs = sorted(set(range(len(paragraphs))) - covered_indices)
    
    if report.total_paragraphs > 0:
        report.coverage_percentage = (report.covered_paragraphs / report.total_paragraphs) * 100
    
    # Calculate character coverage
    covered_chars = sum(len(paragraphs[i]) for i in covered_indices)
    report.covered_characters = covered_chars
    report.uncovered_characters = report.total_characters - covered_chars
    
    if report.total_characters > 0:
        report.character_coverage_percentage = (covered_chars / report.total_characters) * 100
    
    # Identify gaps (consecutive uncovered paragraphs)
    report.gaps = _identify_gaps(report.uncovered_paragraphs, paragraphs)
    
    # Add warnings for significant gaps
    for gap in report.gaps:
        if gap['char_count'] > 500:
            report.warnings.append(
                f"Large gap ({gap['char_count']} chars) at paragraphs {gap['start']}-{gap['end']}"
            )
    
    # Determine if coverage is complete
    report.is_complete = report.coverage_percentage >= 99.0  # Allow 1% tolerance for whitespace differences
    
    logger.info(str(report))
    
    return report


def validate_node_coverage(
    node: dict,
    logger: Optional[logging.Logger] = None
) -> CoverageReport:
    """
    Validate coverage for a single node and its children.
    
    Checks that the node's text is fully covered by its child nodes.
    Useful for validating semantic subdivision results.
    
    Args:
        node: Node dictionary with 'text' and 'nodes' fields
        logger: Optional logger instance
        
    Returns:
        CoverageReport for this node
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    report = CoverageReport()
    
    parent_text = node.get('text', '')
    children = node.get('nodes', [])
    
    if not parent_text:
        report.is_complete = True
        report.coverage_percentage = 100.0
        report.character_coverage_percentage = 100.0
        return report
    
    # Split parent text into paragraphs
    paragraphs = _split_into_paragraphs(parent_text)
    report.total_paragraphs = len(paragraphs)
    report.total_characters = len(parent_text)
    
    if not children:
        # No children means no subdivision - parent text is the only coverage
        report.covered_paragraphs = report.total_paragraphs
        report.covered_characters = report.total_characters
        report.coverage_percentage = 100.0
        report.character_coverage_percentage = 100.0
        report.is_complete = True
        return report
    
    # Check coverage from child nodes using metadata if available
    covered_indices = set()
    
    for child in children:
        metadata = child.get('metadata', {})
        start_para = metadata.get('start_paragraph')
        end_para = metadata.get('end_paragraph')
        
        if start_para is not None and end_para is not None:
            # Use metadata for precise coverage tracking
            for i in range(start_para, end_para + 1):
                if i < len(paragraphs):
                    covered_indices.add(i)
        else:
            # Fallback: check if child text contains paragraphs
            child_text = child.get('text', '')
            child_normalized = _normalize_text(child_text)
            
            for para_idx, paragraph in enumerate(paragraphs):
                para_normalized = _normalize_text(paragraph)
                if para_normalized in child_normalized:
                    covered_indices.add(para_idx)
    
    # Calculate coverage
    report.covered_paragraphs = len(covered_indices)
    report.uncovered_paragraphs = sorted(set(range(len(paragraphs))) - covered_indices)
    
    if report.total_paragraphs > 0:
        report.coverage_percentage = (report.covered_paragraphs / report.total_paragraphs) * 100
    
    # Calculate character coverage
    covered_chars = sum(len(paragraphs[i]) for i in covered_indices)
    report.covered_characters = covered_chars
    report.uncovered_characters = report.total_characters - covered_chars
    
    if report.total_characters > 0:
        report.character_coverage_percentage = (covered_chars / report.total_characters) * 100
    
    # Identify gaps
    report.gaps = _identify_gaps(report.uncovered_paragraphs, paragraphs)
    
    # Add warnings
    for gap in report.gaps:
        if gap['char_count'] > 200:
            report.warnings.append(
                f"Gap in '{node.get('title', 'Unknown')}': paragraphs {gap['start']}-{gap['end']} ({gap['char_count']} chars)"
            )
    
    report.is_complete = report.coverage_percentage >= 99.0
    
    return report


def validate_full_tree_coverage(
    tree: List[dict],
    logger: Optional[logging.Logger] = None
) -> Dict[str, CoverageReport]:
    """
    Validate coverage for all nodes in the tree recursively.
    
    Returns a dictionary mapping node titles to their coverage reports.
    Useful for identifying which specific sections have coverage issues.
    
    Args:
        tree: Root tree structure
        logger: Optional logger instance
        
    Returns:
        Dictionary of {node_title: CoverageReport}
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    reports = {}
    
    def validate_recursive(nodes: List[dict], path: str = ""):
        for node in nodes:
            title = node.get('title', 'Unknown')
            full_path = f"{path}/{title}" if path else title
            
            # Only validate nodes that have semantic_unit children (were subdivided)
            # Skip nodes that only have figure/table/keyword children
            if node.get('nodes'):
                # Count children that could contain text coverage
                # Include sections, semantic_units, and untyped nodes (parsed from markdown)
                # Skip figures, tables, keywords which have specialized content
                semantic_children = [
                    c for c in node['nodes'] 
                    if c.get('node_type') in ('section', 'semantic_unit', None)
                ]
                
                # Skip validation if parent text is very short (likely just a header)
                parent_text = node.get('text', '')
                if len(parent_text.strip()) < 50:
                    continue
                
                if semantic_children:
                    report = validate_node_coverage(node, logger)
                    if not report.is_complete:
                        reports[full_path] = report
                
                # Recurse into children
                validate_recursive(node['nodes'], full_path)
    
    validate_recursive(tree)
    
    # Summary
    if reports:
        logger.warning(f"Found {len(reports)} nodes with incomplete coverage")
        for path, report in reports.items():
            logger.warning(f"  {path}: {report.coverage_percentage:.1f}% coverage")
    else:
        logger.info("All nodes have complete coverage")
    
    return reports


def _split_into_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs."""
    paragraphs = []
    for p in text.split('\n\n'):
        p = p.strip()
        if p:
            paragraphs.append(p)
    
    # If no double-newline paragraphs, try single newlines
    if len(paragraphs) <= 1 and text.strip():
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    
    return paragraphs


def _normalize_text(text: str) -> str:
    """Normalize text for comparison (lowercase, collapse whitespace)."""
    import re
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _collect_leaf_texts(nodes: List[dict], texts: List[str]) -> None:
    """Recursively collect text from leaf nodes."""
    for node in nodes:
        children = node.get('nodes', [])
        if children:
            _collect_leaf_texts(children, texts)
        else:
            # This is a leaf node
            text = node.get('text', '')
            if text:
                texts.append(text)


def _identify_gaps(uncovered_indices: List[int], paragraphs: List[str]) -> List[Dict]:
    """Identify consecutive gaps in coverage."""
    if not uncovered_indices:
        return []
    
    gaps = []
    gap_start = uncovered_indices[0]
    gap_end = uncovered_indices[0]
    
    for idx in uncovered_indices[1:]:
        if idx == gap_end + 1:
            # Continue current gap
            gap_end = idx
        else:
            # End current gap, start new one
            char_count = sum(len(paragraphs[i]) for i in range(gap_start, gap_end + 1) if i < len(paragraphs))
            gaps.append({
                'start': gap_start,
                'end': gap_end,
                'char_count': char_count,
                'paragraph_count': gap_end - gap_start + 1
            })
            gap_start = idx
            gap_end = idx
    
    # Don't forget the last gap
    char_count = sum(len(paragraphs[i]) for i in range(gap_start, gap_end + 1) if i < len(paragraphs))
    gaps.append({
        'start': gap_start,
        'end': gap_end,
        'char_count': char_count,
        'paragraph_count': gap_end - gap_start + 1
    })
    
    return gaps
