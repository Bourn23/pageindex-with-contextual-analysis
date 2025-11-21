"""
Tree integration utilities for granular PageIndex node generation.

This module provides functions to integrate granular nodes (figures, tables,
semantic sub-sections) into the existing PageIndex tree structure.
"""

import logging
from typing import List, Optional, Tuple, Dict
from .semantic_analyzer import SemanticAnalyzer, SemanticUnit
from .figure_detector import FigureDetector, FigureNode
from .table_detector import TableDetector, TableNode


async def apply_semantic_subdivision(
    tree: List[dict],
    page_texts: List[Tuple[str, int]],
    opt,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Apply semantic subdivision to all major sections in the tree.
    
    Traverses the tree recursively and identifies sections that need subdivision.
    For each section, calls SemanticAnalyzer to identify semantic boundaries and
    creates child nodes from semantic units.
    
    Args:
        tree: Root tree structure (list of top-level nodes)
        page_texts: List of (page_text, token_count) tuples
        opt: Configuration options with llm_client and semantic settings
        logger: Optional logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("Starting semantic subdivision")
    logger.info("=" * 60)
    
    # Initialize semantic analyzer
    try:
        analyzer = SemanticAnalyzer(opt.llm_client, logger=logger)
        logger.info("✓ Semantic analyzer initialized")
    except Exception as e:
        logger.error(f"✗ Failed to initialize semantic analyzer: {e}")
        return
    
    # Get minimum pages threshold from config
    min_pages = getattr(opt, 'semantic_min_pages', 0.5)
    
    # Determine max depth based on granularity
    granularity = getattr(opt, 'granularity', 'coarse')
    if granularity == 'fine':
        max_depth = 2  # Recursive subdivision for fine
        logger.info(f"Configuration: granularity=fine, max_depth={max_depth}, min_pages={min_pages}")
    else:
        max_depth = 1  # Single-level subdivision for medium
        logger.info(f"Configuration: granularity={granularity}, max_depth={max_depth}, min_pages={min_pages}")
    
    # Count nodes before subdivision
    def count_nodes(nodes):
        count = len(nodes)
        for node in nodes:
            if 'nodes' in node and node['nodes']:
                count += count_nodes(node['nodes'])
        return count
    
    nodes_before = count_nodes(tree)
    logger.info(f"Tree has {nodes_before} nodes before subdivision")
    
    # Recursively process all nodes
    logger.info("-" * 60)
    logger.info("Processing tree nodes")
    logger.info("-" * 60)
    
    try:
        await _apply_semantic_subdivision_recursive(tree, analyzer, page_texts, min_pages, logger, max_depth)
        
        # Count nodes after subdivision
        nodes_after = count_nodes(tree)
        nodes_added = nodes_after - nodes_before
        
        logger.info("=" * 60)
        logger.info("Semantic Subdivision Summary")
        logger.info("=" * 60)
        logger.info(f"Nodes before: {nodes_before}")
        logger.info(f"Nodes after: {nodes_after}")
        logger.info(f"Nodes added: {nodes_added}")
        logger.info("✓ Semantic subdivision complete")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"✗ Error during semantic subdivision: {e}", exc_info=True)


def _extract_text_for_node(node: dict, page_texts: List[Tuple[str, int]]) -> str:
    """
    Extract text content for a node based on its page range.
    
    Args:
        node: Node dictionary with start_index and end_index
        page_texts: List of (page_text, token_count) tuples
        
    Returns:
        Concatenated text for the node's page range
    """
    start_page = node.get('start_index', 1)
    end_page = node.get('end_index', 1)
    
    # Extract text from pages (convert to 0-indexed)
    texts = []
    for page_num in range(start_page - 1, end_page):
        if page_num < len(page_texts):
            texts.append(page_texts[page_num][0])
    
    return '\n\n'.join(texts)


async def _apply_semantic_subdivision_recursive(
    nodes: List[dict],
    analyzer: SemanticAnalyzer,
    page_texts: List[Tuple[str, int]],
    min_pages: float,
    logger: logging.Logger,
    max_depth: int = 1,
    current_depth: int = 0
) -> None:
    """
    Recursively apply semantic subdivision to nodes with parallel processing.
    
    Args:
        nodes: List of nodes at current level
        analyzer: SemanticAnalyzer instance
        page_texts: List of (page_text, token_count) tuples
        min_pages: Minimum section length for subdivision
        logger: Logger instance
        max_depth: Maximum depth of recursive subdivision (1 = medium, 2+ = fine)
        current_depth: Current depth in recursion (internal use)
    """
    import asyncio
    
    async def process_node(node: dict):
        """Process a single node for semantic subdivision."""
        try:
            # Extract text for this node if not already present
            if 'text' not in node or not node['text']:
                logger.debug(f"Extracting text for node '{node.get('title', 'Unknown')}'")
                node['text'] = _extract_text_for_node(node, page_texts)
            
            # Check if node has text content after extraction
            if not node['text']:
                logger.debug(f"Skipping node '{node.get('title', 'Unknown')}' - no text content after extraction")
                return
            
            # Check if node meets minimum size requirement
            start_page = node.get('start_index', 1)
            end_page = node.get('end_index', 1)
            section_length = end_page - start_page + 1
            
            if section_length < min_pages:
                logger.debug(f"Skipping node '{node.get('title', 'Unknown')}' - too short ({section_length} pages)")
                return
            
            # Analyze section for semantic units (with error handling)
            try:
                logger.info(f"🔍 Analyzing section: '{node.get('title', 'Unknown')}' ({section_length} pages) at depth {current_depth}")
                
                # Run in executor to avoid blocking
                loop = asyncio.get_event_loop()
                semantic_units = await loop.run_in_executor(
                    None,
                    analyzer.analyze_section,
                    node,
                    page_texts,
                    min_pages
                )
                
                logger.info(f"✓ Analysis complete: Found {len(semantic_units)} semantic units")
            except Exception as e:
                logger.error(f"✗ Error analyzing section '{node.get('title', 'Unknown')}': {e}")
                semantic_units = []
            
            if semantic_units:
                logger.info(f"Subdividing '{node.get('title', 'Unknown')}' into {len(semantic_units)} semantic units")
                
                # Create child nodes from semantic units (with error handling)
                try:
                    semantic_nodes = analyzer.create_nodes_from_semantic_units(
                        semantic_units,
                        node,
                        page_texts
                    )
                except Exception as e:
                    logger.error(f"Error creating nodes from semantic units for '{node.get('title', 'Unknown')}': {e}")
                    semantic_nodes = []
                
                if semantic_nodes:
                    # Add semantic nodes as children
                    if 'nodes' not in node:
                        node['nodes'] = []
                    
                    # Insert semantic nodes at the beginning (before any existing children)
                    node['nodes'] = semantic_nodes + node['nodes']
                    
                    logger.debug(f"Added {len(semantic_nodes)} semantic child nodes to '{node.get('title', 'Unknown')}'")
                    
                    # For fine granularity, recursively subdivide the semantic nodes
                    if current_depth < max_depth - 1:
                        logger.debug(f"Recursively subdividing semantic nodes of '{node.get('title', 'Unknown')}' (depth {current_depth + 1}/{max_depth})")
                        await _apply_semantic_subdivision_recursive(
                            semantic_nodes,
                            analyzer,
                            page_texts,
                            min_pages,
                            logger,
                            max_depth,
                            current_depth + 1
                        )
                    
                    # IMPORTANT: Return here to avoid processing children again
                    # The semantic nodes were already processed above
                    return
                    
                else:
                    logger.debug(f"No semantic nodes created for '{node.get('title', 'Unknown')}'")
            else:
                logger.debug(f"No semantic units found for '{node.get('title', 'Unknown')}', keeping original node")
            
            # Only process existing children if we didn't do semantic subdivision
            if 'nodes' in node and node['nodes']:
                await _apply_semantic_subdivision_recursive(
                    node['nodes'],
                    analyzer,
                    page_texts,
                    min_pages,
                    logger,
                    max_depth,
                    current_depth
                )
                
        except Exception as e:
            logger.error(f"Error processing node '{node.get('title', 'Unknown')}': {e}")
    
    # Process all nodes at this level in parallel
    if nodes:
        await asyncio.gather(*[process_node(node) for node in nodes])


async def detect_and_integrate_figures_tables(
    tree: List[dict],
    page_texts: List[Tuple[str, int]],
    doc,
    opt,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Detect figures and tables across all pages and integrate them into the tree.
    
    Batch processes pages for figure/table detection, finds parent sections for
    each detected element based on page range, and inserts them as child nodes.
    
    Uses parallel processing to run figure and table detection concurrently.
    
    Args:
        tree: Root tree structure (list of top-level nodes)
        page_texts: List of (page_text, token_count) tuples
        doc: PDF document object (for FigureDetector and TableDetector)
        opt: Configuration options with llm_client and feature flags
        logger: Optional logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("Starting figure and table detection and integration (parallel)")
    logger.info("=" * 60)
    
    # Check if features are enabled
    enable_figures = getattr(opt, 'enable_figure_detection', True)
    enable_tables = getattr(opt, 'enable_table_detection', True)
    
    logger.info(f"Configuration: figures={enable_figures}, tables={enable_tables}")
    
    if not enable_figures and not enable_tables:
        logger.info("Figure and table detection disabled - skipping")
        return
    
    # Determine page range to process
    total_pages = len(page_texts)
    page_range = (1, total_pages)
    logger.info(f"Processing page range: {page_range[0]} to {page_range[1]} ({total_pages} pages)")
    
    # Run figure and table detection in parallel
    import asyncio
    
    async def detect_figures_async():
        """Async wrapper for figure detection."""
        if not enable_figures:
            logger.info("Figure detection disabled - skipping")
            return []
        
        try:
            logger.info("-" * 60)
            logger.info("Phase 1: Figure Detection (parallel)")
            logger.info("-" * 60)
            
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            figure_detector = FigureDetector(opt.llm_client, doc, logger=logger)
            figures = await loop.run_in_executor(
                None,
                figure_detector.detect_figures,
                page_range,
                page_texts
            )
            logger.info(f"✓ Successfully detected {len(figures)} figures")
            return figures
        except Exception as e:
            logger.error(f"✗ Error in figure detection: {e}", exc_info=True)
            return []
    
    async def detect_tables_async():
        """Async wrapper for table detection."""
        if not enable_tables:
            logger.info("Table detection disabled - skipping")
            return []
        
        try:
            logger.info("-" * 60)
            logger.info("Phase 2: Table Detection (parallel)")
            logger.info("-" * 60)
            
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            table_detector = TableDetector(opt.llm_client, doc, logger=logger)
            tables = await loop.run_in_executor(
                None,
                table_detector.detect_tables,
                page_range,
                page_texts
            )
            logger.info(f"✓ Successfully detected {len(tables)} tables")
            return tables
        except Exception as e:
            logger.error(f"✗ Error in table detection: {e}", exc_info=True)
            return []
    
    # Run both detection tasks in parallel
    logger.info("Running figure and table detection in parallel...")
    figures, tables = await asyncio.gather(
        detect_figures_async(),
        detect_tables_async()
    )
    
    # Initialize counters
    figures_success = 0
    figures_failed = 0
    tables_success = 0
    tables_failed = 0
    
    # Integrate figures into tree
    logger.info("-" * 60)
    logger.info("Phase 3: Figure Integration")
    logger.info("-" * 60)
    
    for figure in figures:
        try:
            figure_node = _create_figure_node(figure)
            success = _insert_node_into_tree(tree, figure_node, logger)
            if success:
                figures_success += 1
                logger.debug(f"✓ Integrated {figure.figure_number} on page {figure.page}")
            else:
                figures_failed += 1
                logger.warning(f"✗ Failed to integrate {figure.figure_number}")
        except Exception as e:
            figures_failed += 1
            logger.error(f"✗ Error integrating figure {figure.figure_number}: {e}")
    
    if figures:
        logger.info(f"Figure integration: {figures_success} successful, {figures_failed} failed")
    
    # Integrate tables into tree
    logger.info("-" * 60)
    logger.info("Phase 4: Table Integration")
    logger.info("-" * 60)
    
    for table in tables:
        try:
            table_node = _create_table_node(table)
            success = _insert_node_into_tree(tree, table_node, logger)
            if success:
                tables_success += 1
                logger.debug(f"✓ Integrated {table.table_number} on page {table.page}")
            else:
                tables_failed += 1
                logger.warning(f"✗ Failed to integrate {table.table_number}")
        except Exception as e:
            tables_failed += 1
            logger.error(f"✗ Error integrating table {table.table_number}: {e}")
    
    if tables:
        logger.info(f"Table integration: {tables_success} successful, {tables_failed} failed")
    
    # Sort children by page number in all nodes
    logger.info("-" * 60)
    logger.info("Phase 5: Sorting nodes by page number")
    logger.info("-" * 60)
    
    try:
        _sort_children_by_page(tree)
        logger.info("✓ Successfully sorted all nodes by page number")
    except Exception as e:
        logger.error(f"✗ Error sorting nodes: {e}")
    
    # Summary
    logger.info("=" * 60)
    logger.info("Figure and Table Integration Summary")
    logger.info("=" * 60)
    logger.info(f"Figures: {len(figures)} detected, {figures_success} integrated, {figures_failed} failed")
    logger.info(f"Tables: {len(tables)} detected, {tables_success} integrated, {tables_failed} failed")
    logger.info(f"Total: {figures_success + tables_success} nodes added to tree")
    logger.info("=" * 60)


def _create_figure_node(figure: FigureNode) -> dict:
    """
    Create a node structure from a FigureNode.
    
    Args:
        figure: FigureNode object
        
    Returns:
        Node dictionary
    """
    node = {
        'title': f"{figure.figure_number}: {figure.caption[:50]}..." if len(figure.caption) > 50 else f"{figure.figure_number}: {figure.caption}",
        'start_index': figure.page,
        'end_index': figure.page,
        'text': f"{figure.caption}\n\n{figure.context}",
        'summary': figure.summary,
        'node_type': 'figure',
        'metadata': {
            'figure_number': figure.figure_number,
            'caption': figure.caption,
            'figure_type': figure.figure_type,
            'bbox': figure.bbox.model_dump() if figure.bbox else None
        },
        'nodes': []
    }
    return node


def _create_table_node(table: TableNode) -> dict:
    """
    Create a node structure from a TableNode.
    
    Args:
        table: TableNode object
        
    Returns:
        Node dictionary
    """
    node = {
        'title': f"{table.table_number}: {table.caption[:50]}..." if len(table.caption) > 50 else f"{table.table_number}: {table.caption}",
        'start_index': table.page,
        'end_index': table.page,
        'text': f"{table.caption}\n\n{table.context}",
        'summary': table.summary,
        'node_type': 'table',
        'metadata': {
            'table_number': table.table_number,
            'caption': table.caption,
            'headers': table.headers,
            'key_values': table.key_values,
            'bbox': table.bbox.model_dump() if table.bbox else None
        },
        'nodes': []
    }
    return node


def insert_node_into_tree(tree: List[dict], node: dict, logger: Optional[logging.Logger] = None) -> bool:
    """
    Insert a node into the tree at the appropriate location based on page range.
    
    Public interface for _insert_node_into_tree.
    
    Args:
        tree: Root tree structure (list of top-level nodes)
        node: Node to insert
        logger: Optional logger instance
        
    Returns:
        True if node was inserted, False otherwise
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    return _insert_node_into_tree(tree, node, logger)


def _insert_node_into_tree(tree: List[dict], node: dict, logger: logging.Logger) -> bool:
    """
    Insert a node into the tree at the appropriate location based on page range.
    
    Finds the appropriate parent section for the node based on its page range
    and inserts it as a child. Handles edge cases where a node might span
    multiple sections. Includes error handling for invalid page ranges.
    
    Args:
        tree: Root tree structure (list of top-level nodes)
        node: Node to insert
        logger: Logger instance
        
    Returns:
        True if node was inserted, False otherwise
    """
    try:
        # Validate node has required fields
        if 'start_index' not in node or 'end_index' not in node:
            logger.warning(f"Node '{node.get('title', 'Unknown')}' missing page indices, cannot insert")
            return False
        
        node_page = node.get('start_index', 1)
        node_end_page = node.get('end_index', node_page)
        
        # Validate page range
        if node_page < 1:
            logger.warning(f"Node '{node.get('title', 'Unknown')}' has invalid start_index {node_page}, adjusting to 1")
            node['start_index'] = 1
            node_page = 1
        
        if node_end_page < node_page:
            logger.warning(f"Node '{node.get('title', 'Unknown')}' has end_index {node_end_page} < start_index {node_page}, adjusting")
            node['end_index'] = node_page
            node_end_page = node_page
        
        # Try to find the best parent section
        try:
            parent = _find_parent_section(tree, node_page, node_end_page)
        except Exception as e:
            logger.error(f"Error finding parent section for node '{node.get('title', 'Unknown')}': {e}")
            parent = None
        
        if parent:
            # Insert node as child of parent
            try:
                if 'nodes' not in parent:
                    parent['nodes'] = []
                
                parent['nodes'].append(node)
                logger.debug(f"Inserted node '{node.get('title', 'Unknown')}' into '{parent.get('title', 'Unknown')}'")
                return True
            except Exception as e:
                logger.error(f"Error inserting node into parent: {e}")
                # Fall through to root level insertion
        
        # No suitable parent found or insertion failed - add to root level
        try:
            tree.append(node)
            logger.debug(f"Inserted node '{node.get('title', 'Unknown')}' at root level")
            return True
        except Exception as e:
            logger.error(f"Error inserting node at root level: {e}")
            return False
            
    except Exception as e:
        logger.error(f"Error in _insert_node_into_tree: {e}")
        return False


def _find_parent_section(nodes: List[dict], start_page: int, end_page: int) -> Optional[dict]:
    """
    Find the most appropriate parent section for a node based on page range.
    
    Searches recursively for the deepest node that fully contains the given
    page range. This ensures that figures/tables are placed in the most
    specific section possible. Includes error handling for malformed nodes.
    
    Args:
        nodes: List of nodes to search
        start_page: Starting page of node to insert
        end_page: Ending page of node to insert
        
    Returns:
        Parent node dictionary, or None if no suitable parent found
    """
    if not nodes:
        return None
    
    best_parent = None
    best_parent_size = float('inf')
    
    for node in nodes:
        try:
            # Validate node has page indices
            if 'start_index' not in node or 'end_index' not in node:
                continue
            
            node_start = node.get('start_index', 1)
            node_end = node.get('end_index', 1)
            
            # Validate page range
            if node_start > node_end:
                continue
            
            # Check if this node fully contains the target page range
            if node_start <= start_page and node_end >= end_page:
                node_size = node_end - node_start + 1
                
                # Check if this is a better (smaller) parent than what we've found
                if node_size < best_parent_size:
                    best_parent = node
                    best_parent_size = node_size
                
                # Recursively search children for an even better parent
                if 'nodes' in node and node['nodes']:
                    try:
                        child_parent = _find_parent_section(node['nodes'], start_page, end_page)
                        if child_parent:
                            child_start = child_parent.get('start_index', 1)
                            child_end = child_parent.get('end_index', 1)
                            child_size = child_end - child_start + 1
                            if child_size < best_parent_size:
                                best_parent = child_parent
                                best_parent_size = child_size
                    except Exception:
                        # If recursion fails, continue with current best parent
                        pass
        except Exception:
            # Skip malformed nodes
            continue
    
    return best_parent


def _sort_children_by_page(nodes: List[dict]) -> None:
    """
    Sort children by page number recursively throughout the tree with error handling.
    
    Ensures that all child nodes are ordered by their start_index (page number)
    for consistent navigation and display. Handles nodes with missing or invalid indices.
    
    Args:
        nodes: List of nodes to sort
    """
    if not nodes:
        return
    
    for node in nodes:
        try:
            if 'nodes' in node and node['nodes']:
                # Sort children by start_index (handle missing indices gracefully)
                try:
                    node['nodes'].sort(key=lambda n: n.get('start_index', float('inf')))
                except Exception as e:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Error sorting children of node '{node.get('title', 'Unknown')}': {e}")
                
                # Recursively sort grandchildren
                try:
                    _sort_children_by_page(node['nodes'])
                except Exception as e:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.error(f"Error recursively sorting children of node '{node.get('title', 'Unknown')}': {e}")
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error processing node in _sort_children_by_page: {e}")


def detect_circular_references(tree: List[dict], logger: Optional[logging.Logger] = None) -> bool:
    """
    Detect circular references in the tree structure.
    
    Args:
        tree: Root tree structure (list of top-level nodes)
        logger: Optional logger instance
        
    Returns:
        True if circular references detected, False otherwise
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    visited = set()
    
    def _check_node(node: dict, path: List[str]) -> bool:
        """Check a single node for circular references."""
        try:
            node_id = id(node)  # Use object ID for tracking
            node_title = node.get('title', 'Unknown')
            
            if node_id in visited:
                logger.error(f"Circular reference detected: {' -> '.join(path)} -> {node_title}")
                return True
            
            visited.add(node_id)
            path.append(node_title)
            
            # Check children
            if 'nodes' in node and node['nodes']:
                for child in node['nodes']:
                    if _check_node(child, path.copy()):
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking node for circular references: {e}")
            return False
    
    # Check all root nodes
    for node in tree:
        if _check_node(node, []):
            return True
    
    return False


def reassign_hierarchical_node_ids(tree: List[dict]) -> None:
    """
    Reassign hierarchical node IDs throughout the tree.
    
    Traverses the tree depth-first and assigns IDs in the format "0001.0002.0003"
    where each level is represented by a 4-digit number. This creates a clear
    hierarchical structure that reflects the tree organization.
    
    This function updates the existing write_node_id() behavior to use
    hierarchical IDs instead of sequential IDs.
    
    Args:
        tree: Root tree structure (list of top-level nodes)
    """
    _reassign_node_ids_recursive(tree, prefix="")


def _reassign_node_ids_recursive(nodes: List[dict], prefix: str, used_ids: Optional[set] = None) -> None:
    """
    Recursively assign hierarchical node IDs with collision detection.
    
    Args:
        nodes: List of nodes at current level
        prefix: ID prefix from parent levels (e.g., "0001.0002")
        used_ids: Set of already used node IDs for collision detection
    """
    if used_ids is None:
        used_ids = set()
    
    for idx, node in enumerate(nodes, start=1):
        try:
            # Create ID for this node
            node_num = str(idx).zfill(4)
            
            if prefix:
                node_id = f"{prefix}.{node_num}"
            else:
                node_id = node_num
            
            # Check for ID collision
            if node_id in used_ids:
                # This should not happen with proper indexing, but handle it
                collision_count = 1
                while f"{node_id}_dup{collision_count}" in used_ids:
                    collision_count += 1
                node_id = f"{node_id}_dup{collision_count}"
            
            # Assign ID to node
            node['node_id'] = node_id
            used_ids.add(node_id)
            
            # Recursively process children
            if 'nodes' in node and node['nodes']:
                try:
                    _reassign_node_ids_recursive(node['nodes'], node_id, used_ids)
                except Exception as e:
                    # Log error but continue with other nodes
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.error(f"Error assigning IDs to children of node '{node.get('title', 'Unknown')}': {e}")
                    
        except Exception as e:
            # Log error but continue with other nodes
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error assigning ID to node at index {idx}: {e}")
