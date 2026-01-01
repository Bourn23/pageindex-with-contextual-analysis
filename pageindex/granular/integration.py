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
    logger: Optional[logging.Logger] = None,
    fill_gaps: bool = True
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
        fill_gaps: If True, create nodes for uncovered paragraphs (default: True)
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
    if granularity == 'keywords':
        max_depth = 3  # Deepest level: sections -> semantic units -> keywords
        logger.info(f"Configuration: granularity=keywords, max_depth={max_depth}, min_pages={min_pages}")
    elif granularity == 'fine':
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
        await _apply_semantic_subdivision_recursive(tree, analyzer, page_texts, min_pages, logger, max_depth, fill_gaps=fill_gaps)
        
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
    current_depth: int = 0,
    fill_gaps: bool = True
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
        fill_gaps: If True, create nodes for uncovered paragraphs (default: True)
    """
    import asyncio
    
    async def process_node(node: dict):
        """Process a single node for semantic subdivision."""
        try:
            node_title = node.get('title', 'Unknown')
            node_type = node.get('node_type', 'unknown')
            has_text = bool(node.get('text'))
            text_len = len(node.get('text', ''))
            
            # Skip nodes with locked text (keywords)
            if node.get('_text_locked'):
                logger.debug(f"Skipping text-locked node '{node_title}'")
                return
            
            # Skip text extraction for keyword nodes - they have their own text
            if node_type == 'keyword':
                logger.debug(f"Skipping keyword node '{node_title}'")
                return
            
            # Skip text extraction for semantic_unit nodes that already have text
            # (they were created with paragraph-specific text)
            if node_type == 'semantic_unit' and has_text:
                logger.debug(f"Semantic unit '{node_title}' already has text ({text_len} chars), skipping extraction")
                # Don't return - we still need to process children
            elif not has_text:
                # Extract text for this node if not already present
                logger.debug(f"Extracting text for '{node_title}'")
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
                    min_pages,
                    1
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
                        page_texts,
                        fill_gaps=fill_gaps
                    )
                except Exception as e:
                    logger.error(f"Error creating nodes from semantic units for '{node.get('title', 'Unknown')}': {e}")
                    semantic_nodes = []
                
                if semantic_nodes:
                    # Check if any semantic node has substantial content
                    # If all nodes are small (gap nodes), extract keywords from parent instead
                    substantial_nodes = [n for n in semantic_nodes if len(n.get('text', '')) >= 100]
                    
                    if not substantial_nodes and max_depth == 3 and len(node.get('text', '')) >= 100:
                        # All child nodes are small gaps - extract keywords from parent
                        logger.info(f"All semantic children are small gaps - extracting keywords from parent '{node_title}'")
                        await _extract_keywords_for_nodes([node], analyzer, logger)
                        return
                    
                    # Add semantic nodes as children
                    if 'nodes' not in node:
                        node['nodes'] = []
                    
                    # Insert semantic nodes at the beginning (before any existing children)
                    node['nodes'] = semantic_nodes + node['nodes']
                    
                    logger.debug(f"Added {len(semantic_nodes)} semantic child nodes to '{node.get('title', 'Unknown')}'")
                    
                    # For fine/keywords granularity, recursively subdivide the semantic nodes
                    # This continues until we reach max_depth - 1 (the level before keywords)
                    if current_depth < max_depth - 1:
                        logger.debug(f"Recursively subdividing semantic nodes of '{node.get('title', 'Unknown')}' (depth {current_depth + 1}/{max_depth})")
                        await _apply_semantic_subdivision_recursive(
                            semantic_nodes,
                            analyzer,
                            page_texts,
                            min_pages,
                            logger,
                            max_depth,
                            current_depth + 1,
                            fill_gaps=fill_gaps
                        )
                        
                        # After recursive subdivision, if we're at the keywords level,
                        # extract keywords from the deepest semantic nodes
                        if max_depth == 3 and current_depth + 1 == max_depth - 1:
                            logger.info(f"Extracting keywords from fine-grained semantic units")
                            await _extract_keywords_from_deepest_nodes(semantic_nodes, analyzer, logger)
                    
                    # IMPORTANT: Return here to avoid processing children again
                    # The semantic nodes were already processed above
                    return
                    
                else:
                    logger.debug(f"No semantic nodes created for '{node.get('title', 'Unknown')}' (leaf node - no meaningful subdivision)")
                    # This is a leaf node - extract keywords if in keywords mode
                    # This happens when all semantic units were skipped (95%+ of parent text)
                    if max_depth == 3 and len(node.get('text', '')) >= 100:
                        logger.info(f"Leaf node (no further subdivision): Extracting keywords from '{node_title}'")
                        await _extract_keywords_for_nodes([node], analyzer, logger)
                        return  # Don't process children - we extracted keywords from this node
            else:
                logger.debug(f"No semantic units found for '{node.get('title', 'Unknown')}', keeping original node")

                # Only extract keywords if this is truly a leaf (no children) and has substantial text
                has_children = 'nodes' in node and node['nodes']
                if max_depth == 3 and len(node.get('text', '')) >= 100 and not has_children:
                    logger.info(f"Leaf node reached for {node_title}: Extracting keywords directly")
                    await _extract_keywords_for_nodes([node], analyzer, logger)
                    return  # Don't process children (there are none)
            
            # Only process existing children if we didn't do semantic subdivision
            if 'nodes' in node and node['nodes']:
                await _apply_semantic_subdivision_recursive(
                    node['nodes'],
                    analyzer,
                    page_texts,
                    min_pages,
                    logger,
                    max_depth,
                    current_depth,
                    fill_gaps=fill_gaps
                )
                
        except Exception as e:
            logger.error(f"Error processing node '{node.get('title', 'Unknown')}': {e}")
    
    # Process all nodes at this level in parallel
    if nodes:
        await asyncio.gather(*[process_node(node) for node in nodes])


async def _extract_keywords_for_nodes(
    nodes: List[dict],
    analyzer: SemanticAnalyzer,
    logger: logging.Logger
) -> None:
    """
    Extract keywords for a list of nodes (typically semantic units).
    
    Skips nodes that are too short (e.g., header-only nodes) to avoid extracting
    meaningless keywords from minimal text.
    
    Args:
        nodes: List of nodes to extract keywords from
        analyzer: SemanticAnalyzer instance
        logger: Logger instance
    """
    import asyncio
    
    # Minimum text length to extract keywords (skip header-only nodes)
    MIN_TEXT_LENGTH_FOR_KEYWORDS = 100
    
    async def extract_for_node(node: dict):
        """Extract keywords for a single node."""
        try:
            node_title = node.get('title', 'Unknown')
            node_text = node.get('text', '')
            
            # Skip nodes with very short text (likely just headers)
            if len(node_text.strip()) < MIN_TEXT_LENGTH_FOR_KEYWORDS:
                logger.debug(f"Skipping keyword extraction for '{node_title}' - text too short ({len(node_text)} chars)")
                return
            
            logger.info(f"🔑 Extracting keywords from: '{node_title}'")
            
            # Run keyword extraction in executor
            loop = asyncio.get_event_loop()
            keywords = await loop.run_in_executor(
                None,
                analyzer.extract_keywords,
                node
            )
            
            if keywords:
                logger.info(f"✓ Extracted {len(keywords)} keywords from '{node_title}'")
                
                # Create keyword nodes
                keyword_nodes = analyzer.create_keyword_nodes(keywords, node)
                
                if keyword_nodes:
                    # Add keyword nodes as children
                    if 'nodes' not in node:
                        node['nodes'] = []
                    
                    node['nodes'].extend(keyword_nodes)
                    logger.debug(f"Added {len(keyword_nodes)} keyword nodes to '{node_title}'")
            else:
                logger.debug(f"No keywords extracted from '{node_title}'")
                
        except Exception as e:
            logger.error(f"Error extracting keywords from '{node.get('title', 'Unknown')}': {e}")
    
    # Process all nodes in parallel
    if nodes:
        await asyncio.gather(*[extract_for_node(node) for node in nodes])

# async def _apply_semantic_subdivision_recursive(
#     nodes: List[dict],
#     analyzer: SemanticAnalyzer,
#     page_texts: List[Tuple[str, int]],
#     min_pages: float,
#     logger: logging.Logger,
#     max_depth: int = 1,
#     current_depth: int = 0,
#     fill_gaps: bool = True
# ) -> None:
#     """
#     Recursively apply semantic subdivision to nodes with parallel processing.
    
#     Args:
#         nodes: List of nodes at current level
#         analyzer: SemanticAnalyzer instance
#         page_texts: List of (page_text, token_count) tuples
#         min_pages: Minimum section length for subdivision
#         logger: Logger instance
#         max_depth: Maximum depth of recursive subdivision (1 = medium, 2+ = fine)
#         current_depth: Current depth in recursion (internal use)
#         fill_gaps: If True, create nodes for uncovered paragraphs (default: True)
#     """
#     import asyncio
    
#     async def process_node(node: dict):
#         """Process a single node for semantic subdivision."""
#         try:
#             node_title = node.get('title', 'Unknown')
#             node_type = node.get('node_type', 'unknown')
#             has_text = bool(node.get('text'))
#             text_len = len(node.get('text', ''))
            
#             # Skip nodes with locked text (keywords)
#             if node.get('_text_locked'):
#                 logger.debug(f"Skipping text-locked node '{node_title}'")
#                 return
            
#             # Skip text extraction for keyword nodes - they have their own text
#             if node_type == 'keyword':
#                 logger.debug(f"Skipping keyword node '{node_title}'")
#                 return
            
#             # 1. Extract text if missing
#             if not node.get('text'):
#                 node['text'] = _extract_text_for_node(node, page_texts)
            
#             if not node['text']:
#                 return

#             # 2. Dynamic Threshold:
#             # If we are deeper in the tree, we must allow smaller chunks to be processed.
#             # If current_depth > 0, we effectively ignore min_pages to allow paragraph splitting.
#             current_min_pages = min_pages if current_depth == 0 else 0
            
#             # 3. Attempt Subdivision
#             # We try to break it down. If the LLM returns nothing, we assume it is Atomic.
#             semantic_units = []
#             try:
#                 # Run in executor
#                 loop = asyncio.get_event_loop()
#                 semantic_units = await loop.run_in_executor(
#                     None,
#                     analyzer.analyze_section,
#                     node,
#                     page_texts,
#                     current_min_pages # Use dynamic threshold
#                 )
#             except Exception as e:
#                 logger.error(f"Analysis failed for '{node_title}': {e}")

#             # ---------------------------------------------------------
#             # PATH A: The Node is divisible (It is a Parent)
#             # ---------------------------------------------------------
#             if semantic_units:
#                 logger.info(f"Subdividing '{node_title}' into {len(semantic_units)} units")
                
#                 # Create the child nodes
#                 semantic_nodes = analyzer.create_nodes_from_semantic_units(
#                     semantic_units, node, page_texts, fill_gaps=fill_gaps
#                 )
                
#                 if semantic_nodes:
#                     if 'nodes' not in node: node['nodes'] = []
#                     # Prepend new semantic nodes
#                     node['nodes'] = semantic_nodes + node['nodes']
                    
#                     # RECURSE: Keep breaking down the children
#                     # We only stop if we hit max_depth (safety valve)
#                     if current_depth < max_depth:
#                         await _apply_semantic_subdivision_recursive(
#                             semantic_nodes, analyzer, page_texts, min_pages,
#                             logger, max_depth, current_depth + 1, fill_gaps
#                         )
#                     return # Done with this node, children are handling the work now

#             # ---------------------------------------------------------
#             # PATH B: The Node is Atomic (It is a Leaf)
#             # ---------------------------------------------------------
#             # If we are here, either:
#             # 1. semantic_units was empty (LLM couldn't split it further)
#             # 2. OR we hit max_depth limit
#             # 3. OR semantic_nodes creation failed
            
#             # If granularity is 'keywords', we extract from this atomic part.
#             # We only do this if it's a semantic_unit (clean text) or a leaf section.
#             if max_depth >= 2: # 'fine' or 'keywords' mode implies deep traversal
#                  # Only extract if it's a semantic unit we created OR it's a leaf section that refused to split
#                 should_extract = (node.get('node_type') == 'semantic_unit') or (not semantic_units)
                
#                 if should_extract:
#                     # Check if we should extract keywords (based on global opt)
#                     # You might need to pass 'opt.granularity' into this function or infer from max_depth
#                     # Assuming max_depth=3 implies keywords mode as per your previous code:
#                     if max_depth == 3: 
#                         logger.info(f"🍃 Atomic Node Reached: '{node_title}'. Extracting keywords.")
#                         await _extract_keywords_for_nodes([node], analyzer, logger)

#         except Exception as e:
#             logger.error(f"Error processing node '{node.get('title', 'Unknown')}': {e}")

#     if nodes:
#         await asyncio.gather(*[process_node(node) for node in nodes])


# async def _apply_semantic_subdivision_recursive(
#     nodes: List[dict],
#     analyzer: SemanticAnalyzer,
#     page_texts: List[Tuple[str, int]],
#     min_pages: float,
#     logger: logging.Logger,
#     max_depth: int = 1,
#     current_depth: int = 0,
#     fill_gaps: bool = True
# ) -> None:
#     """
#     Recursively apply semantic subdivision to nodes with parallel processing.
#     """
#     import asyncio
    
#     async def process_node(node: dict):
#         """Process a single node for semantic subdivision."""
#         try:
#             node_title = node.get('title', 'Unknown')
#             node_type = node.get('node_type', 'unknown')
#             has_text = bool(node.get('text'))
#             text_len = len(node.get('text', ''))
            
#             # Skip nodes with locked text (keywords)
#             if node.get('_text_locked'):
#                 logger.debug(f"Skipping text-locked node '{node_title}'")
#                 return
            
#             # Skip text extraction for keyword nodes - they have their own text
#             if node_type == 'keyword':
#                 logger.debug(f"Skipping keyword node '{node_title}'")
#                 return
            
#             # Skip text extraction for semantic_unit nodes that already have text
#             if node_type == 'semantic_unit' and has_text:
#                 logger.debug(f"Semantic unit '{node_title}' already has text ({text_len} chars), skipping extraction")
#             elif not has_text:
#                 # Extract text for this node if not already present
#                 logger.debug(f"Extracting text for '{node_title}'")
#                 node['text'] = _extract_text_for_node(node, page_texts)

#             # Check if node has text content after extraction
#             if not node.get('text'):
#                 logger.debug(f"Skipping node '{node_title}' - no text content after extraction")
#                 return
            
#             # Check if node meets minimum size requirement
#             start_page = node.get('start_index', 1)
#             end_page = node.get('end_index', 1)
#             section_length = end_page - start_page + 1
            
#             if section_length < min_pages:
#                 logger.debug(f"Skipping node '{node_title}' - too short ({section_length} pages)")
#                 # If it's too short to analyze, it might still be a valid atomic unit for keywords
#                 # But typically we skip very small artifacts. 
#                 # If you want to force keywords on short text, you would add a check here.
#                 return
            
#             # Analyze section for semantic units
#             try:
#                 logger.info(f"🔍 Analyzing section: '{node_title}' at depth {current_depth}")
                
#                 loop = asyncio.get_event_loop()
#                 semantic_units = await loop.run_in_executor(
#                     None,
#                     analyzer.analyze_section,
#                     node,
#                     page_texts,
#                     min_pages
#                 )
                
#                 logger.info(f"✓ Analysis complete: Found {len(semantic_units)} semantic units")
#             except Exception as e:
#                 logger.error(f"✗ Error analyzing section '{node_title}': {e}")
#                 semantic_units = []
            
#             # --- BRANCH A: Successfully Subdivided ---
#             if semantic_units:
#                 logger.info(f"Subdividing '{node_title}' into {len(semantic_units)} semantic units")
                
#                 try:
#                     semantic_nodes = analyzer.create_nodes_from_semantic_units(
#                         semantic_units,
#                         node,
#                         page_texts,
#                         fill_gaps=fill_gaps
#                     )
#                 except Exception as e:
#                     logger.error(f"Error creating nodes from semantic units for '{node_title}': {e}")
#                     semantic_nodes = []
                
#                 if semantic_nodes:
#                     # Add semantic nodes as children
#                     if 'nodes' not in node:
#                         node['nodes'] = []
                    
#                     # Insert semantic nodes at the beginning
#                     node['nodes'] = semantic_nodes + node['nodes']
                    
#                     logger.debug(f"Added {len(semantic_nodes)} semantic child nodes to '{node_title}'")
                    
#                     # RECURSION LOGIC UPDATE:
#                     # If we are in 'keywords' mode (max_depth >= 3), we keep recursing until "atomic" (no semantic units found).
#                     # We use a safety limit (5) to prevent infinite loops.
#                     # If in 'medium/fine' mode, we respect the strict max_depth.
                    
#                     should_recurse = False
#                     if max_depth >= 3:
#                         should_recurse = (current_depth < 5) # Safety cap
#                     else:
#                         should_recurse = (current_depth < max_depth)
                        
#                     if should_recurse:
#                         logger.debug(f"Recursively subdividing semantic nodes of '{node_title}' (depth {current_depth + 1})")
#                         await _apply_semantic_subdivision_recursive(
#                             semantic_nodes,
#                             analyzer,
#                             page_texts,
#                             min_pages,
#                             logger,
#                             max_depth,
#                             current_depth + 1,
#                             fill_gaps=fill_gaps
#                         )
                    
#                     # Return here to avoid processing children in the generic block below
#                     return
                    
#                 else:
#                     logger.debug(f"No semantic nodes created for '{node_title}'")

#             # --- BRANCH B: Subdivision Failed / "Atomic" ---
#             else:
#                 logger.debug(f"No semantic units found for '{node_title}' (Atomic)")
                
#                 # KEYWORD EXTRACTION UPDATE:
#                 # We are at a leaf (Atomic part). If we are in 'keywords' mode, 
#                 # this is the exact moment to extract keywords.
#                 if max_depth >= 3:
#                      logger.info(f"🍃 Atomic leaf reached: Extracting keywords from '{node_title}'")
#                      await _extract_keywords_for_nodes([node], analyzer, logger)
            
#             # Only process existing children if we didn't do semantic subdivision
#             # (This handles pre-existing structure nodes, like subsection headers in the original PDF/MD)
#             if 'nodes' in node and node['nodes']:
#                 await _apply_semantic_subdivision_recursive(
#                     node['nodes'],
#                     analyzer,
#                     page_texts,
#                     min_pages,
#                     logger,
#                     max_depth,
#                     current_depth,
#                     fill_gaps=fill_gaps
#                 )
                
#         except Exception as e:
#             logger.error(f"Error processing node '{node.get('title', 'Unknown')}': {e}")
    
#     # Process all nodes at this level in parallel
#     if nodes:
#         await asyncio.gather(*[process_node(node) for node in nodes])


async def _extract_keywords_from_deepest_nodes(
    nodes: List[dict],
    analyzer: SemanticAnalyzer,
    logger: logging.Logger
) -> None:
    """
    Recursively find the deepest (leaf) semantic nodes and extract keywords from them.
    
    This ensures keywords are extracted from fine-grained semantic units, not their parents.
    Skips nodes that are too short (e.g., header-only nodes) to avoid extracting
    meaningless keywords from minimal text.
    
    Args:
        nodes: List of nodes to search
        analyzer: SemanticAnalyzer instance
        logger: Logger instance
    """
    import asyncio
    
    # Minimum text length to extract keywords (skip header-only nodes)
    MIN_TEXT_LENGTH_FOR_KEYWORDS = 100
    
    async def process_node(node: dict):
        """Process a single node - either extract keywords or recurse to children."""
        try:
            # Check if this node has semantic unit children
            has_semantic_children = False
            if 'nodes' in node and node['nodes']:
                for child in node['nodes']:
                    if child.get('node_type') == 'semantic_unit':
                        has_semantic_children = True
                        break
            
            if has_semantic_children:
                # This node has semantic children, so recurse deeper
                logger.debug(f"Node '{node.get('title', 'Unknown')}' has semantic children, recursing...")
                await _extract_keywords_from_deepest_nodes(node['nodes'], analyzer, logger)
            else:
                # This is a leaf semantic node (no semantic children), extract keywords here
                if node.get('node_type') == 'semantic_unit':
                    node_title = node.get('title', 'Unknown')
                    node_text = node.get('text', '')
                    
                    # Skip nodes with very short text (likely just headers)
                    if len(node_text.strip()) < MIN_TEXT_LENGTH_FOR_KEYWORDS:
                        logger.debug(f"Skipping keyword extraction for '{node_title}' - text too short ({len(node_text)} chars)")
                        return
                    
                    logger.info(f"🔑 Extracting keywords from leaf node: '{node_title}'")
                    
                    # Run keyword extraction in executor
                    loop = asyncio.get_event_loop()
                    keywords = await loop.run_in_executor(
                        None,
                        analyzer.extract_keywords,
                        node
                    )
                    
                    if keywords:
                        logger.info(f"✓ Extracted {len(keywords)} keywords from '{node_title}'")
                        
                        # Create keyword nodes
                        keyword_nodes = analyzer.create_keyword_nodes(keywords, node)
                        
                        if keyword_nodes:
                            # Add keyword nodes as children
                            if 'nodes' not in node:
                                node['nodes'] = []
                            
                            node['nodes'].extend(keyword_nodes)
                            logger.debug(f"Added {len(keyword_nodes)} keyword nodes to '{node_title}'")
                    else:
                        logger.debug(f"No keywords extracted from '{node_title}'")
                        
        except Exception as e:
            logger.error(f"Error processing node '{node.get('title', 'Unknown')}': {e}")
    
    # Process all nodes in parallel
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
