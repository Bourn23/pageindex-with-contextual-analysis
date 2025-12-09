"""
Markdown processing for PageIndex - mirrors the PDF pipeline.

This module processes markdown files using the same pipeline as PDFs:
1. Parse markdown to extract "page_list" equivalent (text chunks with headers)
2. Build tree structure from headers (like TOC extraction for PDFs)
3. Apply granular features (semantic subdivision, keywords, figures, tables)
4. Add text to nodes
"""

import asyncio
import json
import re
import os
import logging
from typing import List, Tuple, Dict, Optional
from .utils import (
    ConfigLoader, JsonLogger, write_node_id, count_tokens,
    post_processing, list_to_tree
)


def parse_markdown_to_page_list(md_path: str, model: str = None) -> Tuple[List[Tuple[str, int]], List[dict], List[str]]:
    """
    Parse markdown file into a page_list-like structure.
    
    For markdown, we treat each major section as a "page" for compatibility
    with the PDF pipeline. This allows reuse of the same granular processing.
    
    Args:
        md_path: Path to markdown file
        model: Model name for token counting
        
    Returns:
        Tuple of:
        - page_list: List of (text, token_count) tuples (one per section or chunk)
        - toc_list: Flat list of TOC entries with structure indices
        - lines: Original markdown lines
    """
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    # Extract headers and their positions
    headers = []
    in_code_block = False
    
    for line_num, line in enumerate(lines):
        stripped = line.strip()
        
        # Track code blocks
        if stripped.startswith('```'):
            in_code_block = not in_code_block
            continue
        
        if in_code_block:
            continue
        
        # Match headers
        match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            headers.append({
                'level': level,
                'title': title,
                'line_num': line_num,
            })
    
    if not headers:
        # No headers found - treat entire document as one section
        text = content
        token_count = count_tokens(text, model)
        return [(text, token_count)], [], lines
    
    # Build TOC list with structure indices (like "1", "1.1", "1.2", "2", etc.)
    toc_list = []
    level_counters = [0] * 7  # Support up to 6 levels
    
    for header in headers:
        level = header['level']
        
        # Increment counter at this level
        level_counters[level] += 1
        
        # Reset all deeper level counters
        for i in range(level + 1, 7):
            level_counters[i] = 0
        
        # Build structure string
        structure_parts = []
        for i in range(1, level + 1):
            if level_counters[i] > 0:
                structure_parts.append(str(level_counters[i]))
        
        structure = '.'.join(structure_parts) if structure_parts else '1'
        
        toc_list.append({
            'structure': structure,
            'title': header['title'],
            'line_num': header['line_num'],
            'level': level,
        })
    
    # Extract text for each section
    for i, entry in enumerate(toc_list):
        start_line = entry['line_num']
        if i + 1 < len(toc_list):
            end_line = toc_list[i + 1]['line_num']
        else:
            end_line = len(lines)
        
        section_text = '\n'.join(lines[start_line:end_line]).strip()
        entry['text'] = section_text
        entry['token_count'] = count_tokens(section_text, model)
    
    # Create page_list - for markdown, we use the full document as one "page"
    # but keep section boundaries for text extraction
    full_text = content
    full_token_count = count_tokens(full_text, model)
    page_list = [(full_text, full_token_count)]
    
    return page_list, toc_list, lines


def build_tree_from_toc(toc_list: List[dict], total_lines: int) -> List[dict]:
    """
    Build hierarchical tree from flat TOC list.
    
    This mirrors the PDF's post_processing + list_to_tree flow.
    
    Args:
        toc_list: Flat list of TOC entries with structure indices
        total_lines: Total number of lines in the document
        
    Returns:
        Hierarchical tree structure
    """
    if not toc_list:
        return []
    
    # Add start_index and end_index (using line numbers)
    for i, entry in enumerate(toc_list):
        entry['start_index'] = entry['line_num'] + 1  # 1-indexed like PDF pages
        entry['physical_index'] = entry['line_num'] + 1
        
        if i + 1 < len(toc_list):
            entry['end_index'] = toc_list[i + 1]['line_num']
        else:
            entry['end_index'] = total_lines
        
        # Mark as appearing at start (for compatibility with PDF pipeline)
        entry['appear_start'] = 'yes'
    
    # Use the same list_to_tree function as PDFs
    tree = list_to_tree(toc_list)
    
    # Add text to each node
    def add_text_to_tree(nodes, toc_map):
        for node in nodes:
            # Find matching TOC entry
            for entry in toc_list:
                if entry['title'] == node['title']:
                    node['text'] = entry.get('text', '')
                    node['line_num'] = entry.get('line_num', 0)
                    break
            
            if 'nodes' in node and node['nodes']:
                add_text_to_tree(node['nodes'], toc_map)
    
    add_text_to_tree(tree, {e['title']: e for e in toc_list})
    
    return tree


async def apply_granular_features(
    tree: List[dict],
    page_list: List[Tuple[str, int]],
    opt,
    logger: logging.Logger
) -> None:
    """
    Apply granular features to the tree structure.
    
    For markdown, we apply semantic subdivision directly since nodes already have text.
    This mirrors the PDF pipeline's recursive subdivision approach.
    """
    from .granular.semantic_analyzer import SemanticAnalyzer
    from .llm_client import get_llm_client
    
    # Get LLM client
    llm_client = get_llm_client()
    opt.llm_client = llm_client
    
    if not opt.enable_semantic_subdivision:
        return
    
    logger.info("Applying semantic subdivision for markdown...")
    
    analyzer = SemanticAnalyzer(llm_client, logger=logger)
    
    # Determine max depth based on granularity (match PDF pipeline)
    granularity = getattr(opt, 'granularity', 'coarse')
    if granularity == 'keywords':
        max_depth = 3  # Match PDF: sections -> semantic units -> finer units -> keywords
    elif granularity == 'fine':
        max_depth = 2
    else:
        max_depth = 1
    
    # Minimum text length for subdivision (shorter = more subdivision)
    min_text_length = 200 if granularity == 'keywords' else 300
    
    # Track processed nodes to avoid infinite loops
    processed_ids = set()
    
    async def subdivide_node(node: dict, depth: int = 0):
        """Recursively subdivide a node into semantic units."""
        
        node_id = id(node)
        if node_id in processed_ids:
            return
        processed_ids.add(node_id)
        
        # Skip keyword nodes
        if node.get('node_type') == 'keyword':
            return
        
        # Get node text
        text = node.get('text', '')
        title = node.get('title', 'Unknown')[:50]
        
        # Skip short sections but still process children
        if not text or len(text) < min_text_length:
            if 'nodes' in node and node['nodes']:
                for child in node['nodes']:
                    await subdivide_node(child, depth)
            return
        
        # Check if this node has section children (not semantic units)
        children = node.get('nodes', [])
        section_children = [c for c in children if c.get('node_type', 'section') == 'section']
        
        # If has section children, recurse into them instead of subdividing this node
        if section_children:
            for child in children:
                await subdivide_node(child, depth)
            return
        
        # Check depth limit
        if depth >= max_depth:
            # Still process existing children at max depth
            for child in children:
                await subdivide_node(child, depth)
            return
        
        # Check if already has semantic_unit children (already subdivided)
        semantic_children = [c for c in children if c.get('node_type') == 'semantic_unit']
        if semantic_children:
            # Already subdivided - recurse into semantic children for further subdivision
            for child in semantic_children:
                await subdivide_node(child, depth + 1)
            return
        
        # Create section node for analyzer
        section_node = {
            'title': node.get('title', ''),
            'text': text,
            'start_index': node.get('start_index', 1),
            'end_index': node.get('end_index', 1)
        }
        
        # Create pseudo page_texts from node text
        node_page_texts = [(text, count_tokens(text, opt.model))]
        
        try:
            logger.info(f"[Depth {depth}] Analyzing: '{title}' ({len(text)} chars)")
            
            # Analyze section for semantic units
            semantic_units = analyzer.analyze_section(
                section_node,
                page_texts=node_page_texts,
                min_pages=0,  # No page minimum for markdown
                min_paragraphs=2  # Allow finer subdivision
            )
            
            if semantic_units:
                logger.info(f"  → Found {len(semantic_units)} semantic units")
                
                # Create child nodes from semantic units
                semantic_nodes = analyzer.create_nodes_from_semantic_units(
                    semantic_units,
                    section_node,
                    page_texts=node_page_texts,
                    fill_gaps=True
                )
                
                if semantic_nodes:
                    # Add semantic nodes as children
                    if 'nodes' not in node:
                        node['nodes'] = []
                    node['nodes'].extend(semantic_nodes)
                    
                    # Recursively subdivide the new semantic nodes
                    for sem_node in semantic_nodes:
                        await subdivide_node(sem_node, depth + 1)
            else:
                logger.debug(f"  → No semantic units found")
                        
        except Exception as e:
            logger.error(f"Error subdividing '{title}': {e}")
        
        # Process any existing non-semantic children
        for child in children:
            if child.get('node_type') != 'semantic_unit':
                await subdivide_node(child, depth)
    
    # Process all top-level nodes
    for node in tree:
        await subdivide_node(node, 0)
    
    logger.info("Semantic subdivision complete")


async def markdown_index_main(md_path: str, opt=None) -> dict:
    """
    Main entry point for markdown processing.
    
    Mirrors page_index_main() for PDFs.
    
    Args:
        md_path: Path to markdown file
        opt: Configuration options
        
    Returns:
        Dict with doc_name and structure
    """
    logger = JsonLogger(md_path)
    
    if not os.path.isfile(md_path):
        raise ValueError(f"File not found: {md_path}")
    
    print('Parsing Markdown...')
    
    # Step 1: Parse markdown (like get_page_tokens for PDF)
    page_list, toc_list, lines = parse_markdown_to_page_list(md_path, opt.model)
    
    logger.info({'total_sections': len(toc_list)})
    logger.info({'total_lines': len(lines)})
    logger.info({'total_tokens': sum([p[1] for p in page_list])})
    
    print(f"Found {len(toc_list)} sections, {len(lines)} lines")
    
    # Step 2: Build tree from TOC (like tree_parser for PDF)
    print('Building tree structure...')
    tree = build_tree_from_toc(toc_list, len(lines))
    
    # Step 3: Apply granular features if enabled
    if opt.granularity in ['medium', 'fine', 'keywords']:
        logger.info(f"Applying granular features for granularity level: {opt.granularity}")
        await apply_granular_features(tree, page_list, opt, logger)
    
    # Step 4: Extract keywords from leaf nodes (for keywords granularity)
    if opt.granularity == 'keywords':
        logger.info("Extracting keywords from leaf nodes...")
        await extract_keywords_from_leaves(tree, opt, logger)
    
    # Step 5: Reassign node IDs
    if opt.if_add_node_id == 'yes':
        write_node_id(tree)
    
    # Step 6: Generate summaries if requested
    if opt.if_add_node_summary == 'yes':
        from .utils import generate_summaries_for_structure
        await generate_summaries_for_structure(tree, model=opt.model, smart_mode=True)
    
    doc_name = os.path.splitext(os.path.basename(md_path))[0]
    
    return {
        'doc_name': doc_name,
        'structure': tree,
        'source': 'markdown'
    }


async def extract_keywords_from_leaves(tree: List[dict], opt, logger: logging.Logger) -> None:
    """
    Extract keywords from all leaf nodes in the tree.
    """
    from .granular.semantic_analyzer import SemanticAnalyzer
    
    analyzer = SemanticAnalyzer(opt.llm_client, logger=logger)
    
    async def process_node(node):
        # Skip keyword nodes
        if node.get('node_type') == 'keyword':
            return
        
        children = node.get('nodes', [])
        non_keyword_children = [c for c in children if c.get('node_type') != 'keyword']
        
        if not non_keyword_children:
            # Leaf node - extract keywords if has substantial text
            text = node.get('text', '')
            if text and len(text) > 50:
                # Skip if already has keywords
                has_keywords = any(c.get('node_type') == 'keyword' for c in children)
                if not has_keywords:
                    try:
                        keywords = analyzer.extract_keywords(node)
                        if keywords:
                            keyword_nodes = analyzer.create_keyword_nodes(keywords, node)
                            if keyword_nodes:
                                if 'nodes' not in node:
                                    node['nodes'] = []
                                node['nodes'].extend(keyword_nodes)
                                logger.info(f"Added {len(keyword_nodes)} keywords to '{node.get('title', 'Unknown')[:40]}'")
                    except Exception as e:
                        logger.error(f"Error extracting keywords: {e}")
        else:
            # Recurse into children
            for child in non_keyword_children:
                await process_node(child)
    
    for node in tree:
        await process_node(node)


def markdown_index(md_path: str, **kwargs) -> dict:
    """
    Synchronous wrapper for markdown_index_main.
    
    Args:
        md_path: Path to markdown file
        **kwargs: Configuration options
        
    Returns:
        Dict with doc_name and structure
    """
    opt = ConfigLoader().load(kwargs)
    return asyncio.run(markdown_index_main(md_path, opt))
