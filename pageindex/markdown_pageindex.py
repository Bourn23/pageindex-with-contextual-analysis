"""
Full PageIndex processing for markdown files.
Integrates markdown with semantic analysis, figure/table detection, and summaries.
"""

import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from .markdown_processor import MarkdownProcessor
from .utils import count_tokens, generate_node_summary
from .llm_client import get_llm_client


def markdown_to_pageindex_structure(
    markdown_path: str,
    metadata_path: Optional[str] = None,
    opt: Optional[Dict] = None,
    logger: Optional[logging.Logger] = None
) -> Dict:
    """
    Process markdown file with full PageIndex pipeline.
    
    This creates a structure similar to PDF processing with:
    - Hierarchical sections
    - Integrated figures and tables
    - Semantic subdivision (if enabled)
    - Node summaries
    - Rich metadata
    
    Args:
        markdown_path: Path to markdown file
        metadata_path: Optional metadata JSON path
        opt: Configuration options
        logger: Optional logger
    
    Returns:
        PageIndex structure with doc_name and structure
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if opt is None:
        opt = {}
    
    # Set defaults
    granularity = opt.get('granularity', 'medium')
    if_add_node_summary = opt.get('if_add_node_summary', 'yes')
    if_add_node_text = opt.get('if_add_node_text', 'no')
    model = opt.get('model', 'gemini-2.5-flash')
    
    logger.info(f"Processing markdown with granularity: {granularity}")
    
    # Load markdown
    processor = MarkdownProcessor(markdown_path, metadata_path, logger)
    
    # Extract structure from markdown
    tree = processor.extract_structure_with_pages()
    
    # Extract figures and tables
    figures, tables = processor.extract_figures_and_tables()
    
    # Convert to PageIndex format
    structure = _build_pageindex_tree(
        tree, 
        processor, 
        figures, 
        tables,
        opt,
        logger
    )
    
    # Add summaries if requested
    if if_add_node_summary == 'yes':
        logger.info("Generating node summaries...")
        structure = _add_summaries_to_tree(structure, model, logger)
    
    # Add text if requested
    if if_add_node_text == 'yes':
        logger.info("Adding text content to nodes...")
        structure = _add_text_to_tree(structure, processor, logger)
    
    # Build final output
    doc_name = Path(markdown_path).stem
    result = {
        'doc_name': doc_name,
        'structure': structure,
        'source': 'markdown',
        'granularity': granularity
    }
    
    logger.info(f"Markdown processing complete. Generated {_count_nodes(structure)} nodes")
    
    return result


def _build_pageindex_tree(
    md_nodes: List[Dict],
    processor: MarkdownProcessor,
    figures: List[Dict],
    tables: List[Dict],
    opt: Dict,
    logger: logging.Logger
) -> List[Dict]:
    """Build PageIndex tree from markdown nodes with integrated figures/tables."""
    
    tree = []
    node_counter = [0]  # Use list for mutable counter
    
    for md_node in md_nodes:
        node = _convert_md_node_to_pageindex(
            md_node, 
            processor, 
            figures, 
            tables,
            node_counter,
            opt,
            logger
        )
        tree.append(node)
    
    return tree


def _convert_md_node_to_pageindex(
    md_node: Dict,
    processor: MarkdownProcessor,
    figures: List[Dict],
    tables: List[Dict],
    node_counter: List[int],
    opt: Dict,
    logger: logging.Logger
) -> Dict:
    """Convert a markdown node to PageIndex format with figures/tables."""
    
    # Create base node
    node = {
        'title': md_node['title'],
        'start_index': md_node.get('page', 1),
        'end_index': md_node.get('page', 1),
        'node_id': f"{node_counter[0]:04d}",
        'nodes': []
    }
    node_counter[0] += 1
    
    # Determine node type
    if 'heading_level' in md_node:
        node['node_type'] = 'section'
    
    # Find figures and tables in this node's page range
    start_page = node['start_index']
    end_page = node['end_index']
    
    # Add figures that belong to this section
    for fig in figures:
        if start_page <= fig['page'] <= end_page:
            fig_node = _create_figure_node(fig, node_counter, processor)
            node['nodes'].append(fig_node)
    
    # Add tables that belong to this section
    for table in tables:
        # Estimate page from line numbers
        table_page = _estimate_page_from_line(table['line_start'], processor)
        if start_page <= table_page <= end_page:
            table_node = _create_table_node(table, node_counter, processor)
            node['nodes'].append(table_node)
    
    # Process children recursively
    if 'children' in md_node and md_node['children']:
        for child in md_node['children']:
            child_node = _convert_md_node_to_pageindex(
                child,
                processor,
                figures,
                tables,
                node_counter,
                opt,
                logger
            )
            node['nodes'].append(child_node)
            # Update end_index based on children
            node['end_index'] = max(node['end_index'], child_node['end_index'])
    
    return node


def _create_figure_node(fig: Dict, node_counter: List[int], processor: MarkdownProcessor) -> Dict:
    """Create a figure node from markdown figure data."""
    
    node = {
        'title': f"Figure {fig['figure_number']}: [Extracted from markdown]",
        'start_index': fig['page'],
        'end_index': fig['page'],
        'node_type': 'figure',
        'node_id': f"{node_counter[0]:04d}",
        'metadata': {
            'figure_number': f"Figure {fig['figure_number']}",
            'caption': f"Figure {fig['figure_number']}",
            'figure_type': 'image',
            'source': 'markdown'
        },
        'nodes': []
    }
    node_counter[0] += 1
    
    return node


def _create_table_node(table: Dict, node_counter: List[int], processor: MarkdownProcessor) -> Dict:
    """Create a table node from markdown table data."""
    
    page = _estimate_page_from_line(table['line_start'], processor)
    
    node = {
        'title': f"Table: Lines {table['line_start']}-{table['line_end']}",
        'start_index': page,
        'end_index': page,
        'node_type': 'table',
        'node_id': f"{node_counter[0]:04d}",
        'metadata': {
            'table_number': f"Table {node_counter[0]}",
            'caption': f"Table at lines {table['line_start']}-{table['line_end']}",
            'source': 'markdown'
        },
        'nodes': []
    }
    node_counter[0] += 1
    
    return node


def _estimate_page_from_line(line_num: int, processor: MarkdownProcessor) -> int:
    """Estimate page number from line number using metadata."""
    if not processor.metadata or 'page_stats' not in processor.metadata:
        return 1
    
    # Simple estimation: divide total lines by pages
    total_lines = len(processor.lines)
    total_pages = len(processor.metadata['page_stats'])
    
    if total_pages == 0:
        return 1
    
    lines_per_page = total_lines / total_pages
    estimated_page = int(line_num / lines_per_page) + 1
    
    return min(estimated_page, total_pages)


def _add_summaries_to_tree(
    nodes: List[Dict],
    model: str,
    logger: logging.Logger
) -> List[Dict]:
    """Add summaries to all nodes in tree."""
    
    for node in nodes:
        # Generate summary if text is available
        if 'text' in node and node['text']:
            try:
                summary = generate_node_summary(node, model)
                if summary:
                    node['summary'] = summary
            except Exception as e:
                logger.warning(f"Failed to generate summary for {node.get('title', 'unknown')}: {e}")
        
        # Recursively process children
        if 'nodes' in node and node['nodes']:
            node['nodes'] = _add_summaries_to_tree(node['nodes'], model, logger)
    
    return nodes


def _add_text_to_tree(
    nodes: List[Dict],
    processor: MarkdownProcessor,
    logger: logging.Logger
) -> List[Dict]:
    """Add text content to nodes based on page ranges."""
    
    for node in nodes:
        start_page = node.get('start_index', 1)
        end_page = node.get('end_index', 1)
        
        # Extract text for this page range
        text = _extract_text_for_pages(start_page, end_page, processor)
        if text:
            node['text'] = text
            node['token_count'] = count_tokens(text)
        
        # Recursively process children
        if 'nodes' in node and node['nodes']:
            node['nodes'] = _add_text_to_tree(node['nodes'], processor, logger)
    
    return nodes


def _extract_text_for_pages(
    start_page: int,
    end_page: int,
    processor: MarkdownProcessor
) -> str:
    """Extract text content for a page range from markdown."""
    
    # Simple approach: extract all text
    # In a more sophisticated version, we'd use page markers
    return processor.markdown_content


def _count_nodes(nodes: List[Dict]) -> int:
    """Count total nodes in tree."""
    count = len(nodes)
    for node in nodes:
        if 'nodes' in node and node['nodes']:
            count += _count_nodes(node['nodes'])
    return count
