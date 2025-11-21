"""
Integration module for markdown-based document processing.
Allows PageIndex to work with markdown files as an alternative to PDF parsing.
"""

from typing import Optional, Dict, List
import logging
from pathlib import Path

from .markdown_processor import MarkdownProcessor, process_markdown_to_tree
from .utils import count_tokens


def markdown_page_index(markdown_path: str,
                       metadata_path: Optional[str] = None,
                       opt: Optional[Dict] = None,
                       logger: Optional[logging.Logger] = None) -> Dict:
    """
    Generate PageIndex structure from markdown file.
    
    Args:
        markdown_path: Path to markdown file
        metadata_path: Optional path to metadata JSON file
        opt: Optional configuration dictionary
        logger: Optional logger instance
    
    Returns:
        Dictionary containing document structure
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Processing markdown file: {markdown_path}")
    
    # Process markdown to tree
    result = process_markdown_to_tree(markdown_path, metadata_path, logger)
    
    tree = result['tree']
    figures = result['figures']
    tables = result['tables']
    
    # Add node IDs
    tree = _add_node_ids(tree)
    
    # Optionally add text content
    if opt and opt.get('if_add_node_text') == 'yes':
        processor = MarkdownProcessor(markdown_path, metadata_path, logger)
        tree = _add_text_to_nodes(tree, processor)
    
    # Optionally add summaries
    if opt and opt.get('if_add_node_summary') == 'yes':
        from .utils import generate_summaries_for_structure
        tree = generate_summaries_for_structure(tree, opt.get('model'))
    
    structure = {
        'tree': tree,
        'page_count': result['page_count'],
        'source': 'markdown',
        'markdown_path': str(markdown_path)
    }
    
    # Add figures and tables if present
    if figures:
        structure['figures'] = figures
    if tables:
        structure['tables'] = tables
    
    logger.info(f"Markdown processing complete. Found {len(tree)} top-level sections")
    
    return structure


def _add_node_ids(nodes: List[Dict], prefix: str = "") -> List[Dict]:
    """Add hierarchical node IDs to tree."""
    for i, node in enumerate(nodes):
        node_id = f"{prefix}{i+1}" if prefix else str(i+1)
        node['node_id'] = node_id
        
        if 'children' in node and node['children']:
            node['children'] = _add_node_ids(node['children'], f"{node_id}.")
    
    return nodes


def _add_text_to_nodes(nodes: List[Dict], processor: MarkdownProcessor) -> List[Dict]:
    """Add text content to nodes based on line ranges."""
    for node in nodes:
        if 'line_start' in node and 'line_end' in node:
            start = node['line_start'] - 1  # Convert to 0-indexed
            end = node['line_end']
            node['text'] = '\n'.join(processor.lines[start:end])
            node['token_count'] = count_tokens(node['text'])
        
        if 'children' in node and node['children']:
            node['children'] = _add_text_to_nodes(node['children'], processor)
    
    return nodes


def enhance_pdf_structure_with_markdown(pdf_structure: Dict,
                                        markdown_path: str,
                                        metadata_path: Optional[str] = None,
                                        logger: Optional[logging.Logger] = None) -> Dict:
    """
    Enhance existing PDF-based structure with markdown information.
    
    This can be useful when you have both PDF and markdown versions,
    and want to use markdown's better text extraction while keeping
    PDF's page structure.
    
    Args:
        pdf_structure: Existing PageIndex structure from PDF
        markdown_path: Path to markdown file
        metadata_path: Optional metadata JSON path
        logger: Optional logger
    
    Returns:
        Enhanced structure combining both sources
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("Enhancing PDF structure with markdown content")
    
    processor = MarkdownProcessor(markdown_path, metadata_path, logger)
    
    # Extract better text from markdown
    page_texts = processor.extract_text_by_page()
    
    # Extract figures and tables from markdown
    figures, tables = processor.extract_figures_and_tables()
    
    # Merge into PDF structure
    enhanced = pdf_structure.copy()
    
    if page_texts:
        enhanced['markdown_text'] = page_texts
    
    if figures:
        enhanced['markdown_figures'] = figures
    
    if tables:
        enhanced['markdown_tables'] = tables
    
    enhanced['enhanced_with_markdown'] = True
    enhanced['markdown_source'] = str(markdown_path)
    
    logger.info(f"Enhanced structure with {len(page_texts)} pages of markdown text")
    
    return enhanced


def create_hybrid_structure(pdf_path: str,
                           markdown_path: str,
                           metadata_path: Optional[str] = None,
                           opt: Optional[Dict] = None,
                           logger: Optional[logging.Logger] = None) -> Dict:
    """
    Create hybrid structure using both PDF and markdown.
    
    Uses PDF for structure detection and markdown for better text extraction.
    
    Args:
        pdf_path: Path to PDF file
        markdown_path: Path to markdown file
        metadata_path: Optional metadata JSON path
        opt: Optional configuration
        logger: Optional logger
    
    Returns:
        Hybrid structure combining both sources
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("Creating hybrid PDF+Markdown structure")
    
    # Import here to avoid circular dependency
    from .page_index import page_index_main
    import fitz
    
    # Get PDF structure
    doc = fitz.open(pdf_path)
    pdf_structure = page_index_main(doc, opt)
    doc.close()
    
    # Enhance with markdown
    hybrid = enhance_pdf_structure_with_markdown(
        pdf_structure,
        markdown_path,
        metadata_path,
        logger
    )
    
    hybrid['source'] = 'hybrid_pdf_markdown'
    hybrid['pdf_path'] = str(pdf_path)
    
    return hybrid
