"""
Markdown adapter for PageIndex.
Converts markdown to page_list format and uses existing PDF processing pipeline.
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import tiktoken

from .markdown_processor import MarkdownProcessor
from .page_index import tree_parser
from .utils import (
    ConfigLoader, 
    JsonLogger, 
    write_node_id, 
    generate_summaries_for_structure,
    create_clean_structure_for_description,
    generate_doc_description,
    remove_structure_text,
    count_tokens
)


class MarkdownDocument:
    """Mock document object for markdown that mimics file path string."""
    
    def __init__(self, markdown_path: str, metadata_path: Optional[str] = None):
        self.markdown_path = markdown_path
        self.metadata_path = metadata_path
        self.processor = MarkdownProcessor(markdown_path, metadata_path)
        self.name = Path(markdown_path).stem
        # Make it behave like a string path for JsonLogger
        self._path = markdown_path
    
    def __str__(self):
        return self._path
    
    def __fspath__(self):
        """Support os.PathLike protocol."""
        return self._path


def markdown_to_page_list(
    markdown_path: str,
    metadata_path: Optional[str] = None,
    model: str = "gpt-4o-2024-11-20"
) -> List[Tuple[str, int]]:
    """
    Convert markdown file to page_list format compatible with PageIndex pipeline.
    
    Args:
        markdown_path: Path to markdown file
        metadata_path: Optional metadata JSON path
        model: Model name for token counting
    
    Returns:
        List of (page_text, token_count) tuples
    """
    processor = MarkdownProcessor(markdown_path, metadata_path)
    
    # Handle Gemini models which tiktoken doesn't know about
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to cl100k_base for Gemini and other unknown models
        enc = tiktoken.get_encoding("cl100k_base")
    
    # If we have metadata with page stats, use that
    if processor.metadata and 'page_stats' in processor.metadata:
        page_count = len(processor.metadata['page_stats'])
        
        # Split markdown content by pages
        # Simple approach: divide content evenly
        # Better approach: use page markers if available
        page_list = []
        
        # Try to find page markers in markdown
        lines = processor.lines
        page_texts = _split_by_page_markers(lines, page_count)
        
        if not page_texts:
            # Fallback: split evenly
            page_texts = _split_evenly(processor.markdown_content, page_count)
        
        for page_text in page_texts:
            token_count = len(enc.encode(page_text))
            page_list.append((page_text, token_count))
        
        return page_list
    
    else:
        # No metadata: treat entire document as one page
        text = processor.markdown_content
        token_count = len(enc.encode(text))
        return [(text, token_count)]


def _split_by_page_markers(lines: List[str], expected_pages: int) -> List[str]:
    """Split markdown by page markers if they exist."""
    import re
    
    page_texts = {}
    current_page = 0
    current_lines = []
    
    for line in lines:
        # Look for page markers (common in marker output)
        # Could be: <!-- Page X --> or similar
        page_match = re.match(r'<!--\s*Page\s+(\d+)\s*-->', line, re.IGNORECASE)
        if page_match:
            if current_lines:
                page_texts[current_page] = '\n'.join(current_lines)
            current_page = int(page_match.group(1))
            current_lines = []
        else:
            current_lines.append(line)
    
    # Add last page
    if current_lines:
        page_texts[current_page] = '\n'.join(current_lines)
    
    # If we found page markers, return in order
    if page_texts and len(page_texts) >= expected_pages * 0.5:  # At least half the pages
        result = []
        for i in range(expected_pages):
            result.append(page_texts.get(i, ''))
        return result
    
    return []


def _split_evenly(text: str, page_count: int) -> List[str]:
    """Split text evenly into pages."""
    if page_count <= 1:
        return [text]
    
    lines = text.split('\n')
    lines_per_page = len(lines) // page_count
    
    pages = []
    for i in range(page_count):
        start = i * lines_per_page
        end = start + lines_per_page if i < page_count - 1 else len(lines)
        page_text = '\n'.join(lines[start:end])
        pages.append(page_text)
    
    return pages


async def markdown_page_index_main(
    markdown_path: str,
    metadata_path: Optional[str] = None,
    opt: Optional[Dict] = None
) -> Dict:
    """
    Process markdown file using the full PageIndex pipeline.
    
    This converts markdown to page_list format and uses the same
    tree_parser and granular features as PDF processing.
    
    Args:
        markdown_path: Path to markdown file
        metadata_path: Optional metadata JSON path
        opt: Configuration options (same as PDF processing)
    
    Returns:
        PageIndex structure with doc_name and structure
    """
    # Load config
    if opt is None:
        opt = ConfigLoader().load({})
    
    # Create mock document for logging
    # Pass markdown_path directly as string - JsonLogger will extract the name
    doc = markdown_path
    logger = JsonLogger(markdown_path)
    
    print('Processing markdown file...')
    
    # Convert markdown to page_list format
    page_list = markdown_to_page_list(markdown_path, metadata_path, opt.model)
    
    logger.info({'total_page_number': len(page_list)})
    logger.info({'total_token': sum([page[1] for page in page_list])})
    logger.info({'source': 'markdown'})
    
    # Use the same tree_parser as PDF processing!
    # This gives us all the same features: TOC detection, semantic subdivision,
    # figure/table detection, etc.
    structure = await tree_parser(page_list, opt, doc=doc, logger=logger)
    
    # Reassign node IDs after all tree modifications
    if opt.if_add_node_id == 'yes':
        write_node_id(structure)
    
    # Add text if requested
    if opt.if_add_node_text == 'yes':
        _add_node_text_from_markdown(structure, page_list)
    
    # Add summaries if requested
    if opt.if_add_node_summary == 'yes':
        if opt.if_add_node_text == 'no':
            _add_node_text_from_markdown(structure, page_list)
        await generate_summaries_for_structure(structure, model=opt.model)
        if opt.if_add_node_text == 'no':
            remove_structure_text(structure)
        
        # Add doc description if requested
        if opt.if_add_doc_description == 'yes':
            clean_structure = create_clean_structure_for_description(structure)
            doc_description = generate_doc_description(clean_structure, model=opt.model)
            return {
                'doc_name': Path(markdown_path).stem,
                'doc_description': doc_description,
                'structure': structure,
                'source': 'markdown'
            }
    
    return {
        'doc_name': Path(markdown_path).stem,
        'structure': structure,
        'source': 'markdown'
    }


def _add_node_text_from_markdown(structure: List[Dict], page_list: List[Tuple[str, int]]):
    """Add text to nodes from page_list (same as PDF processing)."""
    from .utils import add_node_text
    add_node_text(structure, page_list)


def markdown_page_index(
    markdown_path: str,
    metadata_path: Optional[str] = None,
    model: Optional[str] = None,
    granularity: str = 'medium',
    enable_figure_detection: bool = True,
    enable_table_detection: bool = True,
    enable_semantic_subdivision: bool = True,
    semantic_min_pages: float = 0.5,
    if_add_node_id: str = 'yes',
    if_add_node_summary: str = 'yes',
    if_add_doc_description: str = 'no',
    if_add_node_text: str = 'no',
    **kwargs
) -> Dict:
    """
    Simplified interface for markdown processing with PageIndex.
    
    Args:
        markdown_path: Path to markdown file
        metadata_path: Optional metadata JSON path
        model: LLM model to use
        granularity: 'coarse', 'medium', or 'fine'
        enable_figure_detection: Enable figure detection
        enable_table_detection: Enable table detection
        enable_semantic_subdivision: Enable semantic subdivision
        semantic_min_pages: Minimum pages for semantic subdivision
        if_add_node_id: Add node IDs
        if_add_node_summary: Add summaries
        if_add_doc_description: Add document description
        if_add_node_text: Add text content
        **kwargs: Additional options
    
    Returns:
        PageIndex structure
    """
    # Build options
    user_opt = {
        'model': model,
        'granularity': granularity,
        'enable_figure_detection': enable_figure_detection,
        'enable_table_detection': enable_table_detection,
        'enable_semantic_subdivision': enable_semantic_subdivision,
        'semantic_min_pages': semantic_min_pages,
        'if_add_node_id': if_add_node_id,
        'if_add_node_summary': if_add_node_summary,
        'if_add_doc_description': if_add_doc_description,
        'if_add_node_text': if_add_node_text,
    }
    user_opt.update(kwargs)
    
    # Remove None values
    user_opt = {k: v for k, v in user_opt.items() if v is not None}
    
    # Load config with defaults
    opt = ConfigLoader().load(user_opt)
    
    # Process markdown
    return asyncio.run(markdown_page_index_main(markdown_path, metadata_path, opt))
