"""
Markdown Processor V2

This module provides a robust, token-based Markdown processor using markdown-it-py.
It supports granular analysis (coarse, medium, fine, keywords) and handles
local image references.
"""

import os
import logging
import asyncio
import time
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

# Third-party
from markdown_it import MarkdownIt
from markdown_it.tree import SyntaxTreeNode

# Import granular components
from pageindex.granular.semantic_analyzer import SemanticAnalyzer
from pageindex.granular.figure_detector import FigureNode, BoundingBox
from pageindex.granular.table_detector import TableNode
from pageindex.llm_client import LLMClient

# Import utils
try:
    from pageindex.utils import count_tokens, JsonLogger
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pageindex.utils import count_tokens, JsonLogger

try:
    from PIL import Image
except ImportError:
    Image = None

@dataclass
class MarkdownNode:
    """Internal representation of a Markdown node."""
    title: str
    level: int
    content_lines: List[str] = field(default_factory=list)
    children: List['MarkdownNode'] = field(default_factory=list)
    start_line: int = 0
    end_line: int = 0
    
    @property
    def text(self) -> str:
        return '\n'.join(self.content_lines).strip()

class MarkdownParser:
    """
    Robust token-based Markdown parser using markdown-it-py.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.md = MarkdownIt()
    
    def parse(self, markdown_text: str) -> List[Dict[str, Any]]:
        """
        Parse Markdown text into a node tree.
        """
        tokens = self.md.parse(markdown_text)
        lines = markdown_text.split('\n')
        
        root_children = []
        stack: List[MarkdownNode] = []
        
        # We need to track where the last node ended to capture content
        # markdown-it tokens give us structure, but we want to group content under headers
        
        # Strategy: Iterate tokens. When we see a heading, that starts a new section.
        # Everything until the next heading (of same or higher level) belongs to it.
        # But markdown-it returns a flat list of tokens.
        
        # We will iterate through the tokens to find headings.
        # We'll use the line numbers from tokens to slice the original text.
        
        current_node = None
        
        # Create a dummy root node to catch content before the first header
        # But usually we just want top-level sections.
        # If there is content before first header, we can put it in a "Preamble" node or similar,
        # or just attach it to the first section if we want (but that's wrong).
        # Let's create a "Introduction" node if content exists before first header?
        # Or just return it as a node with empty title?
        
        # Let's stick to the stack approach.
        
        for i, token in enumerate(tokens):
            if token.type == 'heading_open':
                # Start of a new section
                level = int(token.tag[1:]) # h1 -> 1
                
                # Get title from the next token (inline)
                title = ""
                if i + 1 < len(tokens) and tokens[i+1].type == 'inline':
                    title = tokens[i+1].content
                
                # Determine start line (0-indexed)
                start_line = token.map[0] if token.map else 0
                
                # Close previous node content
                if stack:
                    # The previous node ends where this one starts (roughly)
                    # We'll refine text extraction later
                    pass

                new_node = MarkdownNode(title=title, level=level, start_line=start_line)
                
                # Pop stack
                while stack and stack[-1].level >= level:
                    completed_node = stack.pop()
                    if not stack:
                        root_children.append(completed_node)
                    else:
                        stack[-1].children.append(completed_node)
                
                stack.append(new_node)
                
        # Unwind stack
        while stack:
            completed_node = stack.pop()
            if not stack:
                root_children.append(completed_node)
            else:
                stack[-1].children.append(completed_node)
        
        # Now we have the structure, but we need the content (text).
        # We can assign text based on line ranges.
        # Flatten the tree to sort by start_line
        all_nodes = self._flatten_nodes(root_children)
        
        # Sort by start_line
        all_nodes.sort(key=lambda x: x.start_line)
        
        # Assign end lines based on next node's start
        for i, node in enumerate(all_nodes):
            node_start = node.start_line
            # The content starts AFTER the header line
            # But wait, start_line from token is the header line.
            # We want to include the header in the text? Usually yes, for context.
            
            if i + 1 < len(all_nodes):
                node_end = all_nodes[i+1].start_line
            else:
                node_end = len(lines)
            
            node.end_line = node_end
            node.content_lines = lines[node_start:node_end]
            
            # Remove children's text from parent?
            # PageIndex usually includes full text in the node, including children?
            # The previous implementation `update_node_list_with_text_token_count` did:
            # "Add all children's text"
            # So yes, parent text includes children text.
            # Our slicing method `lines[node_start:node_end]` AUTOMATICALLY includes children text
            # because children are physically located between start and end of parent.
            # Perfect.
            
        return [self._node_to_dict(node) for node in root_children]

    def _flatten_nodes(self, nodes: List[MarkdownNode]) -> List[MarkdownNode]:
        flat = []
        for node in nodes:
            flat.append(node)
            flat.extend(self._flatten_nodes(node.children))
        return flat

    def _node_to_dict(self, node: MarkdownNode) -> Dict[str, Any]:
        """Convert MarkdownNode to dictionary."""
        node_dict = {
            'title': node.title,
            'text': node.text,
            'line_num': node.start_line + 1, # 1-indexed
            'nodes': [self._node_to_dict(child) for child in node.children]
        }
        return node_dict


class MarkdownFigureExtractor:
    """Extracts figures from Markdown text."""
    
    def __init__(self, llm_client, logger: Optional[logging.Logger] = None):
        self.llm_client = llm_client
        self.logger = logger or logging.getLogger(__name__)
        self.md = MarkdownIt()
    
    async def extract(self, text: str, page_num: int = 1, base_path: str = "") -> List[Dict[str, Any]]:
        """Extract figures and return them as node dictionaries."""
        figures = []
        tokens = self.md.parse(text)
        
        # Iterate tokens to find images
        # Images are usually inline tokens inside a paragraph
        
        for token in tokens:
            if token.type == 'inline':
                for child in token.children:
                    if child.type == 'image':
                        caption = child.content
                        src = child.attrs.get('src', '')
                        
                        # Resolve path
                        full_path = src
                        if base_path and not src.startswith(('http://', 'https://', 'ftp://')):
                            full_path = os.path.join(base_path, src)
                        
                        # Try to load image for summary generation (if we had vision capability here)
                        # For now, we'll just check if it exists
                        image_exists = False
                        if os.path.exists(full_path):
                            image_exists = True
                        
                        # Generate summary
                        summary = caption
                        if not summary:
                            summary = f"Figure: {os.path.basename(src)}"
                        
                        # Create figure node
                        figures.append({
                            'node_type': 'figure',
                            'title': f"Figure: {caption}" if caption else "Figure",
                            'text': f"Image Source: {src}\nCaption: {caption}",
                            'summary': summary,
                            'nodes': [],
                            'metadata': {
                                'src': src,
                                'full_path': full_path,
                                'exists': image_exists
                            }
                        })
            
        return figures


class MarkdownTableExtractor:
    """Extracts tables from Markdown text."""
    
    def __init__(self, llm_client, logger: Optional[logging.Logger] = None):
        self.llm_client = llm_client
        self.logger = logger or logging.getLogger(__name__)
        # We can use markdown-it to detect tables if the plugin is enabled
        # But regex is often simpler for just extracting the block of text
        # Let's stick to regex for now as it's robust enough for extraction
        self.table_row_pattern = re.compile(r'^\s*\|.*\|\s*$')
    
    async def extract(self, text: str, page_num: int = 1) -> List[Dict[str, Any]]:
        """Extract tables and return them as node dictionaries."""
        tables = []
        lines = text.split('\n')
        current_table_lines = []
        
        for line in lines:
            if self.table_row_pattern.match(line):
                current_table_lines.append(line)
            else:
                if current_table_lines:
                    # End of a table
                    if len(current_table_lines) >= 2: # At least header and separator
                        table_text = '\n'.join(current_table_lines)
                        tables.append(self._create_table_node(table_text, page_num))
                    current_table_lines = []
        
        # Check last table
        if current_table_lines and len(current_table_lines) >= 2:
            table_text = '\n'.join(current_table_lines)
            tables.append(self._create_table_node(table_text, page_num))
            
        return tables

    def _create_table_node(self, table_text: str, page_num: int) -> Dict[str, Any]:
        """Create a table node dictionary."""
        return {
            'node_type': 'table',
            'title': "Table",
            'text': table_text,
            'summary': "Data table extracted from Markdown",
            'nodes': []
        }


class MarkdownGranularIntegrator:
    """Integrates granular features into the Markdown tree."""
    
    def __init__(self, llm_client, logger: Optional[logging.Logger] = None):
        self.llm_client = llm_client
        self.logger = logger or logging.getLogger(__name__)
        self.semantic_analyzer = SemanticAnalyzer(llm_client, logger)
        self.figure_extractor = MarkdownFigureExtractor(llm_client, logger)
        self.table_extractor = MarkdownTableExtractor(llm_client, logger)
    
    async def apply_granularity(self, tree: List[Dict[str, Any]], granularity: str, base_path: str = "", keyword_level: str = 'fine'):
        """
        Apply granular processing (semantic subdivision, keywords, etc.) to the tree.
        """
        if granularity == 'coarse':
            return
        
        # Recursively process nodes
        await self._process_nodes_recursive(tree, granularity, depth=0, base_path=base_path, keyword_level=keyword_level)
        

    async def _process_nodes_recursive(self, nodes: List[Dict[str, Any]], granularity: str, depth: int = 0, base_path: str = "", keyword_level: str = 'fine'):
        """Recursively process nodes."""
        if not nodes:
            return
            
        self.logger.info(f"Processing {len(nodes)} nodes at depth {depth}")
        
        # Determine target depth for keywords
        keyword_depth_map = {'section': 0, 'medium': 1, 'fine': 2}
        target_keyword_depth = keyword_depth_map.get(keyword_level, 2)
        
        # Determine subdivision depth limit
        # medium: subdivide sections (depth 0) -> depth 1 units. Limit = 1.
        # fine: subdivide sections (0) -> depth 1 -> depth 2 units. Limit = 2.
        # keywords: depends on keyword_level
        subdivision_depth_limit = 1
        if granularity == 'fine':
            subdivision_depth_limit = 2
        elif granularity == 'keywords':
            subdivision_depth_limit = target_keyword_depth

            
        for node in nodes:
            node_type = node.get('node_type', 'section')
            
            # 1. Extract Figures and Tables (only at top level sections usually, but check all)
            # (Existing logic for figures/tables - assuming they are already in the tree or extracted here)
            # The parser puts them in the tree. We might want to summarize them here if needed.
            # For now, we focus on semantic units.
            if node_type == 'section' and node.get('text'):
                figures = await self.figure_extractor.extract(node['text'], base_path=base_path)
                tables = await self.table_extractor.extract(node['text'])
                
                # Add them as children
                if figures or tables:
                    node.setdefault('nodes', []).extend(figures + tables)

            # 2. Semantic Subdivision
            # We subdivide if:
            # - Granularity requires it (medium+, fine+, keywords+)
            # - We haven't reached the depth limit
            # - It's a section or semantic unit
            should_subdivide = False
            if granularity in ['medium', 'fine', 'keywords']:
                if depth < subdivision_depth_limit:
                    should_subdivide = True
            
            did_subdivide = False
            if should_subdivide and node.get('text'):
                start_time = time.time()
                
                # Mock page texts for SemanticAnalyzer
                mock_page_texts = [(node['text'], count_tokens(node['text']))]
                
                # Create a dummy node structure for analyzer
                analysis_node = {
                    'title': node['title'],
                    'text': node['text'],
                    'start_index': 1,
                    'end_index': 1
                }
                
                # Call analyzer
                self.logger.info(f"Subdividing '{node['title']}' (depth {depth})...")
                semantic_units = self.semantic_analyzer.analyze_section(
                    analysis_node, mock_page_texts, min_pages=0.0, min_paragraphs=1
                )
                
                duration = time.time() - start_time
                self.logger.info(f"  -> Found {len(semantic_units)} units in {duration:.2f}s")
                
                if semantic_units:
                    did_subdivide = True
                    # Convert semantic units to nodes
                    new_children = []
                    for unit in semantic_units:
                        new_children.append({
                            'node_type': 'semantic_unit',
                            'title': unit.title,
                            'text': self._extract_text_for_unit(node['text'], unit),
                            'summary': unit.summary,
                            'nodes': []
                        })
                    
                    # Append new semantic children
                    node.setdefault('nodes', []).extend(new_children)
            
            # 3. Keyword Extraction (if granularity is keywords)
            if granularity == 'keywords':
                should_extract = False
                
                # If we are at the target depth, extract
                if depth == target_keyword_depth:
                    should_extract = True
                
                # If we are not at target depth, but couldn't subdivide further (leaf node), extract
                # This ensures we don't miss content if the structure is shallower than expected
                elif depth < target_keyword_depth and not did_subdivide:
                    # Only extract from sections or semantic units (skip figures/tables)
                    if node_type in ['section', 'semantic_unit']:
                        should_extract = True
                
                if should_extract:
                    node['_extract_keywords'] = True

        # Collect all semantic units for parallel keyword extraction
        semantic_units_to_process = []
        for node in nodes:
            if node.get('_extract_keywords'):
                semantic_units_to_process.append(node)
                del node['_extract_keywords'] # Clean up flag
            
            # Recurse into children (if any)
            if 'nodes' in node and node['nodes']:
                 # Only recurse if we haven't reached the limit or if we need to go deeper for some reason
                 # Actually we should always recurse if there are nodes, to process them.
                 # But we only process 'semantic_unit' children recursively for subdivision.
                 # The 'nodes' list might contain figures/tables too.
                 
                 # Filter for semantic units/sections to recurse
                 children_to_recurse = [n for n in node['nodes'] if n.get('node_type', 'section') in ['section', 'semantic_unit']]
                 if children_to_recurse:
                     await self._process_nodes_recursive(children_to_recurse, granularity, depth + 1, base_path, keyword_level)

        if semantic_units_to_process:
            
            async def extract_keywords_async(node):
                try:
                    start_time = time.time()
                    self.logger.info(f"Extracting keywords for semantic unit: '{node.get('title')}'")
                    
                    # Run in thread to avoid blocking
                    keywords = await asyncio.to_thread(self.semantic_analyzer.extract_keywords, node)
                    
                    duration = time.time() - start_time
                    if keywords:
                        self.logger.info(f"  -> Found {len(keywords)} keywords in {duration:.2f}s")
                        keyword_nodes = self.semantic_analyzer.create_keyword_nodes(keywords, node)
                        node.setdefault('nodes', []).extend(keyword_nodes)
                    else:
                        self.logger.info(f"  -> No keywords found (took {duration:.2f}s)")
                except Exception as e:
                    self.logger.warning(f"Error extracting keywords for '{node.get('title')}': {e}")

            # Run all extractions in parallel
            self.logger.info(f"Starting parallel keyword extraction for {len(semantic_units_to_process)} units...")
            await asyncio.gather(*(extract_keywords_async(node) for node in semantic_units_to_process))

    def _extract_text_for_unit(self, full_text: str, unit) -> str:
        """Extract text for a semantic unit based on paragraph indices."""
        paragraphs = []
        for p in full_text.split('\n\n'):
            p = p.strip()
            if p:
                paragraphs.append(p)
        if len(paragraphs) <= 1:
            paragraphs = [p.strip() for p in full_text.split('\n') if p.strip()]
            
        start = unit.start_paragraph
        end = unit.end_paragraph
        
        if start < 0: start = 0
        if end >= len(paragraphs): end = len(paragraphs) - 1
        
        return '\n\n'.join(paragraphs[start:end+1])


async def process_markdown_v2(file_path: str, granularity: str = 'medium', llm_client: LLMClient = None, keyword_level: str = 'fine') -> Dict[str, Any]:
    """
    Process a markdown file with the specified granularity.
    
    Args:
        file_path: Path to the markdown file
        granularity: 'coarse', 'medium', 'fine', or 'keywords'
        llm_client: Initialized LLMClient
        keyword_level: 'section', 'medium', or 'fine' (only used if granularity='keywords')
    """
    if llm_client is None:
        raise ValueError("LLMClient is required")
        
    start_time = time.time()
    
    # 1. Parse Markdown
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    parser = MarkdownParser()
    tree = parser.parse(content)
    
    # 2. Apply Granularity
    integrator = MarkdownGranularIntegrator(llm_client)
    await integrator.apply_granularity(tree, granularity, keyword_level=keyword_level)
    
    # 3. Add Metadata
    result = {
        "doc_name": os.path.basename(file_path),
        "structure": tree,
        "source": "markdown",
        "granularity": granularity,
        "processor": "v2"
    }
    
    duration = time.time() - start_time
    # print(f"Processing complete in {duration:.2f}s")
    
    return result
