"""
Markdown processor for PageIndex.
Processes markdown files (e.g., from marker library) to extract document structure.
"""

import json
import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging


class MarkdownProcessor:
    """Process markdown files to extract document structure."""
    
    def __init__(self, markdown_path: str, metadata_path: Optional[str] = None, 
                 logger: Optional[logging.Logger] = None):
        """
        Initialize markdown processor.
        
        Args:
            markdown_path: Path to markdown file
            metadata_path: Optional path to metadata JSON file
            logger: Optional logger instance
        """
        self.markdown_path = Path(markdown_path)
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self.logger = logger or logging.getLogger(__name__)
        
        self.markdown_content = None
        self.metadata = None
        self.lines = []
        
        self._load_files()
    
    def _load_files(self):
        """Load markdown and metadata files."""
        try:
            with open(self.markdown_path, 'r', encoding='utf-8') as f:
                self.markdown_content = f.read()
                self.lines = self.markdown_content.split('\n')
            self.logger.info(f"Loaded markdown file: {self.markdown_path}")
        except Exception as e:
            self.logger.error(f"Failed to load markdown file: {e}")
            raise
        
        if self.metadata_path and self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                self.logger.info(f"Loaded metadata file: {self.metadata_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load metadata file: {e}")
    
    def extract_structure(self) -> List[Dict]:
        """
        Extract hierarchical structure from markdown.
        
        Returns:
            List of nodes representing document structure
        """
        nodes = []
        current_stack = []  # Stack to track hierarchy
        
        for line_num, line in enumerate(self.lines, 1):
            # Detect headers
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if header_match:
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                
                # Create node
                node = {
                    'title': title,
                    'level': level,
                    'line_start': line_num,
                    'line_end': None,  # Will be updated later
                    'children': []
                }
                
                # Update hierarchy
                while current_stack and current_stack[-1]['level'] >= level:
                    popped = current_stack.pop()
                    popped['line_end'] = line_num - 1
                
                if current_stack:
                    current_stack[-1]['children'].append(node)
                else:
                    nodes.append(node)
                
                current_stack.append(node)
        
        # Close remaining nodes
        last_line = len(self.lines)
        for node in current_stack:
            if node['line_end'] is None:
                node['line_end'] = last_line
        
        return nodes
    
    def extract_structure_with_pages(self) -> List[Dict]:
        """
        Extract structure with page information from metadata.
        
        Returns:
            List of nodes with page ranges
        """
        if not self.metadata or 'table_of_contents' not in self.metadata:
            self.logger.warning("No table of contents in metadata, falling back to basic structure")
            return self.extract_structure()
        
        toc = self.metadata['table_of_contents']
        nodes = []
        
        for entry in toc:
            node = {
                'title': entry.get('title', ''),
                'page': entry.get('page_id', 0) + 1,  # Convert 0-indexed to 1-indexed
                'heading_level': entry.get('heading_level'),
                'children': []
            }
            nodes.append(node)
        
        # Build hierarchy based on heading levels
        return self._build_hierarchy(nodes)
    
    def _build_hierarchy(self, flat_nodes: List[Dict]) -> List[Dict]:
        """Build hierarchical tree from flat list of nodes."""
        if not flat_nodes:
            return []
        
        root_nodes = []
        stack = []
        
        for node in flat_nodes:
            level = node.get('heading_level')
            
            if level is None:
                # No level info, treat as root
                root_nodes.append(node)
                continue
            
            # Pop stack until we find parent
            while stack and stack[-1]['heading_level'] >= level:
                stack.pop()
            
            if stack:
                # Add as child to parent
                stack[-1]['children'].append(node)
            else:
                # Root level node
                root_nodes.append(node)
            
            stack.append(node)
        
        return root_nodes
    
    def extract_text_by_page(self) -> Dict[int, str]:
        """
        Extract text content organized by page.
        
        Returns:
            Dictionary mapping page numbers to text content
        """
        if not self.metadata or 'page_stats' not in self.metadata:
            self.logger.warning("No page stats in metadata")
            return {}
        
        page_texts = {}
        current_page = 0
        current_text = []
        
        # Simple heuristic: split by page markers or use metadata
        for line in self.lines:
            # Check for page markers (common in marker output)
            page_marker = re.match(r'<!-- Page (\d+) -->', line)
            if page_marker:
                if current_text:
                    page_texts[current_page] = '\n'.join(current_text)
                current_page = int(page_marker.group(1))
                current_text = []
            else:
                current_text.append(line)
        
        # Add last page
        if current_text:
            page_texts[current_page] = '\n'.join(current_text)
        
        return page_texts
    
    def convert_to_pageindex_format(self) -> List[Dict]:
        """
        Convert markdown structure to PageIndex tree format.
        
        Returns:
            Tree structure compatible with PageIndex
        """
        md_nodes = self.extract_structure_with_pages()
        return self._convert_nodes_recursive(md_nodes)
    
    def _convert_nodes_recursive(self, nodes: List[Dict], parent_page: int = 1) -> List[Dict]:
        """Recursively convert nodes to PageIndex format."""
        result = []
        
        for i, node in enumerate(nodes):
            start_page = node.get('page', parent_page)
            
            # Determine end page
            if i + 1 < len(nodes):
                end_page = nodes[i + 1].get('page', start_page + 1) - 1
            else:
                end_page = start_page  # Will be updated by parent
            
            pageindex_node = {
                'title': node['title'],
                'start_page': start_page,
                'end_page': end_page,
                'type': 'section'
            }
            
            # Process children
            if node.get('children'):
                pageindex_node['children'] = self._convert_nodes_recursive(
                    node['children'], 
                    start_page
                )
                # Update end page based on last child
                if pageindex_node['children']:
                    last_child = pageindex_node['children'][-1]
                    pageindex_node['end_page'] = last_child['end_page']
            
            result.append(pageindex_node)
        
        return result
    
    def get_page_count(self) -> int:
        """Get total number of pages from metadata."""
        if self.metadata and 'page_stats' in self.metadata:
            return len(self.metadata['page_stats'])
        return 0
    
    def extract_figures_and_tables(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Extract figure and table references from markdown.
        
        Returns:
            Tuple of (figures, tables) lists
        """
        figures = []
        tables = []
        
        # Pattern for figure references: ![](_page_X_Figure_Y.jpeg)
        figure_pattern = re.compile(r'!\[\]\(_page_(\d+)_Figure_(\d+)\.(jpeg|jpg|png)\)')
        
        # Pattern for table markers
        table_pattern = re.compile(r'^\|.*\|$')  # Simple markdown table detection
        
        in_table = False
        table_start_line = None
        
        for line_num, line in enumerate(self.lines, 1):
            # Check for figures
            fig_match = figure_pattern.search(line)
            if fig_match:
                page_num = int(fig_match.group(1))
                fig_num = fig_match.group(2)
                figures.append({
                    'page': page_num + 1,  # Convert to 1-indexed
                    'figure_number': fig_num,
                    'line': line_num,
                    'type': 'figure'
                })
            
            # Check for tables
            if table_pattern.match(line.strip()):
                if not in_table:
                    in_table = True
                    table_start_line = line_num
            elif in_table and not line.strip():
                # End of table
                tables.append({
                    'line_start': table_start_line,
                    'line_end': line_num - 1,
                    'type': 'table'
                })
                in_table = False
        
        return figures, tables


def process_markdown_to_tree(markdown_path: str, 
                             metadata_path: Optional[str] = None,
                             logger: Optional[logging.Logger] = None) -> Dict:
    """
    Process markdown file and return PageIndex-compatible tree structure.
    
    Args:
        markdown_path: Path to markdown file
        metadata_path: Optional path to metadata JSON
        logger: Optional logger
    
    Returns:
        Dictionary with tree structure and metadata
    """
    processor = MarkdownProcessor(markdown_path, metadata_path, logger)
    
    tree = processor.convert_to_pageindex_format()
    figures, tables = processor.extract_figures_and_tables()
    
    return {
        'tree': tree,
        'figures': figures,
        'tables': tables,
        'page_count': processor.get_page_count(),
        'source': 'markdown'
    }
