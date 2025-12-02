#!/usr/bin/env python3
"""
Process markdown with full granular features (semantic units, keywords).

This script:
1. Extracts structure from markdown headers (fast, accurate)
2. Applies semantic subdivision to each section
3. Extracts keywords from leaf nodes
"""

import asyncio
import json
from pathlib import Path
from pageindex.page_index_md import md_to_tree
from pageindex.markdown_adapter import markdown_to_page_list
from pageindex.granular.semantic_analyzer import SemanticAnalyzer
from pageindex.utils import ConfigLoader, JsonLogger, write_node_id
import logging


async def apply_granular_to_markdown(structure, md_path, metadata_path, opt):
    """Apply granular features to markdown structure."""
    
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    
    # Get page list for text extraction
    page_list = markdown_to_page_list(md_path, metadata_path, opt.model)
    
    # Initialize semantic analyzer
    analyzer = SemanticAnalyzer(opt.llm_client, logger=logger)
    
    granularity = getattr(opt, 'granularity', 'coarse')
    
    async def process_node(node, depth=0):
        """Recursively process nodes."""
        
        # Skip if already processed or no text
        if not node.get('text'):
            return
        
        # Determine if we should subdivide this node
        should_subdivide = False
        
        if granularity == 'medium' and depth == 0:
            should_subdivide = True  # Subdivide top-level sections
        elif granularity in ['fine', 'keywords'] and depth <= 1:
            should_subdivide = True  # Subdivide sections and semantic units
        
        if should_subdivide:
            # Create a mock section node for the analyzer
            section_node = {
                'title': node.get('title', ''),
                'text': node.get('text', ''),
                'start_index': 1,  # Mock page index
                'end_index': 1
            }
            
            # Analyze section
            try:
                semantic_units = analyzer.analyze_section(
                    section_node,
                    page_texts=[(node.get('text', ''), 0)],
                    min_pages=0  # No minimum for markdown
                )
                
                if semantic_units:
                    logger.info(f"Found {len(semantic_units)} semantic units in '{node['title']}'")
                    
                    # Create child nodes from semantic units
                    child_nodes = analyzer.create_nodes_from_semantic_units(
                        semantic_units,
                        section_node,
                        page_texts=[(node.get('text', ''), 0)]
                    )
                    
                    # Add to node
                    if 'nodes' not in node:
                        node['nodes'] = []
                    node['nodes'].extend(child_nodes)
                    
                    # Extract keywords if at keywords granularity
                    if granularity == 'keywords':
                        for child in child_nodes:
                            if child.get('node_type') == 'semantic_unit':
                                keywords = analyzer.extract_keywords(child)
                                if keywords:
                                    keyword_nodes = analyzer.create_keyword_nodes(keywords, child)
                                    if 'nodes' not in child:
                                        child['nodes'] = []
                                    child['nodes'].extend(keyword_nodes)
                                    logger.info(f"  Added {len(keyword_nodes)} keywords to '{child['title']}'")
            
            except Exception as e:
                logger.error(f"Error processing '{node.get('title')}': {e}")
        
        # Recursively process existing children
        if 'nodes' in node:
            for child in node['nodes']:
                await process_node(child, depth + 1)
    
    # Process all top-level nodes
    if isinstance(structure, list):
        for node in structure:
            await process_node(node)
    elif isinstance(structure, dict) and 'nodes' in structure:
        for node in structure['nodes']:
            await process_node(node)


async def main():
    md_file = "tests/markdowns/New Insights into the Compositional Dependence of Li-Ion Transport in polymer-ceramic composite electrolytes/New Insights into the Compositional Dependence of Li-Ion Transport in polymer-ceramic composite electrolytes.md"
    meta_file = "tests/markdowns/New Insights into the Compositional Dependence of Li-Ion Transport in polymer-ceramic composite electrolytes/New Insights into the Compositional Dependence of Li-Ion Transport in polymer-ceramic composite electrolytes_meta.json"
    
    print("="*70)
    print("Processing Markdown with Granular Features")
    print("="*70)
    print(f"Input: {Path(md_file).name}")
    print()
    
    # Step 1: Extract structure from markdown headers
    print("Step 1: Extracting structure from markdown headers...")
    result = await md_to_tree(
        md_file,
        if_add_node_text='yes',
        if_add_node_id='yes',
        if_add_node_summary='no'
    )
    
    structure = result.get('structure', [])
    doc_name = result.get('doc_name', Path(md_file).stem)
    
    def count_nodes(nodes):
        count = len(nodes)
        for n in nodes:
            if 'nodes' in n:
                count += count_nodes(n['nodes'])
        return count
    
    print(f"✓ Found {count_nodes(structure)} nodes from headers")
    
    # Step 2: Apply granular features
    print("\nStep 2: Applying granular features...")
    
    opt = ConfigLoader().load({
        'model': 'gemini-2.5-flash-lite',
        'granularity': 'keywords',
        'enable_semantic_subdivision': True,
        'semantic_min_pages': 0
    })
    
    # Initialize LLM client
    from pageindex.llm_client import get_llm_client
    opt.llm_client = get_llm_client(provider='gemini')
    
    await apply_granular_to_markdown(structure, md_file, meta_file, opt)
    
    print(f"✓ Total nodes after granular processing: {count_nodes(structure)}")
    
    # Step 3: Reassign node IDs
    write_node_id(structure)
    
    # Step 4: Save result
    output_path = Path("results") / f"{doc_name}_markdown_granular.json"
    output_data = {
        'doc_name': doc_name,
        'structure': structure,
        'source': 'markdown_with_granular'
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved to: {output_path}")
    
    # Print statistics
    def count_by_type(nodes, counts=None):
        if counts is None:
            counts = {}
        for n in nodes:
            node_type = n.get('node_type', 'section')
            counts[node_type] = counts.get(node_type, 0) + 1
            if 'nodes' in n:
                count_by_type(n['nodes'], counts)
        return counts
    
    counts = count_by_type(structure)
    print("\nNode types:")
    for node_type, count in sorted(counts.items()):
        print(f"  {node_type}: {count}")


if __name__ == '__main__':
    asyncio.run(main())
