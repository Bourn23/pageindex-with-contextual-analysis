#!/usr/bin/env python3
"""
Visualize markdown structure as an interactive HTML page.
Updated to organize keywords by unique 'kw_xxxx' node_ids.
"""

import json
import argparse
from pathlib import Path
from html import escape


def generate_html(structure_data, output_path):
    """Generate an interactive HTML visualization of the structure."""
    
    doc_name = structure_data.get('doc_name', 'Document')
    title = structure_data.get('title', doc_name)
    structure = structure_data.get('structure', [])
    
    # NOTE: In the f-string below, all CSS/JS curly braces are doubled {{ }} to escape them.
    # Python variables {title}, {generate_tree_html(structure)}, etc. use single braces.
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{escape(title)} - Markdown Structure Visualization</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; color: #333; }}
        
        /* Header */
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header h1 {{ font-size: 1.8rem; margin-bottom: 0.5rem; }}
        .header .subtitle {{ opacity: 0.9; font-size: 0.9rem; }}
        
        /* Layout */
        .container {{ display: flex; height: calc(100vh - 120px); }}
        .sidebar {{ width: 400px; background: white; border-right: 1px solid #e0e0e0; overflow-y: auto; padding: 1rem; }}
        .keyword-panel {{ width: 350px; background: white; border-right: 1px solid #e0e0e0; overflow-y: auto; padding: 1rem; display: none; }}
        .keyword-panel.visible {{ display: block; }}
        .content {{ flex: 1; overflow-y: auto; padding: 2rem; background: white; margin: 1rem; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }}
        
        /* Tree Nodes */
        .tree-node {{ margin-left: 1rem; border-left: 2px solid #e0e0e0; padding-left: 0.5rem; }}
        .tree-node.root {{ margin-left: 0; border-left: none; padding-left: 0; }}
        .tree-node.section {{ border-left: 3px solid #4caf50; background: #f1f8e9; border-radius: 4px; margin: 0.3rem 0; }}
        .tree-node.semantic_group {{ border-left: 2px solid #2196f3; background: #e3f2fd; border-radius: 3px; margin: 0.2rem 0; }}
        .tree-node.sentence {{ border-left: 2px solid #ff9800; background: #fff8e1; border-radius: 3px; margin: 0.2rem 0; }}
        
        /* Node Titles */
        .node-title {{ padding: 0.6rem; margin: 0.25rem 0; cursor: pointer; border-radius: 4px; transition: all 0.2s; display: flex; align-items: center; gap: 0.5rem; font-size: 0.9rem; }}
        .node-title:hover {{ background: rgba(0,0,0,0.05); transform: translateX(2px); }}
        .node-title.active {{ background: #667eea; color: white; font-weight: 500; }}
        
        /* Specific Colors */
        .node-title.section {{ font-weight: 500; color: #2e7d32; }}
        .node-title.section.active {{ background: #4caf50; color: white; }}
        .node-title.semantic_group {{ font-size: 0.85rem; color: #1976d2; }}
        .node-title.semantic_group.active {{ background: #2196f3; color: white; }}
        .node-title.sentence {{ font-size: 0.8rem; color: #e65100; }}
        .node-title.sentence.active {{ background: #ff9800; color: white; }}
        
        .node-badge {{ font-size: 0.7rem; padding: 0.15rem 0.4rem; border-radius: 3px; background: rgba(0,0,0,0.1); margin-left: auto; font-weight: 500; }}
        
        /* Chips */
        .keyword-chips {{ display: flex; flex-wrap: wrap; gap: 0.3rem; margin-top: 0.5rem; padding-left: 2rem; }}
        .keyword-chip {{ background: #e8f5e9; color: #2e7d32; padding: 0.2rem 0.5rem; border-radius: 12px; font-size: 0.7rem; cursor: pointer; transition: all 0.2s; border: 1px solid #c8e6c9; }}
        .keyword-chip:hover {{ background: #c8e6c9; transform: scale(1.05); }}
        .keyword-chip.highlighted {{ background: #fff59d; color: #f57f17; border-color: #f57f17; box-shadow: 0 0 5px rgba(245, 127, 23, 0.5); }}
        
        /* Sidebar List */
        .keyword-item {{ padding: 0.8rem; margin: 0.3rem 0; cursor: pointer; border-radius: 4px; border-left: 3px solid #8bc34a; background: #f1f8e9; transition: all 0.2s; }}
        .keyword-item:hover {{ background: #dcedc8; transform: translateX(2px); }}
        .keyword-item.active {{ background: #689f38; color: white; border-left-color: #33691e; }}
        .keyword-item.multi-ref {{ border-left-color: #ff9800; background: #fff8e1; }}
        .keyword-item.multi-ref.active {{ background: #f57c00; border-left-color: #e65100; }}
        
        .keyword-term {{ font-weight: 600; font-size: 0.95rem; margin-bottom: 0.3rem; display: flex; align-items: center; gap: 0.5rem; }}
        .keyword-id-badge {{ font-family: monospace; font-size: 0.7rem; opacity: 0.7; border: 1px solid currentColor; padding: 0 4px; border-radius: 4px; }}
        .keyword-count {{ background: rgba(0,0,0,0.1); padding: 0.2rem 0.5rem; border-radius: 12px; font-size: 0.75rem; font-weight: 600; margin-left: auto; }}
        
        /* Controls */
        .keyword-search {{ width: 100%; padding: 0.6rem; border: 1px solid #e0e0e0; border-radius: 4px; margin-bottom: 1rem; font-size: 0.9rem; }}
        .toggle-keywords {{ position: fixed; right: 1rem; top: 1rem; background: #8bc34a; color: white; border: none; padding: 0.6rem 1rem; border-radius: 4px; cursor: pointer; font-weight: 500; box-shadow: 0 2px 8px rgba(0,0,0,0.2); z-index: 1000; }}
        
        /* Details */
        .metadata {{ display: flex; gap: 1rem; margin-bottom: 1rem; flex-wrap: wrap; }}
        .metadata-item {{ background: #f5f5f5; padding: 0.5rem 1rem; border-radius: 4px; font-size: 0.9rem; }}
        .text-content {{ background: #f9f9f9; padding: 1.5rem; border-radius: 6px; border-left: 4px solid #667eea; line-height: 1.6; font-family: 'Georgia', serif; margin-bottom: 1rem; }}
        .keyword-detail {{ background: #e8f5e9; padding: 1.5rem; border-radius: 6px; border-left: 4px solid #4caf50; margin-bottom: 1rem; }}
        
        .occurrence-item {{ background: #f9f9f9; padding: 1rem; margin: 0.5rem 0; border-radius: 4px; border-left: 3px solid #8bc34a; cursor: pointer; }}
        .occurrence-item:hover {{ background: #f1f8e9; }}
        
        ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
        ::-webkit-scrollbar-track {{ background: #f1f1f1; }}
        ::-webkit-scrollbar-thumb {{ background: #888; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📄 {escape(title)}</h1>
        <div class="subtitle">Markdown Structure Visualization • <span id="stats-summary">Loading...</span></div>
    </div>
    
    <button class="toggle-keywords" onclick="toggleKeywordPanel()">
        🔑 Keywords (<span id="keyword-count">0</span>)
    </button>
    
    <div class="container">
        <div class="sidebar" id="sidebar">
            <h3 style="margin-bottom: 1rem; color: #667eea;">Document Structure</h3>
            {generate_tree_html(structure)}
        </div>
        
        <div class="keyword-panel" id="keyword-panel">
            <h3 style="margin-bottom: 1rem; color: #8bc34a;">Keywords Index (by ID)</h3>
            <input type="text" class="keyword-search" id="keyword-search" placeholder="Search keywords...">
            <div id="keyword-list"></div>
        </div>
        
        <div class="content" id="content">
            <div class="empty-state" style="color:#999; text-align:center; padding:2rem;">
                👈 Select a node from the tree to view its details
            </div>
        </div>
    </div>
    
    <script>
        const nodeData = {json.dumps(structure, ensure_ascii=False)};
        
        // Index uses node_id as the key
        let keywordIndex = {{}}; 
        let currentKeywordId = null;
        
        function buildKeywordIndex() {{
            keywordIndex = {{}};
            
            
            function traverse(nodes, path = [], parentInfo = {{}}) {{
                nodes.forEach((node, idx) => {{
                    const currentPath = [...path, idx];
                    
                    // Track lineage and line number
                    const effectiveLine = node.line_num || parentInfo.line_num;

                    const currentParentInfo = {{
                        section: parentInfo.section || (node.node_type === 'section' ? node.title : parentInfo.section),
                        semantic_group: parentInfo.semantic_group || (node.node_type === 'semantic_group' ? node.title : parentInfo.semantic_group),
                        sentence: parentInfo.sentence || (node.node_type === 'sentence' ? node.title : parentInfo.sentence),
                        line_num: effectiveLine
                    }};
                    
                    if (node.node_type === 'keyword') {{
                        // Use node_id as the key
                        const kwId = node.node_id;
                        
                        if (kwId) {{
                            if (!keywordIndex[kwId]) {{
                                keywordIndex[kwId] = [];
                            }}
                            
                            keywordIndex[kwId].push({{
                                node: node,
                                path: currentPath,
                                parent: currentParentInfo.section || 'Unknown Section',
                                sentence: currentParentInfo.sentence || 'Unknown Sentence',
                                line_num: effectiveLine
                            }});
                        }}
                    }}
                    
                    if (node.nodes && node.nodes.length > 0) {{
                        traverse(node.nodes, currentPath, currentParentInfo);
                    }}
                }});
            }}
            
            traverse(nodeData);
            
            // Stats
            const uniqueIds = Object.keys(keywordIndex).length;
            const totalOccurrences = Object.values(keywordIndex).reduce((sum, list) => sum + list.length, 0);
            
            document.getElementById('keyword-count').textContent = uniqueIds;
            document.getElementById('stats-summary').textContent = `${{uniqueIds}} unique keyword IDs • ${{totalOccurrences}} occurrences`;
            
            renderKeywordList();
        }}
        
        function renderKeywordList(searchFilter = '') {{
            const listEl = document.getElementById('keyword-list');
            
            // Convert to array for sorting
            let kwArray = Object.entries(keywordIndex).map(([id, occurrences]) => {{
                // Use the first occurrence to get the "canonical" title for this ID
                const primaryNode = occurrences[0].node;
                const term = primaryNode.metadata?.term || primaryNode.title || "Unknown";
                return {{
                    id: id,
                    term: term,
                    summary: primaryNode.summary,
                    occurrences: occurrences
                }};
            }});
            
            // Sort alphabetically by term
            kwArray.sort((a, b) => a.term.localeCompare(b.term));
            
            // Filter
            if (searchFilter) {{
                const lower = searchFilter.toLowerCase();
                kwArray = kwArray.filter(k => 
                    k.term.toLowerCase().includes(lower) || 
                    k.id.toLowerCase().includes(lower)
                );
            }}
            
            listEl.innerHTML = kwArray.map(k => {{
                const count = k.occurrences.length;
                const isMulti = count > 1;
                
                return `
                    <div class="keyword-item ${{isMulti ? 'multi-ref' : ''}}" 
                         onclick="showKeywordGroup('${{k.id}}')" 
                         data-kw-id="${{k.id}}">
                        <div class="keyword-term">
                            ${{escapeHtml(k.term)}}
                            <span class="keyword-id-badge">${{k.id}}</span>
                            <div class="keyword-count">${{count}}</div>
                        </div>
                        <div class="keyword-summary">${{escapeHtml(k.summary || '')}}</div>
                    </div>
                `;
            }}).join('');
        }}
        
        function toggleKeywordPanel() {{
            document.getElementById('keyword-panel').classList.toggle('visible');
        }}
        
        // Show details for a specific Keyword ID (Aggregate View)
        function showKeywordGroup(kwId) {{
            currentKeywordId = kwId;
            const group = keywordIndex[kwId];
            if (!group) return;
            
            const firstNode = group[0].node;
            const term = firstNode.metadata?.term || firstNode.title;
            
            // 1. Highlight Sidebar Item
            document.querySelectorAll('.keyword-item').forEach(el => el.classList.remove('active'));
            document.querySelector(`.keyword-item[data-kw-id="${{kwId}}"]`)?.classList.add('active');
            
            // 2. Highlight Tree Chips
            document.querySelectorAll('.keyword-chip').forEach(el => {{
                el.classList.remove('highlighted');
                if (el.dataset.kwId === kwId) {{
                    el.classList.add('highlighted');
                }}
            }});
            
            // 3. Render Detail View
            const content = document.getElementById('content');
            content.innerHTML = `
                <div class="detail-section">
                    <h2>🔑 ${{escapeHtml(term)}}</h2>
                    
                    <div class="metadata">
                        <div class="metadata-item"><strong>ID:</strong> ${{kwId}}</div>
                        <div class="metadata-item"><strong>Occurrences:</strong> ${{group.length}}</div>
                    </div>
                    
                    <div class="keyword-detail">
                        <h4>Definition</h4>
                        <p>${{escapeHtml(firstNode.summary || firstNode.metadata?.summary || 'No summary.')}}</p>
                        ${{firstNode.metadata?.relevance ? `<br><strong>Relevance:</strong> ${{escapeHtml(firstNode.metadata.relevance)}}` : ''}}
                    </div>
                    
                    <div class="occurrence-list">
                        <h3>Occurrences in Document</h3>
                        ${{group.map((occ, idx) => `
                            <div class="occurrence-item" onclick="scrollToNodeInTree('${{occ.node.node_id}}')">
                                <div style="font-size:0.85rem; color:#666; margin-bottom:0.5rem;">
                                    <strong>#${{idx + 1}}</strong> • Line ${{occ.line_num || '?'}} • ${{escapeHtml(occ.parent)}}
                                </div>
                                <div style="font-size:0.95rem;">
                                    "${{escapeHtml(occ.sentence)}}"
                                </div>
                            </div>
                        `).join('')}}
                    </div>
                </div>
            `;
        }}
        
        // Generic Node Details (Section, Sentence, etc.)
        function showNodeDetails(nodeId) {{
            const node = findNode(nodeData, nodeId);
            if (!node) return;
            
            // If it's a keyword, redirect to the Group view
            if (node.node_type === 'keyword') {{
                showKeywordGroup(node.node_id);
                return;
            }}
            
            // Clear highlights if switching to non-keyword
            document.querySelectorAll('.keyword-chip').forEach(el => el.classList.remove('highlighted'));
            document.querySelectorAll('.keyword-item').forEach(el => el.classList.remove('active'));
            
            // Highlight Tree Title
            document.querySelectorAll('.node-title').forEach(el => el.classList.remove('active'));
            document.querySelector(`.node-title[data-node-id="${{nodeId}}"]`)?.classList.add('active');
            
            const content = document.getElementById('content');
            content.innerHTML = renderStandardNode(node);
        }}
        
        function renderStandardNode(node) {{
            return `
                <div class="detail-section">
                    <h2>${{escapeHtml(node.title)}}</h2>
                    <div class="metadata">
                        <div class="metadata-item"><strong>Type:</strong> ${{node.node_type}}</div>
                        <div class="metadata-item"><strong>ID:</strong> ${{node.node_id}}</div>
                        ${{node.line_num ? `<div class="metadata-item"><strong>Line:</strong> ${{node.line_num}}</div>` : ''}}
                    </div>
                    
                    ${{node.text ? `<div class="text-content">${{escapeHtml(node.text)}}</div>` : ''}}
                    
                    ${{node.nodes && node.nodes.length ? `
                        <div style="margin-top:1rem; background:#e3f2fd; padding:1rem; border-radius:4px;">
                            <strong>Contains ${{node.nodes.length}} child nodes.</strong>
                        </div>
                    ` : ''}}
                </div>
            `;
        }}
        
        function findNode(nodes, nodeId) {{
            for (const node of nodes) {{
                if (node.node_id === nodeId) return node;
                if (node.nodes) {{
                    const found = findNode(node.nodes, nodeId);
                    if (found) return found;
                }}
            }}
            return null;
        }}
        
        function scrollToNodeInTree(nodeId) {{
            const el = document.querySelector(`[data-node-id="${{nodeId}}"]`) || 
                       document.querySelector(`.keyword-chip[onclick*="${{nodeId}}"]`);
            if (el) {{
                el.scrollIntoView({{behavior: 'smooth', block: 'center'}});
                el.classList.add('highlighted');
                setTimeout(() => el.classList.remove('highlighted'), 1000);
            }}
        }}

        function escapeHtml(text) {{
            if (!text) return '';
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }}
        
        document.addEventListener('DOMContentLoaded', () => {{
            buildKeywordIndex();
            document.getElementById('keyword-search').addEventListener('input', (e) => renderKeywordList(e.target.value));
        }});
    </script>
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)


def generate_tree_html(nodes, level=0):
    """Generate HTML for the tree structure."""
    if not nodes:
        return ""
    
    html = ""
    for node in nodes:
        node_id = node.get('node_id', 'unknown')
        title = node.get('title', 'Untitled')
        line_num = node.get('line_num', '')
        node_type = node.get('node_type', '')
        children = node.get('nodes', [])
        
        # Determine node class and icon
        if node_type == 'section':
            icon = '📂'; tree_class = 'section'; node_class = 'section'
        elif node_type == 'semantic_group':
            icon = '🧩'; tree_class = 'semantic_group'; node_class = 'semantic_group'
        elif node_type == 'sentence':
            icon = '📝'; tree_class = 'sentence'; node_class = 'sentence'
        elif node_type == 'keyword':
            icon = '🔑'; tree_class = 'keyword'; node_class = 'keyword'
        else:
            icon = '📄'; tree_class = 'root'; node_class = 'root'
        
        display_title = title if len(title) <= 60 else title[:57] + "..."
        
        html += f'<div class="tree-node {tree_class}">'
        html += f'<div class="node-title {node_class}" data-node-id="{escape(node_id)}" onclick="showNodeDetails(\'{escape(node_id)}\')">'
        html += f'<span class="node-icon">{icon}</span>'
        html += f'<span>{escape(display_title)}</span>'
        
        if line_num:
            html += f'<span class="node-badge">L{line_num}</span>'
        
        html += '</div>'
        
        # Special handling for Sentences to show Keyword Chips
        if node_type == 'sentence' and children:
            keyword_children = [child for child in children if child.get('node_type') == 'keyword']
            if keyword_children:
                html += '<div class="keyword-chips">'
                for kw in keyword_children:
                    kw_title = kw.get('title', 'Unknown')
                    kw_id = kw.get('node_id', 'unknown')
                    kw_summary = kw.get('summary', '')
                    
                    # IMPORTANT: data-kw-id allows us to find all instances of this ID in the tree
                    html += f'''<span class="keyword-chip" 
                                      onclick="showNodeDetails('{escape(kw_id)}')" 
                                      title="{escape(kw_summary)}" 
                                      data-kw-id="{escape(kw_id)}">
                                    {escape(kw_title)}
                                </span>'''
                html += '</div>'
        
        # Recursively render children (skipping keywords as they are rendered as chips)
        non_keyword_children = [child for child in children if child.get('node_type') != 'keyword']
        if non_keyword_children:
            html += generate_tree_html(non_keyword_children, level + 1)
        
        html += '</div>'
    
    return html


def main():
    parser = argparse.ArgumentParser(description='Visualize structure with ID-based keyword grouping')
    parser.add_argument('structure_json', help='Path to structure JSON file')
    parser.add_argument('--output', '-o', help='Output HTML file path')
    
    args = parser.parse_args()
    
    with open(args.structure_json, 'r', encoding='utf-8') as f:
        structure_data = json.load(f)
    
    if args.output:
        output_path = args.output
    else:
        input_path = Path(args.structure_json)
        output_path = input_path.with_suffix('.html')
    
    generate_html(structure_data, output_path)
    
    # print(f"✓ Visualization saved to: {output_path}")
    print(f"  Open in browser: file://{Path(output_path).absolute()}")


if __name__ == '__main__':
    main()