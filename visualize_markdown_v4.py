#!/usr/bin/env python3
"""
Visualize markdown_v3.py structure as an interactive HTML page.
Enhanced version with proper keyword indexing and duplicate detection.
"""

import json
import argparse
from pathlib import Path
from html import escape


def generate_html(structure_data, output_path):
    """Generate an interactive HTML visualization of the markdown_v3 structure."""
    
    doc_name = structure_data.get('doc_name', 'Document')
    title = structure_data.get('title', doc_name)
    structure = structure_data.get('structure', [])
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{escape(title)} - Markdown Structure Visualization</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
            color: #333;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .header h1 {{
            font-size: 1.8rem;
            margin-bottom: 0.5rem;
        }}
        
        .header .subtitle {{
            opacity: 0.9;
            font-size: 0.9rem;
        }}
        
        .container {{
            display: flex;
            height: calc(100vh - 120px);
        }}
        
        .sidebar {{
            width: 400px;
            background: white;
            border-right: 1px solid #e0e0e0;
            overflow-y: auto;
            padding: 1rem;
        }}
        
        .keyword-panel {{
            width: 350px;
            background: white;
            border-right: 1px solid #e0e0e0;
            overflow-y: auto;
            padding: 1rem;
            display: none;
        }}
        
        .keyword-panel.visible {{
            display: block;
        }}
        
        .content {{
            flex: 1;
            overflow-y: auto;
            padding: 2rem;
            background: white;
            margin: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }}
        
        .tree-node {{
            margin-left: 1rem;
            border-left: 2px solid #e0e0e0;
            padding-left: 0.5rem;
        }}
        
        .tree-node.root {{
            margin-left: 0;
            border-left: none;
            padding-left: 0;
        }}
        
        .tree-node.section {{
            border-left: 3px solid #4caf50;
            background: #f1f8e9;
            border-radius: 4px;
            margin: 0.3rem 0;
        }}
        
        .tree-node.semantic_group {{
            border-left: 2px solid #2196f3;
            background: #e3f2fd;
            border-radius: 3px;
            margin: 0.2rem 0;
        }}
        
        .tree-node.sentence {{
            border-left: 2px solid #ff9800;
            background: #fff8e1;
            border-radius: 3px;
            margin: 0.2rem 0;
        }}
        
        .node-title {{
            padding: 0.6rem;
            margin: 0.25rem 0;
            cursor: pointer;
            border-radius: 4px;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.9rem;
        }}
        
        .node-title:hover {{
            background: rgba(0,0,0,0.05);
            transform: translateX(2px);
        }}
        
        .node-title.active {{
            background: #667eea;
            color: white;
            font-weight: 500;
        }}
        
        .node-title.section {{
            font-weight: 500;
            color: #2e7d32;
        }}
        
        .node-title.section.active {{
            background: #4caf50;
            color: white;
        }}
        
        .node-title.semantic_group {{
            font-size: 0.85rem;
            color: #1976d2;
        }}
        
        .node-title.semantic_group.active {{
            background: #2196f3;
            color: white;
        }}
        
        .node-title.sentence {{
            font-size: 0.8rem;
            color: #e65100;
        }}
        
        .node-title.sentence.active {{
            background: #ff9800;
            color: white;
        }}
        
        .node-icon {{
            font-size: 0.9rem;
            opacity: 0.8;
        }}
        
        .node-badge {{
            font-size: 0.7rem;
            padding: 0.15rem 0.4rem;
            border-radius: 3px;
            background: rgba(0,0,0,0.1);
            margin-left: auto;
            font-weight: 500;
        }}
        
        .node-title.active .node-badge {{
            background: rgba(255,255,255,0.3);
        }}
        
        .keyword-chips {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.3rem;
            margin-top: 0.5rem;
            padding-left: 2rem;
        }}
        
        .keyword-chip {{
            background: #e8f5e9;
            color: #2e7d32;
            padding: 0.2rem 0.5rem;
            border-radius: 12px;
            font-size: 0.7rem;
            cursor: pointer;
            transition: all 0.2s;
            border: 1px solid #c8e6c9;
        }}
        
        .keyword-chip:hover {{
            background: #c8e6c9;
            transform: scale(1.05);
        }}
        
        .keyword-chip.highlighted {{
            background: #fff59d;
            color: #f57f17;
            border-color: #f57f17;
            animation: pulse 1s ease-in-out;
        }}
        
        .keyword-chip.duplicate {{
            background: #ffecb3;
            color: #e65100;
            border-color: #ffb74d;
        }}
        
        @keyframes pulse {{
            0%, 100% {{ transform: scale(1); }}
            50% {{ transform: scale(1.05); }}
        }}
        
        .keyword-item {{
            padding: 0.8rem;
            margin: 0.3rem 0;
            cursor: pointer;
            border-radius: 4px;
            border-left: 3px solid #8bc34a;
            background: #f1f8e9;
            transition: all 0.2s;
        }}
        
        .keyword-item:hover {{
            background: #dcedc8;
            transform: translateX(2px);
        }}
        
        .keyword-item.active {{
            background: #689f38;
            color: white;
            border-left-color: #33691e;
        }}
        
        .keyword-item.duplicate {{
            border-left-color: #ff9800;
            background: #fff8e1;
        }}
        
        .keyword-item.duplicate.active {{
            background: #f57c00;
            border-left-color: #e65100;
        }}
        
        .keyword-term {{
            font-weight: 600;
            font-size: 0.95rem;
            margin-bottom: 0.3rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        .keyword-summary {{
            font-size: 0.8rem;
            opacity: 0.8;
            line-height: 1.3;
        }}
        
        .keyword-count {{
            background: rgba(0,0,0,0.1);
            padding: 0.2rem 0.5rem;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-left: auto;
        }}
        
        .keyword-item.active .keyword-count {{
            background: rgba(255,255,255,0.3);
        }}
        
        .duplicate-badge {{
            background: #ff9800;
            color: white;
            padding: 0.1rem 0.4rem;
            border-radius: 8px;
            font-size: 0.65rem;
            font-weight: 600;
        }}
        
        .keyword-search {{
            width: 100%;
            padding: 0.6rem;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            margin-bottom: 1rem;
            font-size: 0.9rem;
        }}
        
        .keyword-search:focus {{
            outline: none;
            border-color: #8bc34a;
        }}
        
        .keyword-filters {{
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }}
        
        .filter-btn {{
            padding: 0.4rem 0.8rem;
            border: 1px solid #e0e0e0;
            background: white;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.8rem;
            transition: all 0.2s;
        }}
        
        .filter-btn.active {{
            background: #8bc34a;
            color: white;
            border-color: #689f38;
        }}
        
        .toggle-keywords {{
            position: fixed;
            right: 1rem;
            top: 1rem;
            background: #8bc34a;
            color: white;
            border: none;
            padding: 0.6rem 1rem;
            border-radius: 4px;
            cursor: pointer;
            font-weight: 500;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            z-index: 1000;
            transition: all 0.2s;
        }}
        
        .toggle-keywords:hover {{
            background: #689f38;
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }}
        
        .detail-section {{
            margin-bottom: 2rem;
        }}
        
        .detail-section h2 {{
            font-size: 1.5rem;
            margin-bottom: 1rem;
            color: #667eea;
        }}
        
        .detail-section h3 {{
            font-size: 1rem;
            margin-bottom: 0.5rem;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 600;
        }}
        
        .metadata {{
            display: flex;
            gap: 1rem;
            margin-bottom: 1rem;
            flex-wrap: wrap;
        }}
        
        .metadata-item {{
            background: #f5f5f5;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            font-size: 0.9rem;
        }}
        
        .metadata-label {{
            font-weight: 600;
            color: #666;
        }}
        
        .text-content {{
            background: #f9f9f9;
            padding: 1.5rem;
            border-radius: 6px;
            border-left: 4px solid #667eea;
            line-height: 1.6;
            white-space: pre-wrap;
            font-family: 'Georgia', serif;
            margin-bottom: 1rem;
        }}
        
        .summary-content {{
            background: #fff9e6;
            padding: 1.5rem;
            border-radius: 6px;
            border-left: 4px solid #ffc107;
            line-height: 1.6;
            margin-bottom: 1rem;
        }}
        
        .keyword-detail {{
            background: #e8f5e9;
            padding: 1.5rem;
            border-radius: 6px;
            border-left: 4px solid #4caf50;
            margin-bottom: 1rem;
        }}
        
        .keyword-detail h4 {{
            color: #2e7d32;
            margin-bottom: 0.5rem;
        }}
        
        .relevance-text {{
            background: #fff3e0;
            padding: 1rem;
            border-radius: 4px;
            border-left: 3px solid #ff9800;
            margin-top: 1rem;
            font-style: italic;
        }}
        
        .empty-state {{
            color: #999;
            font-style: italic;
            padding: 2rem;
            text-align: center;
        }}
        
        .stats {{
            display: flex;
            gap: 1rem;
            margin-bottom: 1rem;
            font-size: 0.85rem;
            flex-wrap: wrap;
        }}
        
        .stat {{
            background: #e8eaf6;
            padding: 0.4rem 0.8rem;
            border-radius: 4px;
        }}
        
        .children-info {{
            background: #e3f2fd;
            padding: 1rem;
            border-radius: 4px;
            margin-top: 1rem;
            font-size: 0.9rem;
        }}
        
        .occurrence-list {{
            margin-top: 1rem;
        }}
        
        .occurrence-item {{
            background: #f9f9f9;
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 4px;
            border-left: 3px solid #8bc34a;
            cursor: pointer;
            transition: all 0.2s;
        }}
        
        .occurrence-item:hover {{
            background: #f1f8e9;
            transform: translateX(2px);
        }}
        
        .occurrence-location {{
            font-size: 0.85rem;
            color: #666;
            margin-bottom: 0.5rem;
            font-weight: 500;
        }}
        
        .occurrence-context {{
            font-size: 0.9rem;
            line-height: 1.4;
        }}
        
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: #f1f1f1;
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: #888;
            border-radius: 4px;
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: #555;
        }}
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
            <h3 style="margin-bottom: 1rem; color: #8bc34a;">Keywords Index</h3>
            <input type="text" class="keyword-search" id="keyword-search" placeholder="Search keywords...">
            
            <div class="keyword-filters">
                <button class="filter-btn active" onclick="filterKeywords('all')" id="filter-all">All</button>
                <button class="filter-btn" onclick="filterKeywords('duplicates')" id="filter-duplicates">Duplicates</button>
                <button class="filter-btn" onclick="filterKeywords('unique')" id="filter-unique">Unique</button>
            </div>
            
            <div id="keyword-list"></div>
        </div>
        
        <div class="content" id="content">
            <div class="empty-state">
                👈 Select a node from the tree to view its details
                <br><br>
                <strong>Structure:</strong> Sections → Semantic Groups → Sentences → Keywords
            </div>
        </div>
    </div>
    
    <script>
        const nodeData = {json.dumps(structure, ensure_ascii=False)};
        let keywordIndex = {{}};
        let currentKeyword = null;
        let currentFilter = 'all';
        
        // Build keyword index on load
        function buildKeywordIndex() {{
            keywordIndex = {{}};
            
            function traverse(nodes, path = [], parentInfo = {{}}) {{
                nodes.forEach((node, idx) => {{
                    const currentPath = [...path, idx];
                    const currentParentInfo = {{
                        section: parentInfo.section || (node.node_type === 'section' ? node.title : parentInfo.section),
                        semantic_group: parentInfo.semantic_group || (node.node_type === 'semantic_group' ? node.title : parentInfo.semantic_group),
                        sentence: parentInfo.sentence || (node.node_type === 'sentence' ? node.title : parentInfo.sentence)
                    }};
                    
                    if (node.node_type === 'keyword') {{
                        const term = node.metadata?.term || node.title;
                        if (!keywordIndex[term]) {{
                            keywordIndex[term] = [];
                        }}
                        keywordIndex[term].push({{
                            node: node,
                            path: currentPath,
                            parent: currentParentInfo.section || 'Unknown Section',
                            sentence: currentParentInfo.sentence || 'Unknown Sentence',
                            semantic_group: currentParentInfo.semantic_group || 'Unknown Group'
                        }});
                    }}
                    
                    if (node.nodes && node.nodes.length > 0) {{
                        traverse(node.nodes, currentPath, currentParentInfo);
                    }}
                }});
            }}
            
            traverse(nodeData);
            
            // Update count and find duplicates
            const uniqueKeywords = Object.keys(keywordIndex).length;
            const totalOccurrences = Object.values(keywordIndex).reduce((sum, occurrences) => sum + occurrences.length, 0);
            const duplicateKeywords = Object.entries(keywordIndex).filter(([term, occurrences]) => occurrences.length > 1);
            
            document.getElementById('keyword-count').textContent = `${{uniqueKeywords}} unique, ${{totalOccurrences}} total`;
            document.getElementById('stats-summary').textContent = `${{uniqueKeywords}} keywords • ${{duplicateKeywords.length}} duplicates`;
            
            // Update filter buttons
            document.getElementById('filter-duplicates').textContent = `Duplicates (${{duplicateKeywords.length}})`;
            document.getElementById('filter-unique').textContent = `Unique (${{uniqueKeywords - duplicateKeywords.length}})`;
            
            // Render keyword list
            renderKeywordList();
            
            console.log(`Found ${{uniqueKeywords}} unique keywords with ${{totalOccurrences}} total occurrences`);
            console.log(`${{duplicateKeywords.length}} keywords appear multiple times:`, duplicateKeywords.map(([term, occ]) => `${{term}} (${{occ.length}}x)`));
        }}
        
        function filterKeywords(type) {{
            currentFilter = type;
            
            // Update active filter button
            document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
            document.getElementById(`filter-${{type}}`).classList.add('active');
            
            renderKeywordList();
        }}
        
        function renderKeywordList(searchFilter = '') {{
            const keywordList = document.getElementById('keyword-list');
            let keywords = Object.keys(keywordIndex).sort();
            
            // Apply filter
            if (currentFilter === 'duplicates') {{
                keywords = keywords.filter(term => keywordIndex[term].length > 1);
            }} else if (currentFilter === 'unique') {{
                keywords = keywords.filter(term => keywordIndex[term].length === 1);
            }}
            
            // Apply search filter
            if (searchFilter) {{
                keywords = keywords.filter(k => k.toLowerCase().includes(searchFilter.toLowerCase()));
            }}
            
            keywordList.innerHTML = keywords.map(term => {{
                const occurrences = keywordIndex[term];
                const firstOcc = occurrences[0];
                const isDuplicate = occurrences.length > 1;
                
                return `
                    <div class="keyword-item ${{isDuplicate ? 'duplicate' : ''}}" onclick="showKeywordOccurrences('${{term.replace(/'/g, "\\\\'")}}')" data-keyword="${{escapeHtml(term)}}">
                        <div class="keyword-term">
                            ${{escapeHtml(term)}}
                            ${{isDuplicate ? `<span class="duplicate-badge">${{occurrences.length}}x</span>` : ''}}
                            <div class="keyword-count">${{occurrences.length}}</div>
                        </div>
                        <div class="keyword-summary">${{escapeHtml(firstOcc.node.summary || 'No summary available')}}</div>
                    </div>
                `;
            }}).join('');
        }}
        
        function toggleKeywordPanel() {{
            const panel = document.getElementById('keyword-panel');
            panel.classList.toggle('visible');
        }}
        
        function showKeywordOccurrences(term) {{
            currentKeyword = term;
            const occurrences = keywordIndex[term];
            
            // Update active state in keyword list
            document.querySelectorAll('.keyword-item').forEach(el => {{
                el.classList.remove('active');
            }});
            document.querySelector(`[data-keyword="${{escapeHtml(term)}}"]`)?.classList.add('active');
            
            // Highlight all occurrences in tree
            document.querySelectorAll('.keyword-chip').forEach(el => {{
                el.classList.remove('highlighted');
                if (el.textContent.trim() === term) {{
                    el.classList.add('highlighted');
                }}
            }});
            
            // Show occurrences in content panel
            const content = document.getElementById('content');
            const isDuplicate = occurrences.length > 1;
            
            content.innerHTML = `
                <div class="detail-section">
                    <h2>🔑 Keyword: ${{escapeHtml(term)}} ${{isDuplicate ? '<span style="color: #ff9800;">(' + occurrences.length + ' occurrences)</span>' : ''}}</h2>
                    
                    <div class="metadata">
                        <div class="metadata-item">
                            <span class="metadata-label">Occurrences:</span> ${{occurrences.length}}
                        </div>
                        <div class="metadata-item">
                            <span class="metadata-label">Type:</span> ${{isDuplicate ? 'Duplicate' : 'Unique'}}
                        </div>
                        <div class="metadata-item">
                            <span class="metadata-label">Sections:</span> ${{new Set(occurrences.map(occ => occ.parent)).size}}
                        </div>
                    </div>
                    
                    <div class="keyword-detail">
                        <h4>Definition</h4>
                        <p>${{escapeHtml(occurrences[0].node.summary || 'No definition available')}}</p>
                        
                        ${{occurrences[0].node.metadata?.relevance ? `
                        <div class="relevance-text">
                            <strong>Relevance:</strong> ${{escapeHtml(occurrences[0].node.metadata.relevance)}}
                        </div>
                        ` : ''}}
                    </div>
                    
                    <div class="occurrence-list">
                        <h3>All Occurrences</h3>
                        ${{occurrences.map((occ, idx) => `
                            <div class="occurrence-item" onclick="showNodeDetails('${{occ.node.node_id}}')">
                                <div class="occurrence-location">
                                    <strong>Occurrence ${{idx + 1}}</strong> • 
                                    Line ${{occ.node.line_num || 'N/A'}} • 
                                    Section: ${{escapeHtml(occ.parent)}}
                                </div>
                                <div class="occurrence-context">
                                    <strong>Sentence:</strong> "${{escapeHtml(occ.sentence.substring(0, 100))}}..."<br>
                                    <strong>Context:</strong> "${{escapeHtml(occ.node.text.substring(0, 150))}}..."
                                </div>
                            </div>
                        `).join('')}}
                    </div>
                </div>
            `;
        }}
        
        function showNodeDetails(nodeId) {{
            const node = findNode(nodeData, nodeId);
            if (!node) return;
            
            // Clear keyword highlights if switching to non-keyword node
            if (node.node_type !== 'keyword' || !currentKeyword) {{
                document.querySelectorAll('.keyword-chip').forEach(el => {{
                    el.classList.remove('highlighted');
                }});
                currentKeyword = null;
            }}
            
            // Update active state
            document.querySelectorAll('.node-title').forEach(el => {{
                el.classList.remove('active');
            }});
            document.querySelector(`[data-node-id="${{nodeId}}"]`)?.classList.add('active');
            
            // Generate content
            const content = document.getElementById('content');
            content.innerHTML = generateNodeDetails(node);
        }}
        
        function findNode(nodes, nodeId) {{
            for (const node of nodes) {{
                if (node.node_id === nodeId) return node;
                if (node.nodes && node.nodes.length > 0) {{
                    const found = findNode(node.nodes, nodeId);
                    if (found) return found;
                }}
            }}
            return null;
        }}
        
        function generateNodeDetails(node) {{
            const hasText = node.text && node.text.trim().length > 0;
            const hasSummary = node.summary && node.summary.trim().length > 0;
            const hasChildren = node.nodes && node.nodes.length > 0;
            const nodeTypeLabel = {{
                'sentence': 'Sentence',
                'keyword': 'Keyword',
                'section': 'Section',
                'semantic_group': 'Semantic Group'
            }}[node.node_type] || 'Document';
            
            let html = `
                <div class="detail-section">
                    <h2>${{escapeHtml(node.title)}}</h2>
                    
                    <div class="metadata">
                        <div class="metadata-item">
                            <span class="metadata-label">Node ID:</span> ${{node.node_id || 'N/A'}}
                        </div>
                        <div class="metadata-item">
                            <span class="metadata-label">Type:</span> ${{nodeTypeLabel}}
                        </div>
                        ${{node.line_num ? `
                        <div class="metadata-item">
                            <span class="metadata-label">Line:</span> ${{node.line_num}}
                        </div>
                        ` : ''}}
                    </div>
                    
                    <div class="stats">
                        <div class="stat">📝 Text: ${{hasText ? node.text.length + ' chars' : 'None'}}</div>
                        <div class="stat">📋 Summary: ${{hasSummary ? 'Available' : 'None'}}</div>
                        <div class="stat">👶 Children: ${{hasChildren ? node.nodes.length : '0'}}</div>
                    </div>
            `;
            
            // Special handling for keyword nodes
            if (node.node_type === 'keyword' && node.metadata) {{
                const term = node.metadata.term || node.title;
                const allOccurrences = keywordIndex[term] || [];
                
                html += `
                    <div class="keyword-detail">
                        <h4>🔑 Keyword Details</h4>
                        ${{node.metadata.term ? `
                        <p><strong>Term:</strong> ${{escapeHtml(node.metadata.term)}}</p>
                        ` : ''}}
                        ${{node.metadata.summary ? `
                        <p><strong>Definition:</strong> ${{escapeHtml(node.metadata.summary)}}</p>
                        ` : ''}}
                        
                        ${{node.metadata.relevance ? `
                        <div class="relevance-text">
                            <strong>Relevance:</strong> ${{escapeHtml(node.metadata.relevance)}}
                        </div>
                        ` : ''}}
                        
                        ${{allOccurrences.length > 1 ? `
                        <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #c8e6c9;">
                            <strong>Total Occurrences:</strong> ${{allOccurrences.length}}
                            <button onclick="showKeywordOccurrences('${{term.replace(/'/g, "\\\\'")}}')" 
                                    style="margin-left: 0.5rem; padding: 0.3rem 0.6rem; background: #8bc34a; color: white; border: none; border-radius: 3px; cursor: pointer;">
                                View All
                            </button>
                        </div>
                        ` : ''}}
                    </div>
                `;
            }}
            
            if (hasSummary) {{
                html += `
                    <h3>Summary</h3>
                    <div class="summary-content">${{escapeHtml(node.summary)}}</div>
                `;
            }}
            
            if (hasText) {{
                const preview = node.text.length > 3000 ? 
                    node.text.substring(0, 3000) + '\\n\\n... (truncated, ' + (node.text.length - 3000) + ' more characters)' : 
                    node.text;
                    
                html += `
                    <h3>Text Content</h3>
                    <div class="text-content">${{escapeHtml(preview)}}</div>
                `;
            }}
            
            if (hasChildren) {{
                const childrenByType = {{}};
                node.nodes.forEach(child => {{
                    const type = child.node_type || 'unknown';
                    if (!childrenByType[type]) childrenByType[type] = [];
                    childrenByType[type].push(child);
                }});
                
                html += `
                    <div class="children-info">
                        <strong>Child Nodes (${{node.nodes.length}}):</strong><br>
                `;
                
                Object.entries(childrenByType).forEach(([type, children]) => {{
                    html += `<div style="margin: 0.5rem 0;"><strong>${{type}}s (${{children.length}}):</strong> `;
                    if (type === 'keyword') {{
                        html += children.map(c => `<span class="keyword-chip" onclick="showNodeDetails('${{c.node_id}}')">${{escapeHtml(c.title)}}</span>`).join(' ');
                    }} else {{
                        html += children.map(c => `<span onclick="showNodeDetails('${{c.node_id}}')" style="cursor: pointer; color: #667eea; text-decoration: underline;">${{escapeHtml(c.title.substring(0, 50))}}...</span>`).join(', ');
                    }}
                    html += '</div>';
                }});
                
                html += '</div>';
            }}
            
            html += '</div>';
            return html;
        }}
        
        function escapeHtml(text) {{
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }}
        
        // Initialize keyword index on load
        document.addEventListener('DOMContentLoaded', () => {{
            buildKeywordIndex();
            
            // Setup keyword search
            document.getElementById('keyword-search').addEventListener('input', (e) => {{
                renderKeywordList(e.target.value);
            }});
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
        
        # Determine node class and icon based on type
        if node_type == 'section':
            icon = '📂'
            node_class = 'section'
            tree_class = 'section'
        elif node_type == 'semantic_group':
            icon = '🧩'
            node_class = 'semantic_group'
            tree_class = 'semantic_group'
        elif node_type == 'sentence':
            icon = '📝'
            node_class = 'sentence'
            tree_class = 'sentence'
        elif node_type == 'keyword':
            icon = '🔑'
            node_class = 'keyword'
            tree_class = 'keyword'
        else:  # Root level
            icon = '📄'
            node_class = 'root'
            tree_class = 'root'
        
        # Truncate long titles
        display_title = title if len(title) <= 60 else title[:57] + "..."
        
        html += f'<div class="tree-node {tree_class}">'
        html += f'<div class="node-title {node_class}" data-node-id="{escape(node_id)}" onclick="showNodeDetails(\'{escape(node_id)}\')">'
        html += f'<span class="node-icon">{icon}</span>'
        html += f'<span>{escape(display_title)}</span>'
        
        # Add badge with line number or child count
        if line_num:
            html += f'<span class="node-badge">L{line_num}</span>'
        elif children:
            html += f'<span class="node-badge">{len(children)}</span>'
        
        html += '</div>'
        
        # Show keyword chips for sentences
        if node_type == 'sentence' and children:
            keyword_children = [child for child in children if child.get('node_type') == 'keyword']
            if keyword_children:
                html += '<div class="keyword-chips">'
                for keyword in keyword_children:
                    keyword_title = keyword.get('title', 'Unknown')
                    keyword_id = keyword.get('node_id', 'unknown')
                    # Check if this keyword appears multiple times
                    term = keyword.get('metadata', {}).get('term', keyword_title)
                    is_duplicate = False  # Will be set by JavaScript after index is built
                    html += f'<span class="keyword-chip" onclick="showNodeDetails(\'{escape(keyword_id)}\')" title="{escape(keyword.get("summary", ""))}" data-term="{escape(term)}">{escape(keyword_title)}</span>'
                html += '</div>'
        
        # Recursively render children (but not keywords, they're shown as chips)
        if children and node_type != 'sentence':
            html += generate_tree_html(children, level + 1)
        
        html += '</div>'
    
    return html


def main():
    parser = argparse.ArgumentParser(description='Visualize markdown_v3.py structure as HTML with enhanced keyword features')
    parser.add_argument('structure_json', help='Path to structure JSON file from markdown_v3.py')
    parser.add_argument('--output', '-o', help='Output HTML file path (default: same name as input with .html extension)')
    
    args = parser.parse_args()
    
    # Load structure
    with open(args.structure_json, 'r', encoding='utf-8') as f:
        structure_data = json.load(f)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        input_path = Path(args.structure_json)
        output_path = input_path.with_suffix('.html')
    
    # Generate HTML
    generate_html(structure_data, output_path)
    
    print(f"✓ Enhanced visualization saved to: {output_path}")
    print(f"  Open in browser: file://{Path(output_path).absolute()}")
    print(f"  Features: Keyword indexing, duplicate detection, filtering")


if __name__ == '__main__':
    main()