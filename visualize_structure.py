#!/usr/bin/env python3
"""
Visualize PageIndex structure as an interactive HTML page.
Makes it easy to browse the hierarchy and check text content.
"""

import json
import argparse
from pathlib import Path
from html import escape


def generate_html(structure_data, output_path):
    """Generate an interactive HTML visualization of the structure."""
    
    # Handle both dict and list formats
    if isinstance(structure_data, dict):
        doc_name = structure_data.get('doc_name', 'Document')
        structure = structure_data.get('structure', [])
    else:
        doc_name = 'Document'
        structure = structure_data
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{escape(doc_name)} - Structure Visualization</title>
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
            width: 350px;
            background: white;
            border-right: 1px solid #e0e0e0;
            overflow-y: auto;
            padding: 1rem;
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
        
        .node-title {{
            padding: 0.5rem;
            margin: 0.25rem 0;
            cursor: pointer;
            border-radius: 4px;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        .node-title:hover {{
            background: #f0f0f0;
        }}
        
        .node-title.active {{
            background: #667eea;
            color: white;
            font-weight: 500;
        }}
        
        .node-icon {{
            font-size: 0.8rem;
            opacity: 0.6;
        }}
        
        .node-badge {{
            font-size: 0.7rem;
            padding: 0.15rem 0.4rem;
            border-radius: 3px;
            background: #e0e0e0;
            margin-left: auto;
        }}
        
        .node-title.active .node-badge {{
            background: rgba(255,255,255,0.2);
        }}
        
        .node-title.keyword {{
            font-size: 0.85rem;
            background: #f1f8e9;
            border-left: 3px solid #8bc34a;
        }}
        
        .node-title.keyword:hover {{
            background: #dcedc8;
        }}
        
        .node-title.keyword.active {{
            background: #689f38;
            color: white;
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
        
        .empty-state {{
            color: #999;
            font-style: italic;
            padding: 1rem;
        }}
        
        .stats {{
            display: flex;
            gap: 1rem;
            margin-bottom: 1rem;
            font-size: 0.85rem;
        }}
        
        .stat {{
            background: #e8eaf6;
            padding: 0.4rem 0.8rem;
            border-radius: 4px;
        }}
        
        .children-info {{
            background: #e3f2fd;
            padding: 0.8rem;
            border-radius: 4px;
            margin-top: 1rem;
            font-size: 0.9rem;
        }}
        
        .warning {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 1rem;
            border-radius: 4px;
            margin-bottom: 1rem;
        }}
        
        .warning-title {{
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: #856404;
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
        <h1>📄 {escape(doc_name)}</h1>
        <div class="subtitle">Structure Visualization</div>
    </div>
    
    <div class="container">
        <div class="sidebar" id="sidebar">
            <h3 style="margin-bottom: 1rem; color: #667eea;">Document Structure</h3>
            {generate_tree_html(structure)}
        </div>
        
        <div class="content" id="content">
            <div class="empty-state">
                👈 Select a node from the tree to view its details
            </div>
        </div>
    </div>
    
    <script>
        const nodeData = {json.dumps(structure, ensure_ascii=False)};
        
        function showNodeDetails(nodeId) {{
            const node = findNode(nodeData, nodeId);
            if (!node) return;
            
            // Update active state
            document.querySelectorAll('.node-title').forEach(el => {{
                el.classList.remove('active');
            }});
            document.querySelector(`[data-node-id="${{nodeId}}"]`).classList.add('active');
            
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
            
            let html = `
                <div class="detail-section">
                    <h2>${{escapeHtml(node.title)}}</h2>
                    
                    <div class="metadata">
                        <div class="metadata-item">
                            <span class="metadata-label">Node ID:</span> ${{node.node_id || 'N/A'}}
                        </div>
                        <div class="metadata-item">
                            <span class="metadata-label">Pages:</span> ${{node.start_index}} - ${{node.end_index}}
                        </div>
                        ${{node.node_type ? `
                        <div class="metadata-item">
                            <span class="metadata-label">Type:</span> ${{node.node_type}}
                        </div>
                        ` : ''}}
                    </div>
                    
                    <div class="stats">
                        <div class="stat">📝 Text: ${{hasText ? node.text.length + ' chars' : 'None'}}</div>
                        <div class="stat">📋 Summary: ${{hasSummary ? node.summary.length + ' chars' : 'None'}}</div>
                        <div class="stat">👶 Children: ${{hasChildren ? node.nodes.length : '0'}}</div>
                    </div>
            `;
            
            // Special handling for keyword nodes
            if (node.node_type === 'keyword' && node.metadata) {{
                html += `
                    <div style="background: #e8f5e9; padding: 1rem; border-radius: 6px; border-left: 4px solid #4caf50; margin-bottom: 1rem;">
                        <h3 style="color: #2e7d32; margin-bottom: 0.5rem;">🔑 Keyword Details</h3>
                        ${{node.metadata.term ? `
                        <div style="margin-bottom: 0.5rem;">
                            <strong>Term:</strong> ${{escapeHtml(node.metadata.term)}}
                        </div>
                        ` : ''}}
                        ${{node.metadata.context ? `
                        <div style="margin-bottom: 0.5rem;">
                            <strong>Context:</strong> ${{escapeHtml(node.metadata.context)}}
                        </div>
                        ` : ''}}
                        ${{node.metadata.relevance ? `
                        <div style="margin-bottom: 0.5rem;">
                            <strong>Relevance:</strong> ${{escapeHtml(node.metadata.relevance)}}
                        </div>
                        ` : ''}}
                        ${{node.metadata.parent_title ? `
                        <div style="margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid #c8e6c9;">
                            <strong>Parent Section:</strong> ${{escapeHtml(node.metadata.parent_title)}}
                            ${{node.metadata.parent_node_type ? ` <span style="font-size: 0.85em; color: #666;">(${{node.metadata.parent_node_type}})</span>` : ''}}
                        </div>
                        ` : ''}}
                    </div>
                `;
            }}
            
            // Check for duplicate text issue
            if (hasChildren && hasText) {{
                const childTexts = node.nodes.filter(n => n.text).map(n => n.text);
                const allSame = childTexts.length > 1 && childTexts.every(t => t === childTexts[0]);
                
                if (allSame && childTexts[0] === node.text) {{
                    html += `
                        <div class="warning">
                            <div class="warning-title">⚠️ Duplicate Text Detected</div>
                            All child nodes have identical text content (same as parent). This typically happens in coarse granularity mode.
                            Consider using medium or fine granularity for proper text segmentation.
                        </div>
                    `;
                }}
            }}
            
            if (hasSummary) {{
                html += `
                    <h3>Summary</h3>
                    <div class="summary-content">${{escapeHtml(node.summary)}}</div>
                `;
            }}
            
            if (hasText) {{
                const preview = node.text.length > 5000 ? 
                    node.text.substring(0, 5000) + '\\n\\n... (truncated, ' + (node.text.length - 5000) + ' more characters)' : 
                    node.text;
                    
                html += `
                    <h3>Text Content</h3>
                    <div class="text-content">${{escapeHtml(preview)}}</div>
                `;
            }}
            
            if (hasChildren) {{
                html += `
                    <div class="children-info">
                        <strong>Child Nodes (${{node.nodes.length}}):</strong><br>
                        ${{node.nodes.map(n => `• ${{escapeHtml(n.title)}} (Pages ${{n.start_index}}-${{n.end_index}})`).join('<br>')}}
                    </div>
                `;
            }}
            
            html += '</div>';
            return html;
        }}
        
        function escapeHtml(text) {{
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }}
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
        start = node.get('start_index', '?')
        end = node.get('end_index', '?')
        node_type = node.get('node_type', '')
        children = node.get('nodes', [])
        
        # Icon based on type
        icon = {
            'figure': '🖼️',
            'table': '📊',
            'semantic_unit': '📝',
            'keyword': '�',
        }.get(node_type, '📄')
        
        tree_class = 'root' if level == 0 else ''
        node_class = f'keyword' if node_type == 'keyword' else ''
        
        html += f'<div class="tree-node {tree_class}">'
        html += f'<div class="node-title {node_class}" data-node-id="{escape(node_id)}" onclick="showNodeDetails(\'{escape(node_id)}\')">'
        html += f'<span class="node-icon">{icon}</span>'
        html += f'<span>{escape(title)}</span>'
        html += f'<span class="node-badge">p{start}-{end}</span>'
        html += '</div>'
        
        if children:
            html += generate_tree_html(children, level + 1)
        
        html += '</div>'
    
    return html


def main():
    parser = argparse.ArgumentParser(description='Visualize PageIndex structure as HTML')
    parser.add_argument('structure_json', help='Path to structure JSON file')
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
    
    print(f"✓ Visualization saved to: {output_path}")
    print(f"  Open in browser: file://{Path(output_path).absolute()}")


if __name__ == '__main__':
    main()
