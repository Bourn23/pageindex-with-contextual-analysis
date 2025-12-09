#!/usr/bin/env python3
"""
Visualize extracted materials as an interactive HTML page.
Shows materials, their properties, processing methods, and source traceability.

Usage:
    python visualize_materials.py results/paper_materials.json
"""

import json
import argparse
from pathlib import Path
from html import escape


def generate_html(materials_data, output_path):
    """Generate an interactive HTML visualization of extracted ionic conductivity data."""
    
    doc_name = materials_data.get('doc_name', 'Document')
    materials = materials_data.get('materials', [])
    material_count = len(materials)
    
    # Group materials by material_class
    by_type = {}
    for mat in materials:
        mtype = mat.get('material_class', 'Other') or 'Other'
        if mtype not in by_type:
            by_type[mtype] = []
        by_type[mtype].append(mat)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{escape(doc_name)} - Materials</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
            color: #333;
        }}
        
        .header {{
            background: linear-gradient(135deg, #2e7d32 0%, #1b5e20 100%);
            color: white;
            padding: 2rem;
        }}
        
        .header h1 {{ font-size: 1.8rem; margin-bottom: 0.5rem; }}
        .header .subtitle {{ opacity: 0.9; font-size: 0.9rem; }}
        
        .stats-bar {{
            background: white;
            padding: 1rem 2rem;
            display: flex;
            gap: 2rem;
            border-bottom: 1px solid #e0e0e0;
            flex-wrap: wrap;
        }}
        
        .stat {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        .stat-value {{
            font-size: 1.5rem;
            font-weight: 700;
            color: #2e7d32;
        }}
        
        .stat-label {{ color: #666; font-size: 0.85rem; }}
        
        .container {{
            display: flex;
            height: calc(100vh - 180px);
        }}
        
        .sidebar {{
            width: 320px;
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
        
        .search-box {{
            width: 100%;
            padding: 0.7rem;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            margin-bottom: 1rem;
            font-size: 0.9rem;
        }}
        
        .search-box:focus {{ outline: none; border-color: #2e7d32; }}
        
        .type-group {{
            margin-bottom: 1.5rem;
        }}
        
        .type-header {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #666;
            padding: 0.5rem;
            background: #f5f5f5;
            border-radius: 4px;
            margin-bottom: 0.5rem;
            display: flex;
            justify-content: space-between;
        }}
        
        .material-item {{
            padding: 0.7rem;
            margin: 0.3rem 0;
            cursor: pointer;
            border-radius: 6px;
            border-left: 3px solid #4caf50;
            background: #f9f9f9;
            transition: all 0.2s;
        }}
        
        .material-item:hover {{
            background: #e8f5e9;
            transform: translateX(3px);
        }}
        
        .material-item.active {{
            background: #2e7d32;
            color: white;
            border-left-color: #1b5e20;
        }}
        
        .material-abbrev {{
            font-weight: 600;
            font-size: 1rem;
        }}
        
        .material-fullname {{
            font-size: 0.8rem;
            opacity: 0.8;
            margin-top: 0.2rem;
        }}
        
        .detail-section {{ margin-bottom: 2rem; }}
        
        .detail-section h2 {{
            font-size: 1.6rem;
            margin-bottom: 0.5rem;
            color: #2e7d32;
        }}
        
        .detail-section h3 {{
            font-size: 0.9rem;
            margin: 1rem 0 0.5rem;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .full-name {{
            font-size: 1.1rem;
            color: #555;
            margin-bottom: 1rem;
        }}
        
        .badge {{
            display: inline-block;
            padding: 0.3rem 0.7rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 500;
            margin-right: 0.5rem;
            margin-bottom: 0.5rem;
        }}
        
        .badge-type {{
            background: #e3f2fd;
            color: #1565c0;
        }}
        
        .badge-composition {{
            background: #fff3e0;
            color: #e65100;
        }}
        
        .badge-processing {{
            background: #f3e5f5;
            color: #7b1fa2;
        }}
        
        .source-card {{
            background: #f9f9f9;
            padding: 1rem;
            border-radius: 6px;
            margin: 0.5rem 0;
            border-left: 3px solid #2e7d32;
        }}
        
        .source-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
        }}
        
        .source-id {{
            font-family: monospace;
            font-size: 0.8rem;
            background: #e0e0e0;
            padding: 0.2rem 0.5rem;
            border-radius: 3px;
        }}
        
        .source-section {{
            font-size: 0.85rem;
            color: #666;
        }}
        
        .source-title {{
            font-weight: 500;
        }}
        
        .empty-state {{
            color: #999;
            font-style: italic;
            padding: 2rem;
            text-align: center;
        }}
        
        .processing-list {{
            list-style: none;
        }}
        
        .processing-list li {{
            padding: 0.5rem 0.7rem;
            background: #f3e5f5;
            border-radius: 4px;
            margin: 0.3rem 0;
            border-left: 3px solid #7b1fa2;
        }}
        
        .composition-list {{
            list-style: none;
        }}
        
        .composition-list li {{
            padding: 0.5rem 0.7rem;
            background: #fff3e0;
            border-radius: 4px;
            margin: 0.3rem 0;
            border-left: 3px solid #e65100;
        }}
        
        ::-webkit-scrollbar {{ width: 8px; }}
        ::-webkit-scrollbar-track {{ background: #f1f1f1; }}
        ::-webkit-scrollbar-thumb {{ background: #888; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>⚡ Ionic Conductivity Data</h1>
        <div class="subtitle">{escape(doc_name)}</div>
    </div>
    
    <div class="stats-bar">
        <div class="stat">
            <span class="stat-value">{material_count}</span>
            <span class="stat-label">Data Points</span>
        </div>
        <div class="stat">
            <span class="stat-value">{len(by_type)}</span>
            <span class="stat-label">Material Classes</span>
        </div>
        <div class="stat">
            <span class="stat-value">{sum(1 for m in materials if m.get('processing_method') and 'N/A' not in m.get('processing_method', ''))}</span>
            <span class="stat-label">Primary Study Materials</span>
        </div>
        <div class="stat">
            <span class="stat-value">{sum(1 for m in materials if m.get('source_node'))}</span>
            <span class="stat-label">Source References</span>
        </div>
    </div>
    
    <div class="container">
        <div class="sidebar">
            <input type="text" class="search-box" id="search" placeholder="Search materials...">
            <div id="material-list">
                {generate_material_list(by_type)}
            </div>
        </div>
        
        <div class="content" id="content">
            <div class="empty-state">
                👈 Select a material to view details
            </div>
        </div>
    </div>
    
    <script>
        const materials = {json.dumps(materials, ensure_ascii=False)};
        let currentIndex = 0;
        
        function showMaterial(index) {{
            const mat = materials[index];
            if (!mat) return;
            
            currentIndex = index;
            
            // Update active state
            document.querySelectorAll('.material-item').forEach(el => {{
                el.classList.remove('active');
            }});
            document.querySelector(`[data-index="${{index}}"]`)?.classList.add('active');
            
            // Generate detail view
            const content = document.getElementById('content');
            content.innerHTML = generateDetails(mat, index);
        }}
        
        function generateDetails(mat, index) {{
            const electrolyte = mat.electrolyte_name || {{}};
            const hasFullName = electrolyte.full_name && electrolyte.full_name.trim();
            const hasAcronym = electrolyte.acronym && electrolyte.acronym.trim();
            const hasProportion = electrolyte.proportion && electrolyte.proportion.trim();
            const hasProcessing = mat.processing_method && mat.processing_method.trim() && !mat.processing_method.includes('N/A');
            const hasDescription = mat.material_description && mat.material_description.trim() && !mat.material_description.includes('N/A');
            const hasSource = mat.source_node;
            
            let html = `
                <div class="detail-section">
                    <h2>${{hasAcronym ? escapeHtml(electrolyte.acronym) : escapeHtml(electrolyte.full_name || 'Unknown')}}</h2>
                    ${{hasFullName && hasAcronym ? `<div class="full-name">${{escapeHtml(electrolyte.full_name)}}</div>` : ''}}
                    
                    <div style="margin: 1rem 0;">
                        <span class="badge badge-type">${{escapeHtml(mat.material_class || 'Other')}}</span>
                        ${{hasProportion ? `<span class="badge badge-composition">${{escapeHtml(electrolyte.proportion)}}</span>` : ''}}
                    </div>
                    
                    <h3>Ionic Conductivity</h3>
                    <div style="background: #e8f5e9; padding: 1rem; border-radius: 6px; border-left: 4px solid #2e7d32; margin-bottom: 1rem;">
                        <div style="font-size: 1.3rem; font-weight: 600; color: #1b5e20; margin-bottom: 0.5rem;">
                            ${{escapeHtml(mat.ionic_conductivity_S_per_cm)}} S/cm
                        </div>
                        <div style="color: #666;">
                            Temperature: ${{escapeHtml(mat.measurement_temperature)}}
                        </div>
                        ${{mat.specific_source_location ? `
                        <div style="color: #666; margin-top: 0.3rem;">
                            Location: ${{escapeHtml(mat.specific_source_location)}}
                        </div>
                        ` : ''}}
                    </div>
            `;
            
            if (hasDescription) {{
                html += `
                    <h3>Material Description</h3>
                    <div style="background: #f9f9f9; padding: 1rem; border-radius: 6px; margin-bottom: 1rem; line-height: 1.6;">
                        ${{escapeHtml(mat.material_description)}}
                    </div>
                `;
            }}
            
            if (hasProcessing) {{
                html += `
                    <h3>Processing Method</h3>
                    <div class="processing-list">
                        <li>${{escapeHtml(mat.processing_method)}}</li>
                    </div>
                `;
            }}
            
            if (hasSource) {{
                html += `
                    <h3>Source Node</h3>
                    <div class="source-card">
                        <div class="source-header">
                            <span class="source-id">${{escapeHtml(mat.source_node.node_id || 'N/A')}}</span>
                            <span class="source-section">${{escapeHtml(mat.source_node.section || '')}}</span>
                        </div>
                        <div class="source-title">${{escapeHtml(mat.source_node.title || 'Unknown')}}</div>
                    </div>
                `;
            }}
            
            html += '</div>';
            return html;
        }}
        
        function escapeHtml(text) {{
            if (!text) return '';
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }}
        
        function escapeAttr(text) {{
            return text.replace(/"/g, '&quot;').replace(/'/g, '&#39;');
        }}
        
        // Search functionality
        document.getElementById('search').addEventListener('input', (e) => {{
            const query = e.target.value.toLowerCase();
            document.querySelectorAll('.material-item').forEach(el => {{
                const acronym = (el.dataset.acronym || '').toLowerCase();
                const fullname = (el.dataset.fullname || '').toLowerCase();
                const visible = acronym.includes(query) || fullname.includes(query);
                el.style.display = visible ? '' : 'none';
            }});
            
            // Show/hide type groups based on visible children
            document.querySelectorAll('.type-group').forEach(group => {{
                const items = group.querySelectorAll('.material-item');
                let hasVisible = false;
                items.forEach(item => {{
                    if (item.style.display !== 'none') hasVisible = true;
                }});
                group.style.display = hasVisible ? '' : 'none';
            }});
        }});
    </script>
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)


def generate_material_list(by_type):
    """Generate HTML for the material list grouped by material class."""
    html = ""
    
    # Sort types for consistent display
    type_order = ['Ceramic', 'Polymer', 'Composite', 'Other']
    sorted_types = sorted(by_type.keys(), key=lambda t: type_order.index(t) if t in type_order else 99)
    
    global_index = 0
    for mtype in sorted_types:
        materials = by_type[mtype]
        html += f'<div class="type-group">'
        html += f'<div class="type-header"><span>{escape(mtype.upper())}</span><span>{len(materials)}</span></div>'
        
        # Sort by acronym or full name
        sorted_materials = sorted(materials, key=lambda m: (
            m.get('electrolyte_name', {}).get('acronym', '') or 
            m.get('electrolyte_name', {}).get('full_name', '')
        ).lower())
        
        for mat in sorted_materials:
            electrolyte = mat.get('electrolyte_name', {})
            acronym = electrolyte.get('acronym') or ''
            fullname = electrolyte.get('full_name') or ''
            conductivity = mat.get('ionic_conductivity_S_per_cm') or ''
            
            display_name = acronym or fullname or 'Unknown'
            
            html += f'''
                <div class="material-item" 
                     data-index="{global_index}"
                     data-acronym="{escape(acronym)}" 
                     data-fullname="{escape(fullname)}"
                     onclick="showMaterial({global_index})">
                    <div class="material-abbrev">{escape(display_name)}</div>
                    <div class="material-fullname">{escape(conductivity)} S/cm</div>
                </div>
            '''
            global_index += 1
        
        html += '</div>'
    
    return html


def main():
    parser = argparse.ArgumentParser(description='Visualize extracted materials as HTML')
    parser.add_argument('materials_json', help='Path to materials JSON file')
    parser.add_argument('--output', '-o', help='Output HTML file path')
    
    args = parser.parse_args()
    
    # Load materials
    with open(args.materials_json, 'r', encoding='utf-8') as f:
        materials_data = json.load(f)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        input_path = Path(args.materials_json)
        output_path = input_path.with_suffix('.html')
    
    # Generate HTML
    generate_html(materials_data, output_path)
    
    print(f"✓ Visualization saved to: {output_path}")
    print(f"  Open in browser: file://{Path(output_path).absolute()}")


if __name__ == '__main__':
    main()
