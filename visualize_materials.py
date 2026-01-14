#!/usr/bin/env python3
"""
Advanced Visualization for Material Extraction Data.
Features: 
- Arrhenius Plot with Confidence encoding
- Chemical Formula formatting
- Multi-source deep linking
- Grouping by stoichiometry
"""

import json
import argparse
import math
import re
from pathlib import Path
from html import escape

def load_structure_map(structure_path):
    """Loads the structure JSON and creates a flat dictionary of node_id -> node_data."""
    try:
        with open(structure_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        nodes = data.get('structure', data) if isinstance(data, dict) else data
        flat_map = {}
        
        def recurse_nodes(node_list, parent_id=None):
            for node in node_list:
                n_id = node.get('node_id')
                if n_id:
                    flat_map[n_id] = {
                        'text': node.get('text', ''),
                        'title': node.get('title', ''),
                        'type': node.get('node_type', 'unknown'),
                        'src': node.get('src', ''),
                        'parent_id': parent_id
                    }
                if 'nodes' in node and isinstance(node['nodes'], list):
                    recurse_nodes(node['nodes'], parent_id=n_id)

        recurse_nodes(nodes)
        print(f"  ✓ Loaded structure map: {len(flat_map)} nodes indexed")
        return flat_map
    except Exception as e:
        print(f"  ⚠️ Warning: Could not load structure map ({e})")
        return {}

def format_formula(formula):
    """Simple helper to create HTML subscripts for chemical formulas."""
    if not formula: return "Unknown"
    # Regex: Match numbers and wrap them in <sub>
    return re.sub(r'(\d+(\.\d+)?)', r'<sub>\1</sub>', str(formula))

def prepare_plot_data(materials):
    """Prepares X/Y coordinates for the Arrhenius plot with Confidence encoding."""
    plot_points = []
    
    for idx, mat in enumerate(materials):
        cond = mat.get('_norm_cond')
        temp = mat.get('_norm_temp')
        
        # 1. Fallback for Room Temperature if normalized value is missing
        is_estimated_temp = False
        if temp is None:
            raw_temp = str(mat.get('measurement_temperature', '')).lower()
            if any(x in raw_temp for x in ['room', 'rt', 'ambient']):
                temp = 25.0
                is_estimated_temp = True
                
        # 2. Validation for plotting
        if isinstance(cond, (int, float)) and cond > 0 and isinstance(temp, (int, float)):
            try:
                # Arrhenius X: 1000 / T(Kelvin)
                x_val = 1000.0 / (temp + 273.15)
                # Arrhenius Y: log10(Conductivity)
                y_val = math.log10(cond)
                
                # Get display name
                el = mat.get('electrolyte_name', {})
                name = mat.get('canonical_formula') or el.get('full_name') or 'Unknown'
                conf = mat.get('confidence', 'low').lower()
                
                # Map Confidence to Symbols
                symbol = 'circle' # Default High
                if conf == 'medium': symbol = 'diamond'
                elif conf == 'low': symbol = 'cross'

                plot_points.append({
                    'x': round(x_val, 3),
                    'y': round(y_val, 3),
                    'text': f"{name} ({temp}°C)",
                    'id': idx, 
                    'group': mat.get('material_class', 'Other'),
                    'confidence': conf,
                    'marker_symbol': symbol,
                    'is_estimated': is_estimated_temp
                })
            except (ValueError, ZeroDivisionError):
                continue
                
    return plot_points

def generate_html(materials_data, structure_map, output_path):
    doc_name = materials_data.get('source_file', 'Document')
    raw_materials = materials_data.get('materials', [])
    
    # 1. Sort data
    materials = sorted(raw_materials, key=lambda m: (
        m.get('material_class', 'Other') or 'Other',
        (m.get('canonical_formula') or '').lower()
    ))
    
    # 2. Add formatted formula to data object for JS to use
    for i, mat in enumerate(materials):
        mat['_ui_index'] = i
        mat['_ui_formula'] = format_formula(mat.get('canonical_formula') or mat.get('electrolyte_name', {}).get('full_name'))

    # 3. Group by Material Class -> Canonical Formula
    sidebar_tree = {}
    for mat in materials:
        mtype = mat.get('material_class', 'Other') or 'Other'
        formula = mat.get('canonical_formula') or "Unspecified"
        
        if mtype not in sidebar_tree: sidebar_tree[mtype] = {}
        if formula not in sidebar_tree[mtype]: sidebar_tree[mtype][formula] = []
        
        sidebar_tree[mtype][formula].append(mat)
        
    # 4. Prepare Plot Data
    plot_data = prepare_plot_data(materials)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Audit: {escape(doc_name)}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    
    <style>
        :root {{ --primary: #1a73e8; --primary-bg: #e8f0fe; --border: #dadce0; --bg: #f8f9fa; --text: #202124; }}
        body {{ font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: var(--bg); margin: 0; height: 100vh; display: flex; flex-direction: column; color: var(--text); }}
        
        /* Header */
        .header {{ background: white; padding: 0.8rem 1.5rem; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; box-shadow: 0 1px 2px rgba(0,0,0,0.05); z-index: 10; }}
        .header h1 {{ font-size: 1.1rem; margin: 0; color: #444; }}
        .stat-badge {{ background: var(--primary); color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8rem; font-weight: bold; }}

        .container {{ display: flex; flex: 1; overflow: hidden; }}
        
        /* Sidebar */
        .sidebar {{ width: 350px; background: white; border-right: 1px solid var(--border); overflow-y: auto; display: flex; flex-direction: column; flex-shrink: 0; }}
        .search-box {{ margin: 1rem; padding: 0.6rem; border: 1px solid #ccc; border-radius: 6px; font-size: 0.9rem; }}
        
        .group-header {{ font-size: 0.75rem; font-weight: 700; color: #5f6368; text-transform: uppercase; padding: 1rem 1rem 0.5rem; background: #fff; position: sticky; top: 0; }}
        .formula-group {{ padding: 0 0.5rem; margin-bottom: 0.5rem; }}
        .formula-title {{ padding: 0.4rem 0.5rem; font-weight: 600; font-size: 0.9rem; color: #3c4043; background: #f1f3f4; border-radius: 4px; margin-bottom: 2px; }}
        
        .mat-item {{ padding: 0.5rem 0.8rem; margin-left: 0.5rem; border-radius: 4px; cursor: pointer; display: flex; justify-content: space-between; align-items: center; font-size: 0.85rem; border-left: 3px solid transparent; }}
        .mat-item:hover {{ background: var(--bg); }}
        .mat-item.active {{ background: var(--primary-bg); border-left-color: var(--primary); color: var(--primary); }}
        .mat-item.invalid {{ border-left-color: #d93025; background: #fce8e6; }}
        
        .conf-dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 6px; }}
        .conf-high {{ background: #1e8e3e; }}
        .conf-medium {{ background: #f9ab00; }}
        .conf-low {{ background: #d93025; }}

        /* Content */
        .content {{ flex: 1; overflow-y: auto; padding: 2rem; max-width: 1400px; margin: 0 auto; width: 100%; }}
        .plot-card {{ background: white; border-radius: 8px; border: 1px solid var(--border); padding: 1rem; margin-bottom: 1.5rem; height: 450px; position: relative; }}
        
        /* Detail View */
        .detail-card {{ background: white; border-radius: 8px; border: 1px solid var(--border); padding: 2rem; }}
        .chem-title {{ font-size: 1.8rem; margin-bottom: 0.5rem; font-family: 'Georgia', serif; }}
        
        .tag-row {{ display: flex; gap: 0.5rem; margin-bottom: 1.5rem; flex-wrap: wrap; }}
        .tag {{ padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }}
        .tag-blue {{ background: #e8f0fe; color: #1967d2; }}
        .tag-green {{ background: #e6f4ea; color: #137333; }}
        
        .data-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem; background: #fafafa; padding: 1.5rem; border-radius: 8px; }}
        .data-item label {{ display: block; font-size: 0.75rem; color: #5f6368; text-transform: uppercase; margin-bottom: 4px; }}
        .data-item div {{ font-size: 1.2rem; font-family: monospace; font-weight: 600; color: #202124; }}
        
        .source-tags {{ display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.5rem; }}
        .source-btn {{ border: 1px solid var(--border); background: white; padding: 4px 10px; border-radius: 16px; font-size: 0.8rem; cursor: pointer; transition: all 0.2s; display: flex; align-items: center; gap: 4px; }}
        .source-btn:hover {{ background: #f1f3f4; border-color: #999; }}
        .source-btn.img-source {{ border-color: #fce8e6; color: #c5221f; background: #fce8e6; }}
        .source-btn.img-source:hover {{ background: #fad2cf; }}

        /* Modal */
        .modal-overlay {{ position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.6); z-index: 1000; display: none; justify-content: center; align-items: center; }}
        .modal-card {{ background: white; width: 90%; max-width: 900px; max-height: 90vh; border-radius: 8px; display: flex; flex-direction: column; box-shadow: 0 10px 25px rgba(0,0,0,0.2); }}
        .modal-header {{ padding: 1rem 1.5rem; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; background: #f8f9fa; }}
        .modal-body {{ padding: 2rem; overflow-y: auto; line-height: 1.6; font-size: 1rem; }}
        .highlight {{ background-color: #ffe082; padding: 2px 0; }}
        .modal-img {{ max-width: 100%; border: 1px solid #ddd; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Extraction Audit: {escape(doc_name)}</h1>
        <span class="stat-badge">{len(materials)} Materials</span>
    </div>
    
    <div class="container">
        <div class="sidebar">
            <input type="text" id="search" class="search-box" placeholder="Filter by formula...">
            <div id="sidebar-list">
                {generate_grouped_sidebar(sidebar_tree)}
            </div>
        </div>
        
        <div class="content">
            <div class="plot-card" id="arrhenius-plot"></div>
            <div id="detail-view">
                <div style="text-align:center; padding: 4rem; color: #999;">
                    Select a data point from the sidebar or plot to view details.
                </div>
            </div>
        </div>
    </div>

    <div class="modal-overlay" id="sourceModal" onclick="closeModal(event)">
        <div class="modal-card" onclick="event.stopPropagation()">
            <div class="modal-header">
                <strong id="modalTitle">Source Context</strong>
                <button onclick="closeModal()" style="border:none; background:none; font-size:1.5rem; cursor:pointer;">&times;</button>
            </div>
            <div class="modal-body" id="modalContent"></div>
        </div>
    </div>

    <script>
        const materials = {json.dumps(materials, ensure_ascii=False)};
        const sourceMap = {json.dumps(structure_map, ensure_ascii=False)};
        const plotData = {json.dumps(plot_data, ensure_ascii=False)};
        
        // --- PLOTLY INIT ---
        function initPlot() {{
            if (plotData.length === 0) {{
                document.getElementById('arrhenius-plot').innerHTML = '<div style="text-align:center; padding:2rem; color:#888">No plottable data available (Requires Temp & Cond)</div>';
                return;
            }}

            const groups = {{}};
            plotData.forEach(p => {{
                if (!groups[p.group]) groups[p.group] = {{ x: [], y: [], text: [], ids: [], symbols: [] }};
                groups[p.group].x.push(p.x);
                groups[p.group].y.push(p.y);
                groups[p.group].text.push(p.text);
                groups[p.group].ids.push(p.id);
                groups[p.group].symbols.push(p.marker_symbol);
            }});

            const traces = Object.keys(groups).map(g => ({{
                x: groups[g].x,
                y: groups[g].y,
                text: groups[g].text,
                customdata: groups[g].ids,
                mode: 'markers',
                type: 'scatter',
                name: g,
                marker: {{ size: 12, opacity: 0.7, symbol: groups[g].symbols }}
            }}));

            const layout = {{
                title: 'Arrhenius Plot',
                xaxis: {{ title: '1000 / T (K⁻¹)' }},
                yaxis: {{ title: 'log₁₀(σ) (S/cm)' }},
                hovermode: 'closest',
                template: 'plotly_white'
            }};

            Plotly.newPlot('arrhenius-plot', traces, layout, {{responsive: true}});
            
            document.getElementById('arrhenius-plot').on('plotly_click', function(data){{
                const pt = data.points[0];
                selectMaterial(pt.customdata);
                const el = document.getElementById('mat-'+pt.customdata);
                if(el) el.scrollIntoView({{behavior: 'smooth', block: 'center'}});
            }});
        }}

        // --- UI LOGIC ---
        function selectMaterial(index) {{
            document.querySelectorAll('.mat-item').forEach(el => el.classList.remove('active'));
            const btn = document.getElementById('mat-' + index);
            if (btn) btn.classList.add('active');
            
            const mat = materials[index];
            renderDetail(mat);
        }}

        function renderDetail(mat) {{
            // Multi-source Parser
            const rawSource = mat.source_sentence_id || "";
            const sourceParts = rawSource.split(/,\s*/);
            
            let sourceBtns = '';
            sourceParts.forEach(part => {{
                part = part.trim();
                if(!part) return;
                
                // Logic: Does it look like a known Node ID?
                if (sourceMap[part]) {{
                    sourceBtns += `<button class="source-btn" onclick="showSource('${{part}}')">📄 ${{part}} (Text)</button>`;
                }} else if (part.toLowerCase().includes('plot') || part.toLowerCase().includes('figure') || part.toLowerCase().includes('derived')) {{
                    // Try to link to the source node if it's an image extraction
                    const fallback = mat.source_node ? mat.source_node.node_id : null;
                    const validFallback = (fallback && sourceMap[fallback]);
                    const action = validFallback ? `onclick="showSource('${{fallback}}')"` : '';
                    const style = validFallback ? '' : 'style="opacity:0.6; cursor:default"';
                    sourceBtns += `<button class="source-btn img-source" ${{action}} ${{style}}>📷 ${{part}}</button>`;
                }} else {{
                    sourceBtns += `<span style="font-size:0.8rem; color:#666; padding:4px;">${{escape(part)}}</span>`;
                }}
            }});
            
            if (sourceBtns === '') sourceBtns = '<span style="color:#999">No direct source link</span>';

            const html = `
                <div class="detail-card">
                    <div class="chem-title">${{mat._ui_formula}}</div>
                    
                    <div class="tag-row">
                        <span class="tag tag-blue">${{mat.material_class}}</span>
                        <span class="tag" style="border:1px solid #ccc">Confidence: ${{mat.confidence}}</span>
                        <span class="tag" style="border:1px solid #ccc">${{mat.data_source}}</span>
                        ${{mat.electrolyte_name.proportion ? `<span class="tag tag-green">${{mat.electrolyte_name.proportion}}</span>` : ''}}
                    </div>
                    
                    ${{(!mat._validation.is_valid) ? 
                        `<div style="background:#fce8e6; color:#c5221f; padding:1rem; border-radius:6px; margin-bottom:1rem;">
                            <strong>Validation Issues:</strong> ${{(mat._validation.issues || []).join(', ')}}
                        </div>` : ''}}

                    <div class="data-grid">
                        <div class="data-item">
                            <label>Conductivity</label>
                            <div>${{mat._norm_cond ? mat._norm_cond.toExponential(2) : mat.ionic_conductivity_S_per_cm}} <small>S/cm</small></div>
                        </div>
                        <div class="data-item">
                            <label>Temperature</label>
                            <div>
                                ${{mat._norm_temp !== null ? mat._norm_temp : mat.measurement_temperature}} 
                                <small>${{mat._norm_temp !== null ? '°C' : ''}}</small>
                            </div>
                        </div>
                    </div>
                    
                    <div style="margin-bottom:1.5rem">
                        <strong>Extraction Logic:</strong>
                        <div style="margin-top:0.5rem; color:#444; line-height:1.5; background:#f8f9fa; padding:1rem; border-radius:4px;">
                            ${{escape(mat.reason)}}
                        </div>
                    </div>
                    
                    <div style="margin-top:2rem; border-top:1px solid #eee; padding-top:1rem;">
                        <small style="text-transform:uppercase; color:#888; font-weight:bold;">Source Traceability</small>
                        <div class="source-tags">${{sourceBtns}}</div>
                    </div>
                </div>
            `;
            document.getElementById('detail-view').innerHTML = html;
        }}

        function showSource(nodeId) {{
            const node = sourceMap[nodeId];
            if (!node) return;
            
            let content = '';
            if (node.type === 'image' || node.text.includes('.jpeg') || node.text.includes('.png')) {{
                content = `<img src="${{node.src}}" class="modal-img"><div style="margin-top:1rem; color:#666">${{escape(node.text)}}</div>`;
            }} else {{
                content = `<div style="white-space:pre-wrap; font-family:serif; font-size:1.1rem; color:#333;">${{escape(node.text)}}</div>`;
                if(node.parent_id) content = `<small>Section: ${{node.parent_id}}</small>` + content;
            }}
            
            document.getElementById('modalTitle').innerText = `Source Node: ${{nodeId}}`;
            document.getElementById('modalContent').innerHTML = content;
            document.getElementById('sourceModal').style.display = 'flex';
        }}

        function closeModal(e) {{
            if(e) e.preventDefault();
            document.getElementById('sourceModal').style.display = 'none';
        }}
        
        function escape(s) {{
            if(!s) return '';
            return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
        }}
        
        // Search Filter
        document.getElementById('search').addEventListener('keyup', (e) => {{
            const val = e.target.value.toLowerCase();
            document.querySelectorAll('.formula-group').forEach(grp => {{
                const txt = grp.innerText.toLowerCase();
                grp.style.display = txt.includes(val) ? 'block' : 'none';
            }});
        }});

        initPlot();
        if(materials.length > 0) selectMaterial(0);
    </script>
</body>
</html>
"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

def generate_grouped_sidebar(tree):
    """Generates the Sidebar HTML grouping materials by Formula."""
    html = ""
    for mtype, formulas in tree.items():
        html += f'<div class="group-header">{escape(mtype)}</div>'
        
        for form_name, mat_list in formulas.items():
            formatted_name = format_formula(form_name)
            html += f'<div class="formula-group"><div class="formula-title">{formatted_name}</div>'
            
            for mat in mat_list:
                idx = mat['_ui_index']
                conf = mat.get('confidence', 'low').lower()
                temp_disp = f"{mat.get('_norm_temp')}°C" if mat.get('_norm_temp') is not None else "Unknown T"
                cond_val = mat.get('_norm_cond')
                cond_disp = f"{cond_val:.1e}" if cond_val else "N/A"
                
                valid_cls = "" if mat['_validation']['is_valid'] else "invalid"
                
                html += f'''
                    <div class="mat-item {valid_cls}" id="mat-{idx}" onclick="selectMaterial({idx})">
                        <div style="display:flex; align-items:center;">
                            <span class="conf-dot conf-{conf}" title="Confidence: {conf}"></span>
                            <span>{temp_disp}</span>
                        </div>
                        <span style="font-family:monospace; color:#555">{cond_disp}</span>
                    </div>
                '''
            html += "</div>"
    return html

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('materials_json', help="Path to the extracted materials JSON")
    parser.add_argument('-o', '--output', help='Output HTML path')
    args = parser.parse_args()
    
    mat_path = Path(args.materials_json)
    structure_path = mat_path.parent / (mat_path.name.replace('_materials.json', '.json').replace('_results.json', '.json'))
    
    # Fallback to check if .json exists with exact name if replacement failed
    if not structure_path.exists():
        structure_path = mat_path.with_suffix('.json')

    print(f"Loading materials: {mat_path}")
    print(f"Loading structure: {structure_path}")
    
    with open(mat_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    structure_map = load_structure_map(structure_path)
    
    out_path = args.output if args.output else mat_path.with_suffix('.html')
    generate_html(data, structure_map, str(out_path))
    
    print(f"✓ Visualization saved to: {out_path}")
    print(f"  Open in browser: file://{Path(out_path).absolute()}")