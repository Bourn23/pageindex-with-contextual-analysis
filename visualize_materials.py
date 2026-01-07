#!/usr/bin/env python3
"""
Visualize extracted materials as an interactive HTML page.
Updated to support validation flags, exact quotes, and normalized data.

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
    raw_materials = materials_data.get('materials', [])
    
    # --- CRITICAL FIX: SORT DATA BEFORE PROCESSING ---
    # We sort the list once here so that the Python-generated sidebar 
    # and the JS-generated detail view share the exact same index order.
    # Sorting by: Material Class -> Acronym -> Full Name
    materials = sorted(raw_materials, key=lambda m: (
        m.get('material_class', 'Other') or 'Other',
        (m.get('electrolyte_name', {}).get('acronym') or '').lower(),
        (m.get('electrolyte_name', {}).get('full_name') or '').lower()
    ))
    
    material_count = len(materials)
    
    # Group for the sidebar display
    by_type = {}
    for i, mat in enumerate(materials):
        mtype = mat.get('material_class', 'Other') or 'Other'
        if mtype not in by_type:
            by_type[mtype] = []
        # Store the global index so we can link sidebar to JS array
        mat['_ui_index'] = i 
        by_type[mtype].append(mat)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{escape(doc_name)} - Extraction Audit</title>
    <style>
        :root {{
            --primary: #2e7d32;
            --primary-light: #e8f5e9;
            --border: #e0e0e0;
            --text: #333;
            --danger: #d32f2f;
            --danger-bg: #ffebee;
            --warning: #ed6c02;
            --bg: #f5f5f5;
        }}
        
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
            height: 100vh;
            display: flex;
            flex-direction: column;
        }}
        
        /* Header & Stats */
        .header {{
            background: white;
            padding: 1rem 2rem;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-shrink: 0;
        }}
        
        .header h1 {{ font-size: 1.2rem; display: flex; align-items: center; gap: 0.5rem; }}
        .header .doc-name {{ color: #666; font-weight: normal; font-size: 0.9rem; margin-top: 0.2rem; }}
        
        .stats {{ display: flex; gap: 1.5rem; }}
        .stat-item {{ text-align: right; }}
        .stat-val {{ font-weight: 700; font-size: 1.2rem; color: var(--primary); }}
        .stat-lbl {{ font-size: 0.7rem; color: #777; text-transform: uppercase; }}

        /* Main Layout */
        .container {{
            display: flex;
            flex: 1;
            overflow: hidden;
        }}
        
        /* Sidebar */
        .sidebar {{
            width: 350px;
            background: white;
            border-right: 1px solid var(--border);
            overflow-y: auto;
            display: flex;
            flex-direction: column;
        }}
        
        .search-container {{ padding: 1rem; border-bottom: 1px solid var(--border); }}
        .search-box {{
            width: 100%;
            padding: 0.6rem;
            border: 1px solid var(--border);
            border-radius: 6px;
            font-size: 0.9rem;
        }}
        
        .material-list {{ flex: 1; overflow-y: auto; padding: 0.5rem; }}
        
        .type-header {{
            font-size: 0.7rem;
            font-weight: 700;
            color: #888;
            text-transform: uppercase;
            padding: 0.8rem 0.5rem 0.4rem;
            letter-spacing: 0.5px;
        }}
        
        .mat-item {{
            padding: 0.8rem;
            margin-bottom: 0.3rem;
            border-radius: 6px;
            cursor: pointer;
            border-left: 3px solid transparent;
            transition: all 0.2s;
        }}
        
        .mat-item:hover {{ background: var(--bg); }}
        .mat-item.active {{ background: var(--primary-light); border-left-color: var(--primary); }}
        
        /* Invalid items get a red indicator in the list */
        .mat-item.invalid {{ border-left-color: var(--danger); opacity: 0.8; }}
        .mat-item.invalid.active {{ background: var(--danger-bg); }}
        
        .mat-name {{ font-weight: 600; font-size: 0.95rem; margin-bottom: 0.2rem; }}
        .mat-cond {{ font-size: 0.8rem; color: #666; font-family: monospace; }}
        
        /* Content Area */
        .content {{
            flex: 1;
            overflow-y: auto;
            padding: 2rem;
            max-width: 1000px;
        }}
        
        .detail-card {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            padding: 2rem;
            margin-bottom: 2rem;
        }}
        
        /* Detail Elements */
        .detail-header {{ margin-bottom: 1.5rem; border-bottom: 1px solid var(--border); padding-bottom: 1rem; }}
        .detail-title {{ font-size: 1.6rem; color: var(--text); margin-bottom: 0.5rem; }}
        .detail-subtitle {{ font-size: 1rem; color: #666; font-weight: 400; }}
        
        .badges {{ display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 1rem; }}
        .badge {{ padding: 0.25rem 0.6rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; }}
        
        .bg-class {{ background: #e3f2fd; color: #1565c0; }}
        .bg-conf-high {{ background: #e8f5e9; color: #2e7d32; }}
        .bg-conf-medium {{ background: #fff3e0; color: #ef6c00; }}
        .bg-conf-low {{ background: #ffebee; color: #c62828; }}
        .bg-source {{ background: #f3e5f5; color: #7b1fa2; }}
        
        /* Validation Box */
        .validation-box {{
            background: var(--danger-bg);
            border: 1px solid var(--danger);
            color: var(--danger);
            padding: 1rem;
            border-radius: 6px;
            margin-bottom: 1.5rem;
        }}
        .validation-title {{ font-weight: 700; display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem; }}
        .validation-list {{ padding-left: 1.5rem; font-size: 0.9rem; }}
        
        /* Grid Layout for Properties */
        .prop-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-bottom: 1.5rem;
        }}
        
        .prop-box {{
            background: var(--bg);
            padding: 1rem;
            border-radius: 6px;
        }}
        
        .prop-label {{ font-size: 0.75rem; color: #666; text-transform: uppercase; margin-bottom: 0.5rem; letter-spacing: 0.5px; }}
        .prop-value {{ font-size: 1.1rem; font-weight: 500; font-family: monospace; }}
        .prop-meta {{ font-size: 0.8rem; color: #888; margin-top: 0.3rem; }}
        
        /* Quote Block */
        .quote-block {{
            border-left: 4px solid var(--primary);
            background: #f9f9f9;
            padding: 1rem 1.5rem;
            font-style: italic;
            color: #555;
            margin: 1.5rem 0;
            border-radius: 0 6px 6px 0;
            line-height: 1.6;
        }}
        
        /* Tables */
        .meta-table {{ width: 100%; font-size: 0.9rem; border-collapse: collapse; }}
        .meta-table td {{ padding: 0.5rem; border-bottom: 1px solid var(--border); }}
        .meta-table td:first-child {{ width: 150px; color: #666; font-weight: 500; }}
        
        .empty-state {{ padding: 4rem; text-align: center; color: #999; }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>⚡ Material Extraction Audit</h1>
            <div class="doc-name">{escape(doc_name)}</div>
        </div>
        <div class="stats">
            <div class="stat-item">
                <div class="stat-val">{material_count}</div>
                <div class="stat-lbl">Total Materials</div>
            </div>
            <div class="stat-item">
                <div class="stat-val" style="color: var(--danger)">
                    {sum(1 for m in materials if not m.get('_validation', {}).get('is_valid', True))}
                </div>
                <div class="stat-lbl">Issues Found</div>
            </div>
        </div>
    </div>
    
    <div class="container">
        <div class="sidebar">
            <div class="search-container">
                <input type="text" id="search" class="search-box" placeholder="Filter materials...">
            </div>
            <div class="material-list" id="sidebar-list">
                {generate_sidebar_html(by_type, type_order=['Composite', 'Ceramic', 'Polymer', 'Other'])}
            </div>
        </div>
        
        <div class="content" id="detail-view">
            <div class="empty-state">Select a material from the sidebar to view extraction details.</div>
        </div>
    </div>

    <script>
        // We dump the ALREADY SORTED list here so indices match perfectly
        const materials = {json.dumps(materials, ensure_ascii=False)};
        
        function selectMaterial(index) {{
            // Remove active class from all items
            document.querySelectorAll('.mat-item').forEach(el => el.classList.remove('active'));
            
            // Add active class to selected
            const sidebarItem = document.getElementById(`mat-${{index}}`);
            if(sidebarItem) sidebarItem.classList.add('active');
            
            renderDetail(index);
        }}
        
        function renderDetail(index) {{
            const mat = materials[index];
            if(!mat) return;
            
            const elName = mat.electrolyte_name || {{}};
            const valid = mat._validation || {{ is_valid: true, issues: [] }};
            
            let html = `
                <div class="detail-card">
                    <div class="detail-header">
                        <div class="detail-title">
                            ${{escape(elName.acronym || elName.full_name || 'Unknown Material')}}
                        </div>
                        ${{elName.full_name && elName.acronym ? 
                            `<div class="detail-subtitle">${{escape(elName.full_name)}}</div>` : ''}}
                        
                        <div class="badges">
                            <span class="badge bg-class">${{escape(mat.material_class)}}</span>
                            <span class="badge bg-conf-${{mat.confidence || 'low'}}">
                                Confidence: ${{mat.confidence}}
                            </span>
                            <span class="badge bg-source">
                                Source: ${{mat.data_source}}
                            </span>
                             ${{valid.audited_by_llm ? 
                                `<span class="badge" style="border:1px solid #ccc; background:white; color:#666">🤖 LLM Audited</span>` : ''}}
                        </div>
                    </div>
            `;
            
            // 1. Validation Warning
            if (!valid.is_valid && valid.issues.length > 0) {{
                html += `
                    <div class="validation-box">
                        <div class="validation-title">⚠️ Validation Issues</div>
                        <ul class="validation-list">
                            ${{valid.issues.map(i => `<li>${{escape(i)}}</li>`).join('')}}
                        </ul>
                    </div>
                `;
            }}
            
            // 2. Primary Data Grid
            html += `
                <div class="prop-grid">
                    <div class="prop-box">
                        <div class="prop-label">Ionic Conductivity</div>
                        <div class="prop-value">${{escape(mat.ionic_conductivity_S_per_cm)}} <span style="font-size:0.8em">S/cm</span></div>
                        
                        ${{typeof mat._norm_cond === 'number' ? 
                            `<div class="prop-meta">Normalized: ${{mat._norm_cond.toExponential(2)}}</div>` : ''}}
                    </div>
                    
                    <div class="prop-box">
                        <div class="prop-label">Temperature</div>
                        <div class="prop-value">${{escape(mat.measurement_temperature)}}</div>
                        
                        ${{typeof mat._norm_temp === 'number' ? 
                            `<div class="prop-meta">Normalized: ${{mat._norm_temp}} °C</div>` : ''}}
                    </div>
                </div>
            `;
            
            // 3. Evidence / Quote
            if (mat.exact_quote) {{
                html += `
                    <div style="margin-bottom: 2rem">
                        <div class="prop-label">Extracted Evidence</div>
                        <div class="quote-block">"${{escape(mat.exact_quote)}}"</div>
                        <div style="text-align: right; font-size: 0.8rem; color: #888;">
                            Location: ${{escape(mat.specific_source_location || 'Unknown')}}
                        </div>
                    </div>
                `;
            }}
            
            // 4. Meta Table
            html += `
                <div class="prop-label">Additional Details</div>
                <table class="meta-table">
                    ${{mat.electrolyte_name.proportion ? 
                        `<tr><td>Proportion</td><td>${{escape(mat.electrolyte_name.proportion)}}</td></tr>` : ''}}
                    <tr>
                        <td>Description</td>
                        <td>${{escape(mat.material_description || 'N/A')}}</td>
                    </tr>
                    <tr>
                        <td>Processing</td>
                        <td>${{escape(mat.processing_method || 'N/A')}}</td>
                    </tr>
                    ${{mat.source_node ? `
                    <tr>
                        <td>Source Node</td>
                        <td>
                            <strong>${{escape(mat.source_node.title)}}</strong><br>
                            <span style="color:#888; font-size:0.8em">Section: ${{escape(mat.source_node.section)}} (ID: ${{mat.source_node.node_id}})</span>
                        </td>
                    </tr>
                    ` : ''}}
                </table>
            `;
            
            html += `</div>`; // End detail card
            document.getElementById('detail-view').innerHTML = html;
        }}
        
        // Utils
        function escape(str) {{
            if (str === null || str === undefined) return '';
            return String(str)
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#039;");
        }}

        // Search
        document.getElementById('search').addEventListener('keyup', (e) => {{
            const term = e.target.value.toLowerCase();
            document.querySelectorAll('.mat-item').forEach(item => {{
                const text = item.innerText.toLowerCase();
                item.style.display = text.includes(term) ? 'block' : 'none';
            }});
        }});
        
        // Select first item on load
        if(materials.length > 0) selectMaterial(0);
    </script>
</body>
</html>
"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)


def generate_sidebar_html(by_type, type_order):
    """Generate the sidebar list HTML."""
    html = ""
    
    # Ensure all types in data are covered even if not in explicit type_order
    all_types = list(by_type.keys())
    sorted_types = sorted(all_types, key=lambda x: type_order.index(x) if x in type_order else 99)
    
    for mtype in sorted_types:
        materials = by_type[mtype]
        html += f'<div class="type-header">{escape(mtype)} ({len(materials)})</div>'
        
        for mat in materials:
            # Determine display name
            el = mat.get('electrolyte_name', {})
            display = el.get('acronym') or el.get('full_name') or 'Unknown'
            
            # Check validity for styling
            is_valid = mat.get('_validation', {}).get('is_valid', True)
            css_class = "mat-item invalid" if not is_valid else "mat-item"
            
            # Use the preserved index from the sorted list
            idx = mat['_ui_index']
            
            html += f'''
                <div class="{css_class}" id="mat-{idx}" onclick="selectMaterial({idx})">
                    <div class="mat-name">{escape(display)}</div>
                    <div class="mat-cond">{escape(mat.get('ionic_conductivity_S_per_cm', 'N/A'))} S/cm</div>
                </div>
            '''
            
    return html

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('materials_json')
    parser.add_argument('-o', '--output', help='Output HTML path')
    args = parser.parse_args()
    
    # Load JSON
    with open(args.materials_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    out_path = args.output if args.output else Path(args.materials_json).with_suffix('.html')
    generate_html(data, str(out_path))
    print(f"✓ Visualization saved to: {out_path}")
    print(f"  Open in browser: file://{Path(out_path).absolute()}")

