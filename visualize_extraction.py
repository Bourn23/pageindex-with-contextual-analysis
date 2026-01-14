#!/usr/bin/env python3
import json
import argparse
import math
import re
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

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
                x_val = 1000.0 / (temp + 273.15)
                y_val = math.log10(cond)
                
                el = mat.get('electrolyte_name', {})
                name = mat.get('canonical_formula') or el.get('full_name') or 'Unknown'
                conf = mat.get('confidence', 'low').lower()
                
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
                    'is_estimated': is_estimated_temp  # Pass this flag to the template
                })
            except (ValueError, ZeroDivisionError):
                continue
                
    return plot_points

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('materials_json', help="Path to the extracted materials JSON")
    parser.add_argument('-t', '--template', default='report_template.html', help="Path to template file")
    parser.add_argument('-a', '--asset_dir', default='', help="Directory prefix for image assets (e.g., 'figures/')")
    parser.add_argument('-o', '--output', help='Output HTML path')
    args = parser.parse_args()
    
    mat_path = Path(args.materials_json)
    
    # Infer structure file path
    structure_path = mat_path.parent / (mat_path.name.replace('_materials.json', '.json').replace('_results.json', '.json'))
    if not structure_path.exists():
        structure_path = mat_path.with_suffix('.json')

    print(f"Loading materials: {mat_path}")
    print(f"Loading structure: {structure_path}")
    
    with open(mat_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    structure_map = load_structure_map(structure_path)
    
    # --- Data Processing for View ---
    raw_materials = data.get('materials', [])
    materials = sorted(raw_materials, key=lambda m: (
        m.get('material_class', 'Other') or 'Other',
        (m.get('canonical_formula') or '').lower()
    ))
    
    # Augment data with UI fields
    for i, mat in enumerate(materials):
        mat['_ui_index'] = i
        mat['_ui_formula'] = format_formula(mat.get('canonical_formula') or mat.get('electrolyte_name', {}).get('full_name'))

    # Create Grouped Tree for Sidebar
    sidebar_tree = {}
    for mat in materials:
        mtype = mat.get('material_class', 'Other') or 'Other'
        formula = mat.get('canonical_formula') or "Unspecified"
        if mtype not in sidebar_tree: sidebar_tree[mtype] = {}
        if formula not in sidebar_tree[mtype]: sidebar_tree[mtype][formula] = []
        sidebar_tree[mtype][formula].append(mat)

    plot_data = prepare_plot_data(materials)
    
    # --- Rendering ---
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template(args.template)
    
    # Handle Asset Dir formatting
    asset_base = args.asset_dir
    if asset_base and not asset_base.endswith('/'):
        asset_base += '/'
    
    html_output = template.render(
        doc_name=data.get('source_file', 'Document'),
        materials=materials,
        structure_map=structure_map,
        plot_data=plot_data,
        sidebar_tree=sidebar_tree,
        asset_dir=asset_base
    )
    
    out_path = args.output if args.output else mat_path.with_suffix('.html')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html_output)
    
    print(f"✓ Visualization saved to: {out_path}")
    print(f"  Open in browser: file://{Path(out_path).absolute()}")

if __name__ == '__main__':
    main()