#!/usr/bin/env python3
"""
Convert extracted materials JSON to CSV format.

Usage:
    python materials_to_csv.py results/paper_materials.json
    python materials_to_csv.py results/paper_materials.json --output materials.csv
"""

import argparse
import csv
import json
from pathlib import Path


def materials_to_csv(materials_data, output_path):
    """Convert materials JSON to CSV format."""
    
    materials = materials_data.get('materials', [])
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            'Abbreviation',
            'Full Name',
            'Material Type',
            'Compositions',
            'Processing Methods',
            'Source Count',
            'Source Node IDs',
            'Source Sections'
        ])
        
        # Write data
        for mat in materials:
            abbrev = mat.get('abbreviation', '')
            full_name = mat.get('full_name', '')
            material_type = mat.get('material_type', '')
            
            # Join lists with semicolons
            compositions = '; '.join(mat.get('compositions', []))
            processing = '; '.join(mat.get('processing_methods', []))
            
            # Source information
            sources = mat.get('source_nodes', [])
            source_count = len(sources)
            source_ids = '; '.join([s.get('node_id', '') for s in sources])
            source_sections = '; '.join([s.get('section', '') for s in sources])
            
            writer.writerow([
                abbrev,
                full_name,
                material_type,
                compositions,
                processing,
                source_count,
                source_ids,
                source_sections
            ])


def materials_to_detailed_csv(materials_data, output_path):
    """
    Convert materials JSON to detailed CSV with one row per source node.
    This format is better for analysis when you want to see each mention separately.
    """
    
    materials = materials_data.get('materials', [])
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            'Abbreviation',
            'Full Name',
            'Material Type',
            'Composition',
            'Processing Method',
            'Source Node ID',
            'Source Node Title',
            'Source Node Type',
            'Source Section'
        ])
        
        # Write data - one row per source node
        for mat in materials:
            abbrev = mat.get('abbreviation', '')
            full_name = mat.get('full_name', '')
            material_type = mat.get('material_type', '')
            compositions = mat.get('compositions', [])
            processing_methods = mat.get('processing_methods', [])
            sources = mat.get('source_nodes', [])
            
            # If no sources, write one row with material info only
            if not sources:
                writer.writerow([
                    abbrev,
                    full_name,
                    material_type,
                    '; '.join(compositions),
                    '; '.join(processing_methods),
                    '',
                    '',
                    '',
                    ''
                ])
            else:
                # Write one row per source
                for source in sources:
                    writer.writerow([
                        abbrev,
                        full_name,
                        material_type,
                        '; '.join(compositions),
                        '; '.join(processing_methods),
                        source.get('node_id', ''),
                        source.get('title', ''),
                        source.get('node_type', ''),
                        source.get('section', '')
                    ])


def main():
    parser = argparse.ArgumentParser(
        description='Convert materials JSON to CSV format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output Formats:
  --format summary   One row per material (default)
  --format detailed  One row per source node (better for analysis)

Examples:
  python materials_to_csv.py results/paper_materials.json
  python materials_to_csv.py results/paper_materials.json --format detailed
  python materials_to_csv.py results/paper_materials.json -o materials.csv
        """
    )
    
    parser.add_argument('materials_json', help='Path to materials JSON file')
    parser.add_argument('--output', '-o', help='Output CSV file path')
    parser.add_argument(
        '--format', '-f',
        choices=['summary', 'detailed'],
        default='summary',
        help='Output format (default: summary)'
    )
    
    args = parser.parse_args()
    
    # Load materials
    input_path = Path(args.materials_json)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        return 1
    
    with open(input_path, 'r', encoding='utf-8') as f:
        materials_data = json.load(f)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        if args.format == 'detailed':
            # Replace .json with _detailed.csv
            output_path = Path(str(input_path).replace('.json', '_detailed.csv'))
        else:
            output_path = input_path.with_suffix('.csv')
    
    # Convert to CSV
    print(f"Converting materials to CSV ({args.format} format)...")
    
    if args.format == 'detailed':
        materials_to_detailed_csv(materials_data, output_path)
    else:
        materials_to_csv(materials_data, output_path)
    
    print(f"✓ CSV saved to: {output_path}")
    
    # Print stats
    materials = materials_data.get('materials', [])
    print(f"  Materials: {len(materials)}")
    
    if args.format == 'detailed':
        total_rows = sum(max(1, len(m.get('source_nodes', []))) for m in materials)
        print(f"  Total rows: {total_rows}")
    
    return 0


if __name__ == '__main__':
    exit(main())
