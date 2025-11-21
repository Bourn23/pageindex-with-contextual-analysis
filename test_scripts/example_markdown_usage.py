#!/usr/bin/env python3
"""
Simple example showing how to use markdown processing in PageIndex.
"""

import json
from pageindex import markdown_page_index
from pageindex.utils import print_toc


def main():
    # Path to your markdown file (e.g., from marker library)
    markdown_path = "PageIndex/tests/markdowns/Influence of the LLZO-PEO interface on the micro- and macro-scale properties of composite polymer electrolytes for solid-state batteries/Influence of the LLZO-PEO interface on the micro- and macro-scale properties of composite polymer electrolytes for solid-state batteries.md"
    
    # Optional metadata file
    metadata_path = "PageIndex/tests/markdowns/Influence of the LLZO-PEO interface on the micro- and macro-scale properties of composite polymer electrolytes for solid-state batteries/Influence of the LLZO-PEO interface on the micro- and macro-scale properties of composite polymer electrolytes for solid-state batteries_meta.json"
    
    print("Processing markdown file...")
    
    # Process the markdown
    structure = markdown_page_index(
        markdown_path=markdown_path,
        metadata_path=metadata_path,
        opt={
            'if_add_node_text': 'no',      # Set to 'yes' to include text
            'if_add_node_summary': 'no',   # Set to 'yes' to generate summaries
            'if_add_node_id': 'yes'        # Add hierarchical IDs
        }
    )
    
    # Display results
    print(f"\n✓ Processing complete!")
    print(f"  Pages: {structure['page_count']}")
    print(f"  Sections: {len(structure['tree'])}")
    print(f"  Figures: {len(structure.get('figures', []))}")
    print(f"  Tables: {len(structure.get('tables', []))}")
    
    print("\nDocument Structure:")
    print_toc(structure['tree'])
    
    # Show first section details
    if structure['tree']:
        first = structure['tree'][0]
        print(f"\nFirst Section:")
        print(f"  ID: {first['node_id']}")
        print(f"  Title: {first['title']}")
        print(f"  Pages: {first.get('start_page', 'N/A')} - {first.get('end_page', 'N/A')}")
    
    # Show figures
    if structure.get('figures'):
        print(f"\nFigures:")
        for fig in structure['figures'][:3]:
            print(f"  - Figure {fig['figure_number']} on page {fig['page']}")
    
    # Save to file
    output_path = "PageIndex/results/example_markdown_output.json"
    with open(output_path, 'w') as f:
        json.dump(structure, f, indent=2)
    print(f"\n✓ Saved to: {output_path}")


if __name__ == "__main__":
    main()
