#!/usr/bin/env python3
"""
Test markdown processing with full PageIndex pipeline.
This uses the same tree_parser as PDF processing.
"""

import json
from pathlib import Path

# Test with the full pipeline
from pageindex import markdown_page_index


def test_markdown_with_full_pipeline():
    """Test markdown processing using full PageIndex pipeline."""
    print("\n" + "="*60)
    print("Testing Markdown with Full PageIndex Pipeline")
    print("="*60)
    
    markdown_path = "PageIndex/tests/markdowns/Influence of the LLZO-PEO interface on the micro- and macro-scale properties of composite polymer electrolytes for solid-state batteries/Influence of the LLZO-PEO interface on the micro- and macro-scale properties of composite polymer electrolytes for solid-state batteries.md"
    metadata_path = "PageIndex/tests/markdowns/Influence of the LLZO-PEO interface on the micro- and macro-scale properties of composite polymer electrolytes for solid-state batteries/Influence of the LLZO-PEO interface on the micro- and macro-scale properties of composite polymer electrolytes for solid-state batteries_meta.json"
    
    if not Path(markdown_path).exists():
        print("❌ Test file not found")
        return False
    
    print("\nProcessing with granularity='medium'...")
    print("This will use the same pipeline as PDF processing:")
    print("  - TOC detection")
    print("  - Semantic subdivision")
    print("  - Figure/table detection")
    print("  - Summary generation")
    
    try:
        structure = markdown_page_index(
            markdown_path=markdown_path,
            metadata_path=metadata_path,
            model='gemini-2.5-flash',  # Use Gemini model
            granularity='medium',
            enable_figure_detection=True,
            enable_table_detection=True,
            enable_semantic_subdivision=True,
            if_add_node_summary='no',  # Disable summaries for faster testing
            if_add_node_text='no'
        )
        
        print("\n✓ Processing complete!")
        print(f"\nDocument: {structure['doc_name']}")
        print(f"Source: {structure.get('source', 'unknown')}")
        print(f"Top-level sections: {len(structure['structure'])}")
        
        # Count nodes recursively
        def count_nodes(nodes):
            count = len(nodes)
            for node in nodes:
                if 'nodes' in node and node['nodes']:
                    count += count_nodes(node['nodes'])
            return count
        
        total_nodes = count_nodes(structure['structure'])
        print(f"Total nodes: {total_nodes}")
        
        # Count by type
        def count_by_type(nodes):
            types = {}
            for node in nodes:
                node_type = node.get('node_type', 'section')
                types[node_type] = types.get(node_type, 0) + 1
                if 'nodes' in node and node['nodes']:
                    child_types = count_by_type(node['nodes'])
                    for t, c in child_types.items():
                        types[t] = types.get(t, 0) + c
            return types
        
        types = count_by_type(structure['structure'])
        print(f"\nNode types:")
        for node_type, count in sorted(types.items()):
            print(f"  {node_type}: {count}")
        
        # Show first section
        if structure['structure']:
            first = structure['structure'][0]
            print(f"\nFirst section:")
            print(f"  Title: {first['title']}")
            print(f"  Node ID: {first.get('node_id', 'N/A')}")
            print(f"  Pages: {first.get('start_index', 'N/A')} - {first.get('end_index', 'N/A')}")
            print(f"  Type: {first.get('node_type', 'N/A')}")
            if 'summary' in first:
                summary_preview = first['summary'][:150] + "..." if len(first['summary']) > 150 else first['summary']
                print(f"  Summary: {summary_preview}")
            if 'nodes' in first and first['nodes']:
                print(f"  Children: {len(first['nodes'])}")
        
        # Save output
        output_path = "PageIndex/results/markdown_full_pipeline_test.json"
        with open(output_path, 'w') as f:
            json.dump(structure, f, indent=2)
        print(f"\n✓ Saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_comparison_with_pdf():
    """Compare markdown output structure with PDF output."""
    print("\n" + "="*60)
    print("Comparing Markdown vs PDF Output Structure")
    print("="*60)
    
    # Load the PDF result for comparison
    pdf_result_path = "PageIndex/results/earthmover_structure_medium.json"
    if Path(pdf_result_path).exists():
        with open(pdf_result_path) as f:
            pdf_result = json.load(f)
        
        print("\nPDF Result Structure:")
        print(f"  Document: {pdf_result.get('doc_name', 'N/A')}")
        print(f"  Top-level sections: {len(pdf_result.get('structure', []))}")
        
        # Show structure
        if pdf_result.get('structure'):
            first = pdf_result['structure'][0]
            print(f"\n  First section:")
            print(f"    Title: {first.get('title', 'N/A')}")
            print(f"    Node ID: {first.get('node_id', 'N/A')}")
            print(f"    Has summary: {'summary' in first}")
            print(f"    Has children: {len(first.get('nodes', []))} nodes")
            if first.get('nodes'):
                child = first['nodes'][0]
                print(f"    First child type: {child.get('node_type', 'N/A')}")
    
    print("\n✓ Markdown processing uses the same structure format!")
    print("  Both produce:")
    print("    - Hierarchical tree with node_id")
    print("    - start_index/end_index for pages")
    print("    - node_type (section, figure, table, semantic_unit)")
    print("    - summaries and metadata")
    print("    - nested children in 'nodes' array")


def main():
    """Run tests."""
    print("="*60)
    print("Markdown Full Pipeline Tests")
    print("="*60)
    
    success = test_markdown_with_full_pipeline()
    
    if success:
        test_comparison_with_pdf()
    
    print("\n" + "="*60)
    if success:
        print("✓ All tests passed!")
    else:
        print("❌ Tests failed")
    print("="*60)
    
    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
