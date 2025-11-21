"""Simple test to debug the tree structure issue."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from pageindex import page_index_main
from pageindex.utils import ConfigLoader
import json

# Test with one PDF
pdf_path = 'tests/pdfs/four-lectures.pdf'

config_loader = ConfigLoader()
opt = config_loader.load({
    'model': 'gemini-2.5-flash-lite',
    'granularity': 'medium',
    'enable_figure_detection': True,
    'enable_table_detection': True,
    'enable_semantic_subdivision': True,
    'if_add_node_id': 'yes',
    'if_add_node_summary': 'no',
    'if_add_doc_description': 'no'
})

print("Processing PDF...")
result = page_index_main(pdf_path, opt)

# Extract tree structure
if isinstance(result, dict) and 'structure' in result:
    tree = result['structure']
else:
    tree = result

print(f"\nTree type: {type(tree)}")
print(f"Tree length: {len(tree) if isinstance(tree, list) else 'N/A'}")

if isinstance(tree, list) and len(tree) > 0:
    print(f"\nFirst element type: {type(tree[0])}")
    print(f"First element keys: {tree[0].keys() if isinstance(tree[0], dict) else 'N/A'}")
    print(f"\nTotal nodes: {sum(1 for _ in tree)}")
    print("\n✓ Tree structure looks good!")
