"""
Test that text extraction works for semantic subdivision.
"""

import sys
sys.path.insert(0, 'PageIndex')

from pageindex.granular.integration import _extract_text_for_node

# Test data
page_texts = [
    ("Page 1 text content here", 5),
    ("Page 2 text content here", 5),
    ("Page 3 text content here", 5),
    ("Page 4 text content here", 5),
]

# Test node spanning pages 2-3
node = {
    'title': 'Test Section',
    'start_index': 2,
    'end_index': 3
}

extracted_text = _extract_text_for_node(node, page_texts)

print("Test: Extract text for node spanning pages 2-3")
print("="*60)
print(f"Node: {node['title']}")
print(f"Pages: {node['start_index']}-{node['end_index']}")
print(f"\nExtracted text:")
print(extracted_text)
print("\n" + "="*60)

expected = "Page 2 text content here\n\nPage 3 text content here"
if extracted_text == expected:
    print("✓ Test PASSED")
else:
    print("✗ Test FAILED")
    print(f"Expected: {expected}")
    print(f"Got: {extracted_text}")
