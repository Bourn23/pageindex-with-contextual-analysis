#!/bin/bash

# Quick test for medium granularity with verbose output

echo "=========================================="
echo "Quick Test: Medium Granularity"
echo "=========================================="
echo ""
echo "This will:"
echo "  1. Run PageIndex with medium granularity"
echo "  2. Show progress as it runs"
echo "  3. Should take 30-60 seconds"
echo ""
echo "Starting..."
echo ""

python run_pageindex.py \
  --pdf_path tests/pdfs/earthmover.pdf \
  --granularity medium \
  --if-add-node-summary yes \
  --if-add-doc-description no 2>&1 | tee /tmp/pageindex_test.log

echo ""
echo "=========================================="
echo "Test complete!"
echo "=========================================="
echo ""

# Quick analysis
if [ -f "results/earthmover_structure.json" ]; then
    echo "✓ Output file created"
    
    # Count nodes
    node_count=$(python -c "
import json
with open('results/earthmover_structure.json') as f:
    data = json.load(f)
    
def count_nodes(nodes):
    count = len(nodes)
    for node in nodes:
        if 'nodes' in node and node['nodes']:
            count += count_nodes(node['nodes'])
    return count

tree = data.get('nodes', data.get('structure', []))
print(count_nodes(tree))
" 2>/dev/null)
    
    echo "  Total nodes: $node_count"
    
    # Check for semantic units
    semantic_count=$(grep -o '"node_type": "semantic_unit"' results/earthmover_structure.json 2>/dev/null | wc -l | tr -d ' ')
    echo "  Semantic units: $semantic_count"
    
    if [ "$semantic_count" -gt 0 ]; then
        echo ""
        echo "🎉 SUCCESS! Semantic subdivision is working!"
    else
        echo ""
        echo "⚠️  No semantic units found. Check the log above for errors."
    fi
else
    echo "✗ Output file not created"
    echo "  Check /tmp/pageindex_test.log for errors"
fi
