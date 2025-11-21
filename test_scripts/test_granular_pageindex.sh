#!/bin/bash

# Test script for granular PageIndex features

echo "=========================================="
echo "Testing PageIndex with Granular Features"
echo "=========================================="
echo ""

PDF_PATH="tests/pdfs/earthmover.pdf"

# Check if PDF exists
if [ ! -f "$PDF_PATH" ]; then
    echo "Error: PDF not found at $PDF_PATH"
    exit 1
fi

# Check if GEMINI_API_KEY is set
if [ -z "$GEMINI_API_KEY" ]; then
    echo "Warning: GEMINI_API_KEY not set"
    echo "Make sure it's in your .env file"
fi

echo "Test 1: Coarse granularity (baseline)"
echo "--------------------------------------"
python run_pageindex.py \
  --pdf_path "$PDF_PATH" \
  --granularity coarse \
  --if-add-node-summary no \
  --if-add-doc-description no

echo ""
echo "Test 2: Medium granularity (with figures, tables, semantic)"
echo "-----------------------------------------------------------"
python run_pageindex.py \
  --pdf_path "$PDF_PATH" \
  --granularity medium \
  --if-add-node-summary yes \
  --if-add-doc-description no

echo ""
echo "=========================================="
echo "Tests complete!"
echo "=========================================="
echo ""
echo "Check results in: results/earthmover_structure.json"
echo ""
echo "To compare node counts:"
echo "  Coarse: Should have ~8-12 nodes"
echo "  Medium: Should have ~30-50+ nodes"
