#!/bin/bash

# Script to test all granularity levels and compare results

set -e  # Exit on error

echo "=========================================="
echo "PageIndex Granularity Test Suite"
echo "=========================================="
echo ""

PDF="tests/pdfs/earthmover.pdf"

# Check if PDF exists
if [ ! -f "$PDF" ]; then
    echo "Error: PDF not found at $PDF"
    exit 1
fi

# Check if GEMINI_API_KEY is set
if [ -z "$GEMINI_API_KEY" ]; then
    echo "Warning: GEMINI_API_KEY not set in environment"
    echo "Checking .env file..."
    if [ -f ".env" ]; then
        export $(cat .env | grep GEMINI_API_KEY | xargs)
        echo "✓ Loaded GEMINI_API_KEY from .env"
    else
        echo "Error: No .env file found"
        exit 1
    fi
fi

echo "Test 1: Coarse Granularity (Baseline)"
echo "--------------------------------------"
python run_pageindex.py \
  --pdf_path "$PDF" \
  --granularity coarse \
  --if-add-node-summary yes \
  --if-add-doc-description no

mv results/earthmover_structure.json results/earthmover_structure_coarse.json
echo "✓ Coarse complete"
echo ""

echo "Test 2: Medium Granularity (1-level semantic)"
echo "----------------------------------------------"
python run_pageindex.py \
  --pdf_path "$PDF" \
  --granularity medium \
  --if-add-node-summary yes \
  --if-add-doc-description no

mv results/earthmover_structure.json results/earthmover_structure_medium.json
echo "✓ Medium complete"
echo ""

echo "Test 3: Fine Granularity (2-level recursive)"
echo "---------------------------------------------"
python run_pageindex.py \
  --pdf_path "$PDF" \
  --granularity fine \
  --if-add-node-summary yes \
  --if-add-doc-description no

mv results/earthmover_structure.json results/earthmover_structure_fine.json
echo "✓ Fine complete"
echo ""

echo "=========================================="
echo "Comparing Results"
echo "=========================================="
python compare_granularity.py

echo ""
echo "=========================================="
echo "Checking Semantic Units"
echo "=========================================="
python test_semantic_subdivision.py

echo ""
echo "=========================================="
echo "All tests complete!"
echo "=========================================="
echo ""
echo "Results saved in:"
echo "  - results/earthmover_structure_coarse.json"
echo "  - results/earthmover_structure_medium.json"
echo "  - results/earthmover_structure_fine.json"
