#!/bin/bash

# Script to run run_advanced_pipeline.py on multiple *_materials.json files
# Usage: ./batch_run_advanced.sh [parent_directory]

PARENT_DIR=${1:-"."}
PYTHON_SCRIPT="run_advanced_pipeline.py"

# Check if the python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: $PYTHON_SCRIPT not found in the current directory."
    exit 1
fi

# Find all *_materials.json files
echo "Searching for *_materials.json files in $PARENT_DIR..."
FILES=$(find "$PARENT_DIR" -name "*_materials.json" -type f)
COUNT=$(echo "$FILES" | grep -c "_materials.json" || echo 0)

if [ "$COUNT" -eq 0 ]; then
    echo "No matching files found."
    exit 0
fi

echo "Found $COUNT files to process."
echo "--------------------------------------------------"

# Iterate and run the pipeline
while IFS= read -r file; do
    echo "Processing: $file"
    python "$PYTHON_SCRIPT" "$file"
    echo "--------------------------------------------------"
done <<< "$FILES"

echo "Batch processing complete."
