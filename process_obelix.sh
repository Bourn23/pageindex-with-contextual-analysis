#!/bin/bash

# Set the input directory
INPUT_DIR="fetched_papers/obelix_md2"
SCRIPT_PATH="markdown_v3.py"

# Check if the script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: $SCRIPT_PATH not found in the current directory."
    exit 1
fi

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Directory $INPUT_DIR not found."
    exit 1
fi

echo "Starting batch processing of markdown files in $INPUT_DIR..."

# Find all .md files in the subdirectories (at depth 2)
# Using -print0 and read -d '' to safely handle filenames with spaces or special characters
find "$INPUT_DIR" -maxdepth 2 -name "*.md" -print0 | while IFS= read -r -d '' md_file; do
    echo "--------------------------------------------------"
    echo "Processing: $md_file"
    
    # Run the python script
    python3 "$SCRIPT_PATH" "$md_file"
    
    # Check if the previous command was successful
    if [ $? -eq 0 ]; then
        echo "Successfully processed: $md_file"
    else
        echo "Error processing: $md_file"
    fi
done

echo "--------------------------------------------------"
echo "Batch processing complete."
