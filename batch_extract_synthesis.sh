#!/bin/bash

# ==============================================================================
# Batch Synthesis Extraction Script
# ==============================================================================
# This script iterates through subdirectories in a parent directory, finds the
# associated .md file, looks up the DOI from a mapping CSV, and runs the
# extract_synthesis_method.py script.

# Configuration
CSV_MAPPING="obelix_doi_yields_with_titles_normalized.csv"
GROUND_TRUTH="./OBELiX/data/processed.csv"
SCRIPT_PATH="./extract_synthesis_method.py"

# Function to retrieve DOI from title (normalized logic from test_doi_lookup.sh)
find_doi_from_title() {
    local paper_title="$1"
    local normalized_title
    
    # Normalize the title (lowercase, replace spaces/special chars with underscores)
    normalized_title=$(echo "$paper_title" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/_/g' | sed 's/_\+/_/g' | sed 's/^_//;s/_$//')
    
    # Search in CSV for matching title or DOI
    local doi=$(awk -F',' -v title="$normalized_title" '
        BEGIN { IGNORECASE=1 }
        NR>1 {
            # Normalize CSV title (Column 3)
            csv_title = tolower($3)
            gsub(/[^a-z0-9]/, "_", csv_title)
            gsub(/_+/, "_", csv_title)
            gsub(/^_|_$/, "", csv_title)
            
            # Normalize DOI for comparison (handles folders named after DOIs)
            csv_doi = tolower($1)
            gsub(/[^a-z0-9]/, "_", csv_doi)
            gsub(/_+/, "_", csv_doi)
            gsub(/^_|_$/, "", csv_doi)

            # Check for match (fuzzy on title or exact-ish on DOI)
            if (csv_title ~ title || title ~ csv_title || csv_doi == title) {
                print $1
                exit
            }
        }
    ' "$CSV_MAPPING")
    
    echo "$doi"
}

# Check if parent directory is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <parent_directory>"
    exit 1
fi

PARENT_DIR="$1"

if [ ! -d "$PARENT_DIR" ]; then
    echo "Error: Directory '$PARENT_DIR' not found."
    exit 1
fi

if [ ! -f "$CSV_MAPPING" ]; then
    echo "Warning: CSV mapping file not found at $CSV_MAPPING. DOI lookup might fail."
fi

echo "Starting batch synthesis extraction in: $PARENT_DIR"
echo "================================================================================"

# Iterate through subdirectories
for d in "$PARENT_DIR"/*/; do
    if [ -d "$d" ]; then
        folder_name=$(basename "$d")
        echo "Processing folder: $folder_name"
        
        # 1. Find the .md file
        md_file=$(find "$d" -maxdepth 1 -name "*.md" | head -n 1)
        
        if [ -z "$md_file" ]; then
            echo "  [SKIP] No .md file found in $folder_name"
            continue
        fi
        
        echo "  Found MD: $(basename "$md_file")"
        
        # 2. Lookup DOI
        doi=$(find_doi_from_title "$folder_name")
        
        if [ -z "$doi" ]; then
            echo "  [WARNING] Could not determine DOI for $folder_name. Running without GT context."
            doi_arg=""
        else
            echo "  Using DOI: $doi"
            doi_arg="--doi $doi"
        fi
        
        # 3. Run extraction script
        # Command: python extract_synthesis_method.py <md_file> --doi <doi> --ground-truth <gt> --asset_dir <dir>
        python3 "$SCRIPT_PATH" "$md_file" $doi_arg --ground-truth "$GROUND_TRUTH" --asset_dir "$d"
        
        echo "  Done."
        echo "--------------------------------------------------------------------------------"
    fi
done

echo "Batch processing complete."
