#!/bin/bash

# Configuration
CSV_MAPPING="obelix_doi_yields_with_titles_normalized.csv"

# Function to retrieve DOI from title (normalized logic)
find_doi_from_title() {
    local paper_title="$1"
    local normalized_title
    
    # Normalize the title (lowercase, replace spaces/special chars with underscores)
    normalized_title=$(echo "$paper_title" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/_/g' | sed 's/_\+/_/g' | sed 's/^_//;s/_$//')
    
    # Search in CSV for matching title or DOI
    local doi=$(awk -F',' -v title="$normalized_title" '
        BEGIN { IGNORECASE=1 }
        NR>1 {
            # Normalize CSV title
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

# Test helper
run_test() {
    local test_name="$1"
    local input_title="$2"
    local expected_doi="$3"
    
    echo "Running Test: $test_name"
    echo "Input Title: $input_title"
    
    local actual_doi=$(find_doi_from_title "$input_title")
    
    if [ "$actual_doi" == "$expected_doi" ]; then
        echo -e "\033[0;32m[PASS]\033[0m Found DOI: $actual_doi"
    else
        echo -e "\033[0;31m[FAIL]\033[0m Expected: $expected_doi, Got: $actual_doi"
    fi
    echo "----------------------------------------"
}

# Suite to run standard tests
run_test_suite() {
    echo "Running DOI Lookup Test Suite"
    echo "CSV Mapping: $CSV_MAPPING"
    echo "========================================"
    run_test "Exact Match" "ionic_conductivity_of_solid_electrolytes_based_on_lithium_titanium_phosphate" "10.1149/1.2086597"
    run_test "Case/Special Chars" "Ionic Conductivity of Solid Electrolytes Based on Lithium Titanium Phosphate!" "10.1149/1.2086597"
    run_test "Partial Match" "nasicon-type li1_xmxti2_x_po4_3" "10.1016/j.ssi.2014.07.018"
    run_test "No Match" "A Completely Random Title That Does Not Exist" ""
}

# Process a directory of subdirectories
process_directory() {
    local target_dir="$1"
    echo "Processing Directory: $target_dir"
    echo "========================================"
    echo -e "Folder Name\tDOI"
    echo -e "----------------------------------------"

    for d in "$target_dir"/*/; do
        if [ -d "$d" ]; then
            local folder_name=$(basename "$d")
            local doi=$(find_doi_from_title "$folder_name")
            echo -e "$folder_name\t\033[0;34m$doi\033[0m"
        fi
    done
}

# Main Logic
if [ ! -f "$CSV_MAPPING" ]; then
    echo "Error: $CSV_MAPPING not found!"
    exit 1
fi

case "$1" in
    "test")
        run_test_suite
        ;;
    "")
        echo "Usage: $0 [test | directory_path]"
        echo "  test           - Run the internal test suite"
        echo "  directory_path - Process subdirectories in the given path"
        exit 1
        ;;
    *)
        if [ -d "$1" ]; then
            process_directory "$1"
        else
            echo "Error: '$1' is not a valid directory or command."
            exit 1
        fi
        ;;
esac
