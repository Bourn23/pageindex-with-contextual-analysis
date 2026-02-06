#!/bin/bash

# =============================================================================
# Batch Data Extraction Pipeline v5
# =============================================================================
# This script automates the processing of multiple papers by iterating through
# subdirectories of a parent folder and calling run_pipeline_v5.sh for each.
#
# Usage:
#   ./batch_run_pipeline_v5.sh <parent_dir>
#
# Example:
#   ./batch_run_pipeline_v5.sh "./fetched_papers/obelix_md"
# =============================================================================

set -e
set -o pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}=========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}=========================================${NC}"
}

print_info() {
    echo -e "${BLUE}[BATCH INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[BATCH SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[BATCH ERROR]${NC} $1"
}

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <parent_dir>"
    echo ""
    echo "Example: $0 './fetched_papers/obelix_md'"
    exit 1
fi

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    echo "Usage: $0 <parent_dir>"
    exit 0
fi

PARENT_DIR="$1"

if [ ! -d "$PARENT_DIR" ]; then
    print_error "Directory not found: $PARENT_DIR"
    exit 1
fi

PIPELINE_SCRIPT="./run_pipeline_v5.sh"

if [ ! -f "$PIPELINE_SCRIPT" ]; then
    print_error "v5 Pipeline script not found: $PIPELINE_SCRIPT"
    exit 1
fi

# Track statistics
TOTAL=0
SUCCESS=0
FAILED=0
SKIPPED=0

# Summary log for the batch
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BATCH_SUMMARY_FILE="batch_summary_v5_${TIMESTAMP}.log"

print_header "STARTING BATCH PROCESSING v5"
print_info "Parent Directory: $PARENT_DIR"
echo "-----------------------------------------"

# Iterate through subdirectories
REALS=()
while IFS= read -r -d '' dir; do
    REALS+=("$dir")
done < <(find "$PARENT_DIR" -maxdepth 1 -mindepth 1 -type d -print0)

for folder in "${REALS[@]}"; do
    TOTAL=$((TOTAL + 1))
    folder_name=$(basename "$folder")
    
    print_info "Processing ($TOTAL/${#REALS[@]}): $folder_name"
    
    # Check if it contains an MD file
    if ! find "$folder" -maxdepth 1 -name "*.md" -type f | grep -q .; then
        print_info "Skipping: No .md file found in $folder_name"
        SKIPPED=$((SKIPPED + 1))
        echo "$folder_name: SKIPPED (No MD file)" >> "$BATCH_SUMMARY_FILE"
        echo ""
        continue
    fi
    
    # Run the v5 pipeline
    if "$PIPELINE_SCRIPT" "$folder"; then
        print_success "Completed: $folder_name"
        SUCCESS=$((SUCCESS + 1))
        echo "$folder_name: SUCCESS" >> "$BATCH_SUMMARY_FILE"
    else
        print_error "Sub-pipeline failed: $folder_name"
        FAILED=$((FAILED + 1))
        echo "$folder_name: FAILED" >> "$BATCH_SUMMARY_FILE"
    fi
    echo ""
done

# Final Summary
print_header "BATCH PROCESSING COMPLETE v5"
echo "Total Folders: $TOTAL"
echo -e "Success:       ${GREEN}$SUCCESS${NC}"
echo -e "Failed:        ${RED}$FAILED${NC}"
echo -e "Skipped:       ${YELLOW}$SKIPPED${NC}"
echo "-----------------------------------------"
echo "Log saved to: $BATCH_SUMMARY_FILE"
echo ""
