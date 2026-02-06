#!/bin/bash

# =============================================================================
# Unified Data Extraction Pipeline
# =============================================================================
# This script automates the complete data extraction and evaluation pipeline:
#   1. Convert MD file to JSON structure (markdown_v3.py)
#   2. Extract material data from structure (run_extraction_md.py)
#   3. Evaluate extraction against ground truth dataset (llm_evaluator.py)
#
# Usage:
#   ./run_pipeline.sh &lt;path_to_md_folder&gt; [doi]
#
# Example:
#   ./run_pipeline.sh "./fetched_papers/obelix_md/Effect of Si substitution on the structural and transport properties of superionic Li-argyrodites"
#   ./run_pipeline.sh "./fetched_papers/obelix_md/Effect of Si substitution on the structural and transport properties of superionic Li-argyrodites" --skip-step1 --skip-step3
#   ./run_pipeline.sh "./fetched_papers/obelix_md/Effect of Si substitution on the structural and transport properties of superionic Li-argyrodites" "10.1039/c7ta08581h"
# =============================================================================

set -e  # Exit on error
set -o pipefail  # Ensure pipe errors are caught

# Initialize conda/mamba
# This is required because mamba is a shell function, not a direct executable
CONDA_BASE="/Users/bourn23/miniforge3"
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
fi
if [ -f "$CONDA_BASE/etc/profile.d/mamba.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/mamba.sh"
fi

# Mamba environment name
MAMBA_ENV="pokeagent"

# Helper function to run Python commands in the mamba environment
run_python() {
    # Set PYTHONUNBUFFERED=1 and use -u flag for real-time log streaming
    export PYTHONUNBUFFERED=1
    mamba run -n "$MAMBA_ENV" python -u "$@"
}

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CSV_MAPPING="./obelix_doi_yields_with_titles_normalized.csv"
DATASET_CSV="./OBELiX/data/processed.csv"
RESULTS_DIR="./fetched_papers/obelix_parsed2"

# =============================================================================
# Helper Functions
# =============================================================================

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_step() {
    echo -e "${GREEN}[STEP $1]${NC} $2"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Function to find DOI from paper title
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

# =============================================================================
# Main Pipeline
# =============================================================================

main() {
    # Initialize variables
    MD_FOLDER=""
    PROVIDED_DOI=""
    SKIP_STEP1=false
    SKIP_STEP2=false
    SKIP_STEP3=false

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-step1)
                SKIP_STEP1=true
                shift
                ;;
            --skip-step2)
                SKIP_STEP2=true
                shift
                ;;
            --skip-step3)
                SKIP_STEP3=true
                shift
                ;;
            --help|-h)
                echo "Usage: $0 <path_to_md_folder> [doi] [--skip-step1] [--skip-step2] [--skip-step3]"
                exit 0
                ;;
            -*)
                print_error "Unknown option: $1"
                exit 1
                ;;
            *)
                if [ -z "$MD_FOLDER" ]; then
                    MD_FOLDER="$1"
                elif [ -z "$PROVIDED_DOI" ]; then
                    PROVIDED_DOI="$1"
                else
                    print_error "Unexpected argument: $1"
                    exit 1
                fi
                shift
                ;;
        esac
    done

    # Check required arguments
    if [ -z "$MD_FOLDER" ]; then
        print_error "Missing required argument: path to MD folder"
        echo ""
        echo "Usage: $0 <path_to_md_folder> [doi] [--skip-step1] [--skip-step2] [--skip-step3]"
        echo ""
        echo "Example:"
        echo "  $0 './fetched_papers/obelix_md/Paper_Folder'"
        echo "  $0 './fetched_papers/obelix_md/Paper_Folder' --skip-step3"
        exit 1
    fi
    
    # Validate MD folder exists
    if [ ! -d "$MD_FOLDER" ]; then
        print_error "MD folder not found: $MD_FOLDER"
        exit 1
    fi
    
    # Find the .md file in the folder
    MD_FILE=$(find "$MD_FOLDER" -maxdepth 1 -name "*.md" -type f | head -n 1)
    
    if [ -z "$MD_FILE" ]; then
        print_error "No .md file found in: $MD_FOLDER"
        exit 1
    fi
    
    # Extract paper title from folder name
    PAPER_TITLE=$(basename "$MD_FOLDER")
    
    # Create log file with timestamp
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    BASENAME=$(basename "$MD_FILE" .md)
    LOG_FILE="$RESULTS_DIR/${BASENAME}_pipeline_${TIMESTAMP}.log"
    
    # Ensure results directory exists
    mkdir -p "$RESULTS_DIR"
    
    # Function to log output to both console and file
    log_output() {
        tee -a "$LOG_FILE"
    }
    
    # Start logging
    {
        print_header "UNIFIED DATA EXTRACTION PIPELINE"
        print_info "Paper: $PAPER_TITLE"
        print_info "MD File: $MD_FILE"
        print_info "Asset Directory: $MD_FOLDER"
        print_info "Log File: $LOG_FILE"
        echo ""
    } | log_output
    
    # =============================================================================
    # STEP 1: Convert MD to JSON Structure
    # =============================================================================
    STRUCTURE_JSON="$RESULTS_DIR/${BASENAME}_structure.json"

    if [ "$SKIP_STEP1" = true ]; then
        {
            print_info "Skipping STEP 1: Using existing structure: $STRUCTURE_JSON"
            echo ""
        } | log_output
    else
        {
            print_step "1/3" "Converting MD file to JSON structure..."
            echo ""
        } | log_output
        
        if run_python markdown_v3.py "$MD_FILE" --output "$STRUCTURE_JSON" 2>&1 | log_output; then
            {
                print_info "✓ Structure saved to: $STRUCTURE_JSON"
                echo ""
            } | log_output
        else
            {
                print_error "Failed to convert MD to JSON structure"
            } | log_output
            exit 1
        fi
    fi
    
    # =============================================================================
    # STEP 2: Extract Material Data
    # =============================================================================
    MATERIALS_JSON="$RESULTS_DIR/${BASENAME}_structure_materials.json"

    if [ "$SKIP_STEP2" = true ]; then
        {
            print_info "Skipping STEP 2: Using existing materials: $MATERIALS_JSON"
            echo ""
        } | log_output
    else
        {
            print_step "2/3" "Extracting material data from structure..."
            echo ""
        } | log_output
        
        if run_python run_extraction_md.py "$STRUCTURE_JSON" --asset_dir "$MD_FOLDER" --output "$MATERIALS_JSON" 2>&1 | log_output; then
            {
                print_info "✓ Materials extracted to: $MATERIALS_JSON"
                echo ""
            } | log_output
        else
            {
                print_error "Failed to extract material data"
            } | log_output
            exit 1
        fi
    fi
    
    # =============================================================================
    # STEP 3: Evaluate Extraction
    # =============================================================================
    if [ "$SKIP_STEP3" = true ]; then
        {
            print_info "Skipping STEP 3: Evaluation"
            echo ""
        } | log_output
    else
        {
            print_step "3/3" "Evaluating extraction against ground truth..."
        } | log_output
        
        # Determine DOI
        if [ -n "$PROVIDED_DOI" ]; then
            DOI="$PROVIDED_DOI"
            {
                print_info "Using provided DOI: $DOI"
            } | log_output
        else
            {
                print_info "Searching for DOI based on paper title..."
            } | log_output
            DOI=$(find_doi_from_title "$PAPER_TITLE")
            
            if [ -z "$DOI" ]; then
                {
                    print_warning "Could not find DOI for paper: $PAPER_TITLE"
                    print_warning "Skipping evaluation step. You can manually run:"
                    print_warning "  python llm_evaluator.py --extraction '$MATERIALS_JSON' --dataset '$DATASET_CSV' --doi 'YOUR_DOI'"
                    print_info ""
                    print_header "PIPELINE COMPLETED (WITHOUT EVALUATION)"
                    print_info "Structure JSON: $STRUCTURE_JSON"
                    print_info "Materials JSON: $MATERIALS_JSON"
                    print_info "Log File: $LOG_FILE"
                } | log_output
                exit 0
            else
                {
                    print_info "Found DOI: $DOI"
                } | log_output
            fi
        fi
        
        # Check if dataset exists
        if [ ! -f "$DATASET_CSV" ]; then
            {
                print_warning "Ground truth dataset not found: $DATASET_CSV"
                print_warning "Skipping evaluation step."
                print_info ""
                print_header "PIPELINE COMPLETED (WITHOUT EVALUATION)"
                print_info "Structure JSON: $STRUCTURE_JSON"
                print_info "Materials JSON: $MATERIALS_JSON"
                print_info "Log File: $LOG_FILE"
            } | log_output
            exit 0
        fi
        
        # Run evaluation
        if run_python llm_evaluator.py --extraction "$MATERIALS_JSON" --dataset "$DATASET_CSV" --doi "$DOI" 2>&1 | log_output; then
            {
                print_info "✓ Evaluation complete"
                print_info "✓ Report saved to: llm_evaluation_report.csv"
                echo ""
            } | log_output
        else
            {
                print_error "Evaluation failed"
            } | log_output
            exit 1
        fi
    fi
    
    # =============================================================================
    # Summary
    # =============================================================================
    {
        print_header "PIPELINE COMPLETED SUCCESSFULLY"
        print_info "Paper: $PAPER_TITLE"
        print_info "DOI: $DOI"
        echo ""
        print_info "Generated Files:"
        print_info "  1. Structure JSON:    $STRUCTURE_JSON"
        print_info "  2. Materials JSON:    $MATERIALS_JSON"
        print_info "  3. Evaluation Report: llm_evaluation_report.csv"
        print_info "  4. Pipeline Log:      $LOG_FILE"
        echo ""
    } | log_output
}

# Run main function
main "$@"
