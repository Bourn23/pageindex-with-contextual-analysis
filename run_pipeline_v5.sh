#!/bin/bash

# =============================================================================
# Unified Data Extraction Pipeline v5 (Direct MD -> Extraction)
# =============================================================================
# This script automates the data extraction and evaluation pipeline:
#   1. Extract data directly from MD file (basic_extraction_md_v5.py)
#   2. Evaluate extraction against ground truth dataset (basic_evaluation.py)
#
# Usage:
#   ./run_pipeline_v5.sh <path_to_md_folder> [doi]
#
# Example:
#   ./run_pipeline_v5.sh "./fetched_papers/obelix_md/Effect of Si substitution on the structural and transport properties of superionic Li-argyrodites"
# =============================================================================

set -e  # Exit on error
set -o pipefail  # Ensure pipe errors are caught

# Initialize conda/mamba
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
RESULTS_DIR="./fetched_papers/obelix_parsed_v5"

# =============================================================================
# Helper Functions
# =============================================================================

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}=========================================${NC}"
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

# Function to find DOI from paper title (reused from run_pipeline.sh)
find_doi_from_title() {
    local paper_title="$1"
    local normalized_title
    
    # Normalize the title
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
            
            # Normalize DOI for comparison
            csv_doi = tolower($1)
            gsub(/[^a-z0-9]/, "_", csv_doi)
            gsub(/_+/, "_", csv_doi)
            gsub(/^_|_$/, "", csv_doi)

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
    MD_FOLDER=""
    PROVIDED_DOI=""
    SKIP_EVAL=false

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help|-h)
                echo "Usage: $0 <path_to_md_folder> [doi] [--skip-eval]"
                exit 0
                ;;
            --skip-eval)
                SKIP_EVAL=true
                shift
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

    if [ -z "$MD_FOLDER" ]; then
        print_error "Missing required argument: path to MD folder"
        exit 1
    fi
    
    if [ ! -d "$MD_FOLDER" ]; then
        print_error "MD folder not found: $MD_FOLDER"
        exit 1
    fi
    
    MD_FILE=$(find "$MD_FOLDER" -maxdepth 1 -name "*.md" -type f | head -n 1)
    if [ -z "$MD_FILE" ]; then
        print_error "No .md file found in: $MD_FOLDER"
        exit 1
    fi
    
    PAPER_TITLE=$(basename "$MD_FOLDER")
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    BASENAME=$(basename "$MD_FILE" .md)
    LOG_FILE="$RESULTS_DIR/${BASENAME}_v5_pipeline_${TIMESTAMP}.log"
    
    mkdir -p "$RESULTS_DIR"
    
    log_output() {
        tee -a "$LOG_FILE"
    }
    
    {
        print_header "DATA EXTRACTION PIPELINE v5"
        print_info "Paper: $PAPER_TITLE"
        print_info "MD File: $MD_FILE"
        print_info "Asset Directory: $MD_FOLDER"
        print_info "Log File: $LOG_FILE"
        echo ""
    } | log_output
    
    # =============================================================================
    # STEP 1: Extraction (Direct MD -> JSON)
    # =============================================================================
    {
        print_step "1/2" "Extracting data directly from MD file..."
        echo ""
    } | log_output
    
    # basic_extraction_md_v5.py defaults to robust_results_v5.json in the same folder as the MD
    # We will move it to the results directory after
    EXTRACTED_JSON_V5="$MD_FOLDER/robust_results_v5.json"
    FINAL_EXTRACTED_JSON="$RESULTS_DIR/${BASENAME}_v5_extracted.json"

    if run_python basic_extraction_md_v5.py "$MD_FILE" --asset_dir "$MD_FOLDER" 2>&1 | log_output; then
        if [ -f "$EXTRACTED_JSON_V5" ]; then
            mv "$EXTRACTED_JSON_V5" "$FINAL_EXTRACTED_JSON"
            {
                print_info "✓ Extraction saved to: $FINAL_EXTRACTED_JSON"
                echo ""
            } | log_output
        else
            {
                print_error "Expected output file not found: $EXTRACTED_JSON_V5"
            } | log_output
            exit 1
        fi
    else
        {
            print_error "Failed during extraction step"
        } | log_output
        exit 1
    fi
    
    # =============================================================================
    # STEP 2: Evaluate Extraction
    # =============================================================================
    if [ "$SKIP_EVAL" = true ]; then
        {
            print_warning "Skipping evaluation step as requested."
            echo ""
            print_header "PIPELINE COMPLETED (WITHOUT EVALUATION)"
            print_info "Extracted JSON: $FINAL_EXTRACTED_JSON"
            print_info "Log File: $LOG_FILE"
        } | log_output
        exit 0
    fi

    {
        print_step "2/2" "Evaluating extraction against ground truth..."
    } | log_output
    
    if [ -n "$PROVIDED_DOI" ]; then
        DOI="$PROVIDED_DOI"
        print_info "Using provided DOI: $DOI" | log_output
    else
        print_info "Searching for DOI based on paper title..." | log_output
        DOI=$(find_doi_from_title "$PAPER_TITLE")
        
        if [ -z "$DOI" ]; then
            {
                print_warning "Could not find DOI for paper: $PAPER_TITLE"
                print_warning "Skipping evaluation step."
                echo ""
                print_header "PIPELINE COMPLETED (WITHOUT EVALUATION)"
                print_info "Extracted JSON: $FINAL_EXTRACTED_JSON"
                print_info "Log File: $LOG_FILE"
            } | log_output
            exit 0
        else
            print_info "Found DOI: $DOI" | log_output
        fi
    fi
    
    if [ ! -f "$DATASET_CSV" ]; then
        {
            print_warning "Ground truth dataset not found: $DATASET_CSV"
            print_warning "Skipping evaluation step."
            echo ""
            print_header "PIPELINE COMPLETED (WITHOUT EVALUATION)"
            print_info "Extracted JSON: $FINAL_EXTRACTED_JSON"
            print_info "Log File: $LOG_FILE"
        } | log_output
        exit 0
    fi
    
    # basic_evaluation.py outputs to validation_report.json/txt in the SAME folder as the extracted json
    # Since we moved the extracted json to results_dir, the reports will be there too.
    if run_python basic_evaluation.py --extracted "$FINAL_EXTRACTED_JSON" --ground-truth "$DATASET_CSV" --doi "$DOI" 2>&1 | log_output; then
        {
            print_info "✓ Evaluation complete"
            echo ""
        } | log_output
    else
        {
            print_error "Evaluation failed"
        } | log_output
        exit 1
    fi
    
    {
        print_header "PIPELINE v5 COMPLETED SUCCESSFULLY"
        print_info "Paper: $PAPER_TITLE"
        print_info "DOI: $DOI"
        echo ""
        print_info "Generated Files:"
        print_info "  1. Extracted JSON:    $FINAL_EXTRACTED_JSON"
        print_info "  2. Validation JSON:   $RESULTS_DIR/validation_report_${BASENAME}.json"
        print_info "  3. Validation Text:   $RESULTS_DIR/validation_report_${BASENAME}.txt"
        print_info "  4. Pipeline Log:      $LOG_FILE"
        echo ""
    } | log_output
}

main "$@"
