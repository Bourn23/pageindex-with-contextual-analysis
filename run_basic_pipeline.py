import os
import argparse
import subprocess
import pandas as pd
from pathlib import Path
import re

def normalize_text(text):
    """Normalize text for comparison (similar to run_pipeline.sh)"""
    # Lowercase, replace non-alphanumeric with underscores, collapse underscores, strip
    text = text.lower()
    text = re.sub(r'[^a-z0-9]', '_', text)
    text = re.sub(r'_+', '_', text)
    return text.strip('_')

def find_doi_from_title(folder_name, csv_path):
    """Find DOI from paper title (folder name) using the mapping CSV"""
    if not os.path.exists(csv_path):
        print(f"Warning: CSV mapping file not found at {csv_path}")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        normalized_query = normalize_text(folder_name)
        
        for _, row in df.iterrows():
            # The CSV columns are: DOI, (some index?), Title
            # Let's assume columns are DOI, count, Title based on run_pipeline.sh using $1 and $3
            # In run_pipeline.sh: $1 = DOI, $3 = Title
            
            doi = str(row.iloc[0])
            title = str(row.iloc[2]) if len(row) > 2 else ""
            
            normalized_csv_title = normalize_text(title)
            normalized_csv_doi = normalize_text(doi)
            
            # Match query against title or DOI
            if normalized_query == normalized_csv_title or \
               normalized_csv_title in normalized_query or \
               normalized_query in normalized_csv_title or \
               normalized_query == normalized_csv_doi:
                return doi
                
    except Exception as e:
        print(f"Error searching for DOI: {e}")
    
    return None

def main():
    parser = argparse.ArgumentParser(description='Basic Extraction and Evaluation Pipeline')
    parser.add_argument('md_folder', type=Path, help='Path to the folder containing the markdown file')
    parser.add_argument('--doi', help='Manually provide DOI (overrides automatic lookup)')
    parser.add_argument('--dataset', default='./OBELiX/data/processed.csv', help='Path to ground truth dataset')
    parser.add_argument('--mapping', default='./obelix_doi_yields_with_titles_normalized.csv', help='Path to DOI mapping CSV')
    args = parser.parse_args()

    if not args.md_folder.is_dir():
        print(f"Error: {args.md_folder} is not a directory.")
        return

    # 1. Find the .md file
    md_files = list(args.md_folder.glob("*.md"))
    if not md_files:
        print(f"Error: No .md file found in {args.md_folder}")
        return
    md_file = md_files[0]
    folder_name = args.md_folder.name
    print(f"Processing: {folder_name}")
    print(f"Found MD file: {md_file}")

    # 2. Determine DOI
    doi = args.doi
    if not doi:
        print("Searching for DOI...")
        doi = find_doi_from_title(folder_name, args.mapping)
    
    if not doi:
        print(f"Error: Could not determine DOI for {folder_name}. Please provide it manually with --doi.")
        return
    
    print(f"Using DOI: {doi}")

    # 3. Step 1: Run Extraction
    print("\n--- Step 1: Extraction ---")
    extracted_json = args.md_folder / "basic_extraction_results.json"
    
    # We call basic_extraction_md.py as a subprocess
    # Command: python basic_extraction_md.py <md_file>
    try:
        subprocess.run(["python", "basic_extraction_md.py", str(md_file)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Extraction failed: {e}")
        return

    # 4. Step 2: Run Evaluation
    print("\n--- Step 2: Evaluation ---")
    # Command: python basic_evaluation.py -gt ./OBELiX/data/processed.csv -i "doi" -ex "extracted_points.txt"
    # Wait, basic_evaluation.py takes JSON now? 
    # Let me check basic_evaluation.py args again.
    # It takes --extracted -ex which points to a file. 
    # basic_extraction_md.py saves to basic_extraction_results.json AND extracted_points.txt.
    # Let's use the JSON as it's more structured if the script supports it, 
    # but the user's prompt said "uses the extracted_data.txt in the same markdown folder".
    # Actually basic_extraction_md.py saves to 'extracted_points.txt' in the parent dir.
    
    extracted_txt = args.md_folder / "extracted_points.txt"
    if not extracted_txt.exists():
        print(f"Error: {extracted_txt} not found after extraction.")
        return

    try:
        # Note: basic_evaluation.py uses pandas, we just fixed that.
        # Arguments: -gt, -i (doi), -ex (extracted file)
        subprocess.run([
            "python", "basic_evaluation.py", 
            "-gt", args.dataset, 
            "-i", doi, 
            "-ex", str(extracted_txt)
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed: {e}")
        return

    print("\n--- Pipeline Completed Successfully ---")

if __name__ == "__main__":
    main()
