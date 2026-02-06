import os
import pandas as pd
import glob
import re

def normalize_doi(doi):
    """Normalize DOI for mapping (lowercase, strip whitespace)."""
    return str(doi).strip().lower()

def doi_to_slug(doi):
    """Convert DOI to a slug format typically used for folder names."""
    return str(doi).replace('/', '_').replace('.', '_').strip().lower()

def main():
    processed_csv_path = 'OBELiX/data/processed.csv'
    mapping_csv_path = 'obelix_doi_yields_with_titles_normalized.csv'
    output_csv_path = 'obelix_data_with_processing_method.csv'
    search_dirs = [
        'fetched_papers/obelix_md2_already_parsed',
        'fetched_papers/obelix_md2',
        'fetched_papers/obelix_md'
    ]

    print(f"Loading mapping data from {mapping_csv_path}...")
    mapping_df = pd.read_csv(mapping_csv_path)
    
    # Map title -> DOI and DOI_slug -> DOI
    title_to_doi = {}
    slug_to_doi = {}
    for _, row in mapping_df.iterrows():
        doi = normalize_doi(row['DOI'])
        title = str(row['Title']).strip().lower()
        title_to_doi[title] = doi
        slug_to_doi[doi_to_slug(doi)] = doi

    print("Searching for synthesis_method.txt files...")
    doi_to_method = {}
    
    for base_search_dir in search_dirs:
        if not os.path.exists(base_search_dir):
            continue
            
        print(f"Scanning {base_search_dir}...")
        # Find all synthesis_method.txt files
        search_pattern = os.path.join(base_search_dir, "**", "synthesis_method.txt")
        method_files = glob.glob(search_pattern, recursive=True)
        
        for method_file in method_files:
            folder_path = os.path.dirname(method_file)
            folder_name = os.path.basename(folder_path).lower()
            
            # Identify DOI from folder name
            target_doi = None
            if folder_name in title_to_doi:
                target_doi = title_to_doi[folder_name]
            elif folder_name in slug_to_doi:
                target_doi = slug_to_doi[folder_name]
            else:
                # Try partial slug match or direct DOI match if the slug is complex
                # Some folders might be direct DOIs with underscores
                for slug, doi in slug_to_doi.items():
                    if slug in folder_name or folder_name in slug:
                        target_doi = doi
                        break
            
            if target_doi:
                try:
                    with open(method_file, 'r', encoding='utf-8') as f:
                        method_text = f.read().strip()
                        if method_text:
                            # Use list to store multiple if we ever have them, but for now simple overwrite/store
                            doi_to_method[target_doi] = method_text
                            print(f"  Found method for {target_doi} in {folder_name}")
                except Exception as e:
                    print(f"  Error reading {method_file}: {e}")

    print(f"Updating dataset...")
    # Load the base processed.csv
    if not os.path.exists(processed_csv_path):
        print(f"Error: {processed_csv_path} not found.")
        return
        
    base_df = pd.read_csv(processed_csv_path)
    
    # Load existing output if it exists to preserve data
    if os.path.exists(output_csv_path):
        print(f"Merging with existing {output_csv_path}...")
        existing_df = pd.read_csv(output_csv_path)
        # We'll use this to fill in missing values later if needed
        # But honestly, it's safer to just re-match everything from the discovered files
        # and then merge with existing to avoid losing data from directories not scanned this time.
    else:
        existing_df = None

    # Ensure "Synthesis Method" column exists
    if 'Synthesis Method' not in base_df.columns:
        base_df['Synthesis Method'] = ""

    # Map DOI to Synthesis Method in the base dataframe
    def get_method(doi_val):
        if pd.isna(doi_val):
            return ""
        
        # DOI can be multiple separated by |
        dois = [normalize_doi(d) for d in str(doi_val).split('|')]
        for d in dois:
            if d in doi_to_method:
                return doi_to_method[d]
        return ""

    # Update based on current crawl
    base_df['Synthesis Method'] = base_df['DOI'].apply(get_method)

    # If we had existing data, preserve it where current crawl didn't find anything
    if existing_df is not None and 'Synthesis Method' in existing_df.columns:
        # Match by ID or DOI? ID is better if it's unique
        # Let's use ID as the key
        if 'ID' in base_df.columns and 'ID' in existing_df.columns:
            existing_methods = existing_df.set_index('ID')['Synthesis Method'].to_dict()
            for idx, row in base_df.iterrows():
                if not row['Synthesis Method'] and row['ID'] in existing_methods:
                    method_val = existing_methods[row['ID']]
                    if pd.notna(method_val) and method_val != "":
                        base_df.at[idx, 'Synthesis Method'] = method_val

    # Save output
    base_df.to_csv(output_csv_path, index=False)
    print(f"Saved merged dataset to {output_csv_path}")
    print(f"Total records: {len(base_df)}")
    print(f"Records with synthesis methods: {base_df['Synthesis Method'].apply(lambda x: 1 if x != '' and pd.notna(x) else 0).sum()}")

if __name__ == "__main__":
    main()
