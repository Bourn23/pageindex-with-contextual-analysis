import os
import glob
import re
import pandas as pd

def extract_dois_from_logs(log_dir):
    """
    Extract DOIs from all .log files in the specified directory.
    Example line: "[INFO] Found DOI: 10.1016/j.ssi.2014.07.018"
    """
    doi_pattern = re.compile(r"Found DOI:\s*([\w\d\.\/\-]+)", re.IGNORECASE)
    extracted_dois = set()
    
    log_files = glob.glob(os.path.join(log_dir, "*.log"))
    print(f"Found {len(log_files)} log files in {log_dir}")
    
    for log_file in log_files:
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    match = doi_pattern.search(line)
                    if match:
                        doi = match.group(1).strip()
                        extracted_dois.add(doi)
        except Exception as e:
            print(f"Error reading {log_file}: {e}")
            
    return extracted_dois

def main():
    log_directory = "fetched_papers/obelix_parsed_v5_combined"
    csv_path = "OBELiX/data/processed.csv"
    output_path = "obelix_ground_truth_matches.csv"
    
    # 1. Extract DOIs from logs
    print(f"Extracting DOIs from logs in {log_directory}...")
    dois = extract_dois_from_logs(log_directory)
    print(f"Extracted {len(dois)} unique DOIs from logs.")
    
    if not dois:
        print("No DOIs found in log files. Exiting.")
        return

    # 2. Load OBELiX processed data
    print(f"Loading ground truth data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return
        
    print(f"Total rows in OBELiX CSV: {len(df)}")
    
    # 3. Filter for matches
    # Ensure DOIs are stripped and compared case-insensitively if needed
    # (Though DOIs are typically case-insensitive in prefix but sensitive in suffix, 
    # matched strings usually follow the source)
    
    # We'll do a simple set intersection based on the 'DOI' column
    df['DOI'] = df['DOI'].astype(str).str.strip()
    
    matched_df = df[df['DOI'].isin(dois)]
    
    print(f"Found {len(matched_df)} matching rows in ground truth data.")
    
    # 4. Save results
    if not matched_df.empty:
        matched_df.to_csv(output_path, index=False)
        print(f"Successfully saved matches to {output_path}")
        
        # summary of matches vs missing
        unique_matched_dois = set(matched_df['DOI'].unique())
        missing_dois = dois - unique_matched_dois
        if missing_dois:
            print(f"Note: {len(missing_dois)} DOIs from logs were not found in the OBELiX CSV.")
            # Optional: print first 5 missing
            print("Examples of missing DOIs:", list(missing_dois)[:5])
    else:
        print("No matches found in the OBELiX CSV for the extracted DOIs.")

if __name__ == "__main__":
    main()
