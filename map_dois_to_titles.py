import pandas as pd
from habanero import Crossref
from pathlib import Path
import time
from tqdm import tqdm

def fetch_titles(csv_path, output_path):
    """
    Reads a CSV with a DOI column and adds a Title column by fetching from CrossRef.
    """
    df = pd.read_csv(csv_path)
    if 'DOI' not in df.columns:
        print(f"Error: No DOI column in {csv_path}")
        return

    cr = Crossref()
    titles = []
    
    # Check for existing titles to skip (if resuming)
    if 'Title' in df.columns:
        # Fill missing titles
        mask = df['Title'].isna()
    else:
        df['Title'] = None
        mask = pd.Series([True] * len(df))

    print(f"[*] Fetching titles for {mask.sum()} DOIs...")
    
    for idx in tqdm(df.index[mask]):
        doi = df.loc[idx, 'DOI']
        # Handle cases where multiple DOIs are separated by |
        primary_doi = doi.split('|')[0].strip()
        
        try:
            res = cr.works(ids=primary_doi)
            if 'message' in res and 'title' in res['message']:
                title = res['message']['title'][0]
                df.at[idx, 'Title'] = title
            else:
                df.at[idx, 'Title'] = "Unknown Title"
        except Exception as e:
            print(f"\n[!] Error fetching DOI {primary_doi}: {e}")
            df.at[idx, 'Title'] = "Error Fetching"
        
        # Respect ratelimits/be polite
        time.sleep(0.2)
        
        # Save periodically
        if (idx + 1) % 20 == 0:
            df.to_csv(output_path, index=False)

    df.to_csv(output_path, index=False)
    print(f"[++] Saved updated CSV to {output_path}")

if __name__ == "__main__":
    input_csv = "obelix_doi_yields.csv"
    output_csv = "obelix_doi_yields_with_titles.csv"
    fetch_titles(input_csv, output_csv)
