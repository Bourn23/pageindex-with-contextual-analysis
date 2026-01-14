import pandas as pd
from pathlib import Path

def generate_yield_csv(input_csv, output_csv):
    """
    Generates a CSV of DOIs and their entry counts from OBELiX, 
    sorted from highest to lowest yield.
    """
    path = Path(input_csv)
    if not path.exists():
        print(f"Error: File not found at {input_csv}")
        return

    # Load the dataset
    df = pd.read_csv(path)
    
    # Calculate entry counts per DOI
    yield_df = df.groupby('DOI').size().reset_index(name='entry_count')
    
    # Sort by entry_count descending
    yield_df = yield_df.sort_values(by='entry_count', ascending=False)
    
    # Save to CSV
    yield_df.to_csv(output_csv, index=False)
    print(f"[++] Successfully generated {output_csv}")
    print(f"[*] Total unique DOIs: {len(yield_df)}")
    print("\nTop 10 High-Yield DOIs:")
    print(yield_df.head(10).to_string(index=False))

if __name__ == "__main__":
    generate_yield_csv("OBELiX/data/processed.csv", "obelix_doi_yields.csv")
