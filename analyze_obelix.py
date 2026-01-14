import pandas as pd
from pathlib import Path

def analyze_obelix(csv_path: str):
    """
    Analyzes the OBELiX processed.csv to determine the average number 
    of material entries per paper (DOI).
    """
    path = Path(csv_path)
    if not path.exists():
        print(f"Error: File not found at {csv_path}")
        return

    # Load the dataset
    df = pd.read_csv(path)
    
    # Basic counts
    total_entries = len(df)
    unique_dois = df['DOI'].nunique()
    
    if unique_dois == 0:
        print("No DOIs found in the dataset.")
        return

    avg_entries_per_paper = total_entries / unique_dois
    
    # Distribution analysis
    entries_per_doi = df.groupby('DOI').size().sort_values(ascending=False)
    max_entries = entries_per_doi.max()
    min_entries = entries_per_doi.min()
    median_entries = entries_per_doi.median()

    print("=" * 40)
    print("OBELiX DATASET ANALYSIS")
    print("=" * 40)
    print(f"Total entries:          {total_entries}")
    print(f"Unique papers (DOIs):   {unique_dois}")
    print(f"Average entries/paper:  {avg_entries_per_paper:.2f}")
    print(f"Median entries/paper:   {median_entries}")
    print(f"Max entries in one paper: {max_entries}")
    print(f"Min entries in one paper: {min_entries}")
    print("-" * 40)
    print("\nTop 5 papers by number of entries:")
    print(entries_per_doi.head(5))

if __name__ == "__main__":
    csv_file = "OBELiX/data/processed.csv"
    analyze_obelix(csv_file)
