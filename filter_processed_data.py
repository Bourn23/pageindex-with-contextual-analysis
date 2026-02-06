import pandas as pd
import os

def main():
    input_csv = 'obelix_data_with_processing_method.csv'
    output_csv = 'obelix_data_with_processing_method_filtered.csv'
    
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found.")
        return
        
    print(f"Loading {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Filter rows where Synthesis Method is not empty
    # Handle both NaN and empty strings
    filtered_df = df[df['Synthesis Method'].notna() & (df['Synthesis Method'].str.strip() != "")]
    
    print(f"Filtered {len(df)} records down to {len(filtered_df)} records with synthesis methods.")
    
    filtered_df.to_csv(output_csv, index=False)
    print(f"Saved filtered dataset to {output_csv}")

if __name__ == "__main__":
    main()
