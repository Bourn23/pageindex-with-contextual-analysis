import csv
import subprocess
import sys
import os

# --- Configuration ---
INPUT_CSV = 'obelix_doi_yields.csv'
FETCH_SCRIPT = 'fetch_paper.py'
EMAIL = "brfarnood@gmail.com"

def main():
    # Check if files exist
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Error: CSV file '{INPUT_CSV}' not found.")
        return
    if not os.path.exists(FETCH_SCRIPT):
        print(f"❌ Error: Python script '{FETCH_SCRIPT}' not found.")
        return

    print(f"📂 Reading from: {INPUT_CSV}")
    
    successful_runs = 0
    failed_runs = 0

    # Open the CSV file
    # encoding='utf-8-sig' handles potential BOM characters created by Excel
    with open(INPUT_CSV, mode='r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)

        # Validate Column Name
        if 'DOI' not in reader.fieldnames:
            print(f"❌ Error: Column 'DOI' not found. Found columns: {reader.fieldnames}")
            return

        # Iterate through rows
        for i, row in enumerate(reader):
            doi = row['DOI'].strip()

            if not doi:
                continue # Skip empty rows

            print(f"Processing #{i+1}: {doi}")

            # Construct the command
            # Using sys.executable ensures we use the same Python env running this script
            cmd = [sys.executable, FETCH_SCRIPT, doi, "--email", EMAIL]

            try:
                # Run the command and wait for it to finish
                subprocess.run(cmd, check=True)
                successful_runs += 1
            except subprocess.CalledProcessError as e:
                print(f"⚠️ Failed to fetch {doi}.")
                failed_runs += 1
            except KeyboardInterrupt:
                print("\n🛑 Process stopped by user.")
                break

    print("-" * 30)
    print(f"Done! Papers Downloaded: {successful_runs}, Download Failures: {failed_runs}")

if __name__ == "__main__":
    main()