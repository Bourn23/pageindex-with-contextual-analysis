import csv
import re
import os

def normalize_name(filename, is_dir=False):
    """
    Normalize a filename or title using the same logic as normalize_filenames.py
    """
    if is_dir:
        name = filename
        ext = ""
    else:
        name, ext = os.path.splitext(filename)
    
    # 1. Lowercase
    name = name.lower()
    
    # 2. Handle chemical formula markup (sub/sup)
    name = name.replace("_sub_", "_").replace("_sup_", "_")
    
    # 3. Replace all whitespace (spaces, tabs, newlines) with underscores
    name = re.sub(r'\s+', '_', name)
    
    # 4. Remove/Replace special characters
    # Keep alphanumeric, underscores, and dashes. Replace everything else (including dots) with underscores.
    name = re.sub(r'[^a-zA-Z0-9\-_]', '_', name)
    
    # 5. Deduplicate underscores
    name = re.sub(r'_+', '_', name)
    
    # 6. Strip leading/trailing underscores
    name = name.strip('_')
    
    if is_dir:
        return name
    else:
        # Standardize extension to lowercase and remove non-alphanumeric if any
        clean_ext = ext.lower().strip()
        return name + clean_ext

def update_csv_titles(input_csv, output_csv):
    """
    Read the CSV file, normalize the Title column, and write to output
    """
    rows = []
    
    # Read the CSV
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        for row in reader:
            # Normalize the title (treating it as a directory name, no extension)
            original_title = row['Title']
            normalized_title = normalize_name(original_title, is_dir=True)
            row['Title'] = normalized_title
            rows.append(row)
    
    # Write the updated CSV
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Updated {len(rows)} titles in {output_csv}")

if __name__ == "__main__":
    input_file = "/Users/bourn23/Downloads/general/PageIndex/obelix_doi_yields_with_titles.csv"
    output_file = "/Users/bourn23/Downloads/general/PageIndex/obelix_doi_yields_with_titles_normalized.csv"
    
    update_csv_titles(input_file, output_file)
    print(f"\nOriginal file: {input_file}")
    print(f"Updated file: {output_file}")
