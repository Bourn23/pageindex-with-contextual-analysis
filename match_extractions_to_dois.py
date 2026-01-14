import pandas as pd
import json
from pathlib import Path
import re

def get_words(text):
    """Extraction of significant words for set comparison."""
    if not isinstance(text, str):
        return set()
    # Normalize minus signs and other common variations
    text = text.replace('−', '-').replace('–', '-')
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove sanitization markers
    text = re.sub(r'_(sub|i|sup|em|strong)_', ' ', text, flags=re.IGNORECASE)
    # Treat underscores as spaces for better word splitting in sanitized filenames
    text = text.replace('_', ' ')
    # Normalize to words
    words = re.findall(r'\b\w+\b', text.lower())
    # Filter out very short words or numbers if they are common (optional)
    return set(words)

def match_files():
    csv_path = "obelix_doi_yields_with_titles.csv"
    extractions_dir = Path("obelix_md/extractions")
    
    if not Path(csv_path).exists():
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    files = list(extractions_dir.glob("*.json"))
    
    mapping = []
    
    print(f"[*] Matching {len(files)} extraction files using word overlap...")
    
    for f in files:
        fname = f.name
        # Filenames usually have _structure_materials.json or _structure.json at the end
        clean_fname = fname.replace("_structure_materials.json", "").replace("_structure.json", "")
        # The extraction pipeline often replaces spaces/special chars with underscores or similar
        # Let's use a robust matching approach
        fname_words = get_words(clean_fname)
        
        best_match = None
        max_overlap = 0
        
        for _, row in df.iterrows():
            doi = row['DOI']
            title = str(row['Title'])
            title_words = get_words(title)
            
            if not title_words or not fname_words:
                continue
                
            intersection = title_words.intersection(fname_words)
            overlap_ratio = len(intersection) / len(title_words)
            
            # Lowered threshold to 0.65 for robustness
            if overlap_ratio > 0.65:
                if overlap_ratio > max_overlap:
                    max_overlap = overlap_ratio
                    best_match = {
                        "DOI": doi,
                        "Title": title,
                        "ExtractionFile": fname,
                        "Overlap": overlap_ratio
                    }
        
        if best_match:
            mapping.append(best_match)
            print(f"[+] Matched: {fname} -> {best_match['DOI']} (Overlap: {best_match['Overlap']:.2f})")
        else:
            print(f"[-] Could not match: {fname}")
            # print(f"    Words: {fname_words}")

    output_path = "extraction_doi_mapping.json"
    with open(output_path, "w") as jf:
        json.dump(mapping, jf, indent=2)
    
    print(f"\n[++] Mapping saved to {output_path}")
    print(f"[*] Total matches: {len(mapping)} / {len(files)}")

if __name__ == "__main__":
    match_files()
