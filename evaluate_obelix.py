import pandas as pd
import json
from pathlib import Path
import numpy as np
import re

def normalize_formula(formula):
    """Normalize chemical formulas for comparison."""
    if not isinstance(formula, str):
        return ""
    # Remove HTML tags
    formula = re.sub(r'<[^>]+>', '', formula)
    # ### NEW: Keep the dot (.)! It distinguishes Li0.5 from Li5
    # Remove parenthesis and whitespace only
    formula = re.sub(r'[\(\)\s]', '', formula)
    return formula.lower()

def match_points(gt_point, ext_points, used_indices, temp_tolerance=10, cond_tolerance=0.5):
    """
    Find best match for a ground truth point in the extracted points.
    gt_point: Series from processed.csv
    ext_points: list of dicts from materials.json
    """
    gt_cond = gt_point['Ionic conductivity (S cm-1)']
    gt_comp = normalize_formula(gt_point['Composition'])
    gt_temp_c = 25
    
    best_match = None
    min_cond_diff = float('inf')
    
    for ext in ext_points:
        if ext.get('_index') in used_indices:
            continue
        norm_cond = ext.get('_norm_cond')
        norm_temp = ext.get('_norm_temp')
        # handle not specified cases
        if isinstance(norm_temp, (int, float)):
            pass 

        # 2. If it's a string, check for keywords or convert
        elif isinstance(norm_temp, str):
            norm_temp_str = norm_temp.lower()
            # Check for text like "not specified", "room temp", "ambient"
            if any(x in norm_temp_str for x in ['not', 'spec', 'room', 'rt', 'amb']):
                norm_temp = 25.0
            # Check if it's a numeric string (e.g., "300")
            elif norm_temp.replace('.', '', 1).isdigit():
                norm_temp = float(norm_temp)
            else:
                # Fallback for weird strings -> Default to 25 or set to None
                norm_temp = 25.0 

        # 3. If it is None, default to 25.0 (optional, depends on your preference)
        elif norm_temp is None:
             norm_temp = 25.0

        if norm_cond is None:
            continue

        ### Temperature Filter
        # If both have temp, they must be close. 
        # If one is missing, we proceed (permissive match), or you can be strict.
        if gt_temp_c is not None and norm_temp is not None:
            if abs(gt_temp_c - norm_temp) > temp_tolerance:
                continue
            
        # Log distance comparison
        try:
            log_gt = np.log10(float(gt_cond))
            log_ext = np.log10(float(norm_cond))
            cond_diff = abs(log_gt - log_ext)
            
            if cond_diff < cond_tolerance:
                # # Basic check for composition overlap or keyword match
                # ext_name = normalize_formula(ext.get('electrolyte_name', {}).get('full_name', ""))
                # ext_prop = normalize_formula(ext.get('electrolyte_name', {}).get('proportion', ""))
                # ext_desc = normalize_formula(ext.get('material_description', ""))
                
                # ### NEW: Use Canonical Formula if available! -- replaces above
                # It is much cleaner than the raw names.
                ext_canon = normalize_formula(ext.get('canonical_formula', ""))
                ext_name = normalize_formula(ext.get('electrolyte_name', {}).get('full_name', ""))
                ext_prop = normalize_formula(ext.get('electrolyte_name', {}).get('proportion', ""))


                # Check if GT comp parts are found in extraction fields
                # This is a simple heuristic; can be improved
                is_comp_match = (gt_comp in ext_canon or 
                               gt_comp in ext_name or 
                               ext_canon in gt_comp)

                if is_comp_match:
                    if cond_diff < min_cond_diff:
                        min_cond_diff = cond_diff
                        best_match = {
                            "ext_index": ext.get('_index'),
                            "ext_cond": norm_cond,
                            "cond_diff": cond_diff,
                            "ext_comp_raw": ext.get('canonical_formula') or ext_name,
                            "ext_temp": norm_temp
                        }
        except:
            continue
            
    return best_match

def run_evaluation():
    # Paths
    mapping_path = Path("extraction_doi_mapping.json")
    gt_path = Path("OBELiX/data/processed.csv")
    extractions_dir = Path("obelix_md/extractions")
    
    if not mapping_path.exists() or not gt_path.exists():
        print("Error: Missing mapping or ground truth file.")
        return

    with open(mapping_path, 'r') as f:
        mapping = json.load(f)

    gt_df = pd.read_csv(gt_path)
    
    total_gt_points = 0
    total_matched_points = 0
    cond_diffs = []
    
    results = []

    print(f"[*] Starting evaluation for {len(mapping)} papers...")
    
    for entry in mapping:
        doi = entry['DOI']
        ext_file = extractions_dir / entry['ExtractionFile']
        
        if not ext_file.exists():
            print(f"[!] Warning: Extraction file not found: {ext_file}")
            continue
            
        with open(ext_file, 'r') as f:
            ext_data = json.load(f)
        
        # Filter GT points for this DOI
        # Handle cases where DOI is a list in GT (piped |)
        paper_gt = gt_df[gt_df['DOI'].str.contains(re.escape(doi), na=False, case=False)]
        used_indices_for_paper = set()
        
        paper_results = {
            "DOI": doi,
            "GT_Points": len(paper_gt),
            "Matched": 0,
            "Avg_Log_Error": 0
        }
        
        ext_materials = ext_data.get('materials', [])
        
        for _, gt_row in paper_gt.iterrows():
            total_gt_points += 1
            match = match_points(gt_row, ext_materials, used_indices_for_paper)
            
            status = "FOUND" if match else "MISSING"
            res_entry = {
                "DOI": doi,
                "GT_ID": gt_row['ID'],
                "GT_Comp": gt_row['Composition'],
                "GT_Cond": gt_row['Ionic conductivity (S cm-1)'],
                "Status": status,
                "Ext_Cond": match['ext_cond'] if match else None,
                "Ext_Comp_Raw": match['ext_comp_raw'] if match else None,
                "Ext_Temp": match['ext_temp'] if match else None,
                # "Ext_Desc": match['ext_desc'] if match else None,
                "Log_Error": match['cond_diff'] if match else None,
                "Match_Index": match['ext_index'] if match else None
            }
            results.append(res_entry)
            
            if match:
                # Mark this extraction as "taken"
                used_indices_for_paper.add(match['ext_index'])
                total_matched_points += 1
                paper_results["Matched"] += 1
                cond_diffs.append(match['cond_diff'])

        print(f"[+] DOI {doi:25}: Found {paper_results['Matched']}/{paper_results['GT_Points']} entries")

    # Overall Metrics
    recall = total_matched_points / total_gt_points if total_gt_points > 0 else 0
    avg_log_error = np.mean(cond_diffs) if cond_diffs else 0
    
    print("\n" + "="*40)
    print("OVERALL EVALUATION RESULTS")
    print("="*40)
    print(f"Total Ground Truth Points: {total_gt_points}")
    print(f"Total Matched Points:      {total_matched_points}")
    print(f"Recall (Sensitivity):     {recall:.2%}")
    print(f"Avg Log10 Error:          {avg_log_error:.4f} (approx. {10**avg_log_error:.2f}x multiplier)")
    print("="*40)

    # Save detailed results
    res_df = pd.DataFrame(results)
    res_df.to_csv("evaluation_report.csv", index=False)
    print(f"\n[++] Detailed report saved to evaluation_report.csv")

if __name__ == "__main__":
    run_evaluation()
