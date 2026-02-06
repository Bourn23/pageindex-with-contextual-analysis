import os
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def standardize_conductivity(value: float, unit_str: str):
    """Standardizes ionic conductivity to S/cm based on unit string."""
    if value is None:
        return None
    
    if not unit_str:
        try: return float(value)
        except: return None

    try:
        val = float(value)
    except (ValueError, TypeError):
        return None

    u_clean = str(unit_str).lower().replace(" ", "").replace("·", "").replace(".", "")
    
    multiplier = 1.0
    
    # 1. Metric Prefix
    if "ms" in u_clean:
        multiplier = 1e-3
    elif "us" in u_clean or "μs" in u_clean or "µs" in u_clean:
        multiplier = 1e-6
    elif "ns" in u_clean:
        multiplier = 1e-9
    elif "ks" in u_clean:
        multiplier = 1000.0
    
    # 2. Geometry (cm vs m)
    if "m" in u_clean and "cm" not in u_clean and "mm" not in u_clean:
         if "m-1" in u_clean or "/m" in u_clean or "sm-1" in u_clean:
             multiplier *= 0.01

    return val * multiplier

def normalize_formula(formula):
    """Normalize chemical formulas for comparison."""
    if not isinstance(formula, str):
        return ""
    formula = re.sub(r'<[^>]+>', '', formula)
    formula = re.sub(r'[\\(\\)\\s]', '', formula)
    return formula.lower()

def get_elements(formula):
    """Extract elements from a formula."""
    return set(re.findall(r'[a-z]+', formula))

def get_normalized_temperature(ext: dict) -> float:
    """Extracts and normalizes temperature to Celsius."""
    norm_temp = ext.get('normalized_temperature_c')
    if norm_temp is not None:
        try: return float(norm_temp)
        except: pass
        
    raw_temp = ext.get('raw_temperature')
    if isinstance(raw_temp, (int, float)):
        return float(raw_temp)
    elif isinstance(raw_temp, str):
        if any(x in raw_temp.lower() for x in ['not', 'spec', 'room', 'rt', 'amb']):
            return 25.0
        else:
            try:
                nums = re.findall(r"[-+]?\\d*\\.\\d+|\\d+", raw_temp)
                return float(nums[0]) if nums else 25.0
            except:
                return 25.0
    return 25.0

def match_points(gt_point, ext_points, used_indices, temp_tolerance=10):
    """Find best match for a ground truth point in the extracted points."""
    gt_cond = gt_point['Ionic conductivity (S cm-1)']
    gt_comp = normalize_formula(gt_point['Composition'])
    gt_temp_c = 25  # Default RT
    
    best_match = None
    min_cond_diff = float('inf')
    
    for idx, ext in enumerate(ext_points):
        if idx in used_indices:
            continue
            
        norm_cond = ext.get('normalized_conductivity')
        if norm_cond is None:
            raw_cond = ext.get('raw_conductivity')
            unit = ext.get('raw_conductivity_unit', 'S/cm')
            norm_cond = standardize_conductivity(raw_cond, unit)
            
        norm_temp = get_normalized_temperature(ext)

        if norm_cond is None:
            continue

        if gt_temp_c is not None and norm_temp is not None:
            if abs(gt_temp_c - norm_temp) > temp_tolerance:
                continue
            
        try:
            log_gt = np.log10(float(gt_cond))
            log_ext = np.log10(float(norm_cond))
            cond_diff = abs(log_gt - log_ext)
            
            IF_MATCH_TOL = 2.0
            
            if cond_diff < IF_MATCH_TOL:
                ext_canon = normalize_formula(ext.get('canonical_formula', ""))
                ext_raw = normalize_formula(ext.get('raw_composition', ""))
                
                is_comp_match = (gt_comp in ext_canon or 
                               gt_comp in ext_raw or 
                               ext_canon in gt_comp or
                               ext_raw in gt_comp)
                
                if not is_comp_match:
                    gt_elems = get_elements(gt_comp)
                    ext_elems = get_elements(ext_canon) or get_elements(ext_raw)
                    if gt_elems == ext_elems and len(gt_elems) > 1:
                        is_comp_match = True

                if is_comp_match:
                    if cond_diff < min_cond_diff:
                        min_cond_diff = cond_diff
                        best_match = {
                            "index": idx,
                            "cond_diff": cond_diff,
                            "ext_canon": ext.get('canonical_formula'),
                            "ext_raw": ext.get('raw_composition'),
                            "ext_cond": norm_cond,
                            "ext_temp": norm_temp,
                            "ext_full": ext  # Store full extraction for analysis
                        }
        except:
            continue
            
    return best_match

def analyze_outliers(config_name="Combined (Table, Text, Vision)", threshold=0.5):
    """Analyze outliers for a specific configuration."""
    
    base_dir = "/Users/bourn23/Downloads/general/PageIndex/fetched_papers"
    folder_map = {
        "Combined (Table, Text, Vision)": "obelix_parsed_v5_combined_table_text_vision",
        "Text Only (Flash)": "obelix_parsed_v5_text_only",
        "Vision Only (Flash)": "obelix_parsed_v5_vision_only_gemini_3_flash",
        "Vision Only (Scaffolding)": "obelix_parsed_v5_vision_only_scaffolding"
    }
    
    folder = folder_map.get(config_name)
    if not folder:
        print(f"Unknown config: {config_name}")
        return
    
    folder_path = os.path.join(base_dir, folder)
    
    # Load ground truth and mapping
    gt_df = pd.read_csv('obelix_ground_truth_matches.csv')
    mapping_df = pd.read_csv('obelix_doi_yields_with_titles_normalized.csv')
    slug_to_doi = dict(zip(mapping_df['Title'], mapping_df['DOI']))
    
    outliers = []
    within_threshold = []
    
    # Process each JSON file
    json_files = list(Path(folder_path).glob("*_extracted.json"))
    
    for json_path in json_files:
        # Extract paper slug from filename
        slug = json_path.stem.replace("_v5_extracted", "")
        doi = slug_to_doi.get(slug)
        
        if not doi:
            continue
        
        with open(json_path, 'r') as f:
            ext_data = json.load(f)
            ext_list = ext_data.get('measurements', [])
        
        paper_gt = gt_df[gt_df['DOI'].str.contains(re.escape(doi), na=False, case=False)]
        
        if paper_gt.empty:
            continue
        
        used_indices = set()
        
        for _, gt_row in paper_gt.iterrows():
            match = match_points(gt_row, ext_list, used_indices)
            if match:
                used_indices.add(match['index'])
                
                gt_cond = gt_row['Ionic conductivity (S cm-1)']
                ext_cond = match['ext_cond']
                
                try:
                    log_gt = np.log10(float(gt_cond))
                    log_ext = np.log10(float(ext_cond))
                    error = log_ext - log_gt
                    abs_error = abs(error)
                    
                    data_point = {
                        'paper': slug,
                        'doi': doi,
                        'gt_composition': gt_row['Composition'],
                        'ext_composition_canon': match['ext_canon'],
                        'ext_composition_raw': match['ext_raw'],
                        'gt_conductivity': gt_cond,
                        'ext_conductivity': ext_cond,
                        'log_gt': log_gt,
                        'log_ext': log_ext,
                        'error': error,
                        'abs_error': abs_error,
                        'ext_source': match['ext_full'].get('source', 'unknown'),
                        'ext_figure': match['ext_full'].get('figure_id', 'unknown'),
                        'ext_raw_temp': match['ext_full'].get('raw_temperature', 'unknown'),
                        'ext_norm_temp': match['ext_temp'],
                        'ext_raw_cond_unit': match['ext_full'].get('raw_conductivity_unit', 'unknown'),
                        'ext_raw_cond_value': match['ext_full'].get('raw_conductivity', 'unknown'),
                    }
                    
                    if abs_error > threshold:
                        outliers.append(data_point)
                    else:
                        within_threshold.append(data_point)
                        
                except Exception as e:
                    print(f"Error processing match: {e}")
                    continue
    
    outliers_df = pd.DataFrame(outliers)
    within_df = pd.DataFrame(within_threshold)
    
    print(f"\n{'='*80}")
    print(f"OUTLIER ANALYSIS: {config_name}")
    print(f"Threshold: ±{threshold} log units")
    print(f"{'='*80}\n")
    
    print(f"Total matched points: {len(outliers) + len(within_threshold)}")
    print(f"Points within threshold: {len(within_threshold)} ({len(within_threshold)/(len(outliers)+len(within_threshold))*100:.1f}%)")
    print(f"Outliers (>±{threshold}): {len(outliers)} ({len(outliers)/(len(outliers)+len(within_threshold))*100:.1f}%)")
    
    if len(outliers) > 0:
        # Save outliers to CSV
        outliers_csv = f"outliers_{config_name.replace(' ', '_').replace('(', '').replace(')', '').lower()}.csv"
        outliers_df.to_csv(outliers_csv, index=False)
        print(f"\n✓ Outliers saved to: {outliers_csv}")
        
        # Analyze by source
        print(f"\n--- Outliers by Source ---")
        source_counts = outliers_df['ext_source'].value_counts()
        print(source_counts)
        
        # Analyze by paper
        print(f"\n--- Outliers by Paper ---")
        paper_counts = outliers_df['paper'].value_counts()
        print(paper_counts)
        
        # Show worst outliers
        print(f"\n--- Top 10 Worst Outliers (by absolute error) ---")
        worst = outliers_df.nlargest(10, 'abs_error')[['paper', 'gt_composition', 'ext_composition_canon', 
                                                         'gt_conductivity', 'ext_conductivity', 'abs_error', 
                                                         'ext_source', 'ext_figure']]
        print(worst.to_string(index=False))
        
        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Error distribution
        ax1 = axes[0, 0]
        ax1.hist(outliers_df['abs_error'], bins=20, edgecolor='black', alpha=0.7, color='red', label='Outliers')
        ax1.axvline(x=threshold, color='orange', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
        ax1.set_xlabel('Absolute Error (log₁₀ units)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title(f'Distribution of Outlier Errors\n{config_name}', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Outliers by source
        ax2 = axes[0, 1]
        source_counts.plot(kind='bar', ax=ax2, color='steelblue', edgecolor='black')
        ax2.set_xlabel('Extraction Source', fontsize=12)
        ax2.set_ylabel('Number of Outliers', fontsize=12)
        ax2.set_title('Outliers by Extraction Source', fontsize=13, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Outliers by paper
        ax3 = axes[1, 0]
        paper_counts.plot(kind='barh', ax=ax3, color='coral', edgecolor='black')
        ax3.set_xlabel('Number of Outliers', fontsize=12)
        ax3.set_ylabel('Paper', fontsize=10)
        ax3.set_title('Outliers by Paper', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')
        
        # 4. GT vs Extracted for outliers only
        ax4 = axes[1, 1]
        ax4.scatter(outliers_df['log_gt'], outliers_df['log_ext'], 
                   c=outliers_df['abs_error'], cmap='Reds', s=100, 
                   edgecolors='black', linewidth=0.5, alpha=0.7)
        
        min_val = min(outliers_df['log_gt'].min(), outliers_df['log_ext'].min())
        max_val = max(outliers_df['log_gt'].max(), outliers_df['log_ext'].max())
        ax4.plot([min_val, max_val], [min_val, max_val], 'b--', linewidth=2, label='Perfect Match')
        ax4.fill_between([min_val, max_val], 
                        [min_val - threshold, max_val - threshold],
                        [min_val + threshold, max_val + threshold],
                        alpha=0.2, color='green', label=f'±{threshold} threshold')
        
        ax4.set_xlabel('Ground Truth log₁₀(σ / S cm⁻¹)', fontsize=12)
        ax4.set_ylabel('Extracted log₁₀(σ / S cm⁻¹)', fontsize=12)
        ax4.set_title('Outliers: GT vs Extracted', fontsize=13, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        cbar = plt.colorbar(ax4.collections[0], ax=ax4)
        cbar.set_label('Absolute Error', fontsize=10)
        
        plt.tight_layout()
        plot_path = f"outlier_analysis_{config_name.replace(' ', '_').replace('(', '').replace(')', '').lower()}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Outlier analysis plots saved to: {plot_path}")
        
        # Detailed comparison: outliers vs within-threshold
        if len(within_threshold) > 0:
            print(f"\n--- Source Distribution Comparison ---")
            print(f"\nOutliers by source:")
            outlier_source_pct = (outliers_df['ext_source'].value_counts() / len(outliers_df) * 100).round(1)
            print(outlier_source_pct)
            
            print(f"\nWithin-threshold by source:")
            within_source_pct = (within_df['ext_source'].value_counts() / len(within_df) * 100).round(1)
            print(within_source_pct)
    
    return outliers_df, within_df

if __name__ == "__main__":
    import sys
    
    config = "Combined (Table, Text, Vision)"
    threshold = 0.5
    
    if len(sys.argv) > 1:
        config = sys.argv[1]
    if len(sys.argv) > 2:
        threshold = float(sys.argv[2])
    
    outliers_df, within_df = analyze_outliers(config, threshold)
