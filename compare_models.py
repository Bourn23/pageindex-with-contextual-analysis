import os
import re
import json
import math
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional

def standardize_conductivity(value: float, unit_str: str) -> Optional[float]:
    """
    Standardizes ionic conductivity to S/cm based on unit string.
    Returns None if value is missing or invalid.
    """
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
    if "ms" in u_clean:          # Milli (10^-3)
        multiplier = 1e-3
    elif "us" in u_clean or "μs" in u_clean or "µs" in u_clean: # Micro (10^-6)
        multiplier = 1e-6
    elif "ns" in u_clean:        # Nano (10^-9)
        multiplier = 1e-9
    elif "ks" in u_clean:        # Kilo (10^3)
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
    # Remove HTML tags
    formula = re.sub(r'<[^>]+>', '', formula)
    # Remove parenthesis and whitespace
    formula = re.sub(r'[\(\)\s]', '', formula)
    return formula.lower()

def get_elements(formula):
    """Extract elements from a formula."""
    return set(re.findall(r'[a-z]+', formula))

def get_normalized_temperature(ext: dict) -> float:
    """
    Extracts and normalizes temperature to Celsius from extracted measurement.
    Defaults to 25.0 if not specified.
    """
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
                nums = re.findall(r"[-+]?\d*\.\d+|\d+", raw_temp)
                return float(nums[0]) if nums else 25.0
            except:
                return 25.0
    return 25.0

def match_points(gt_point, ext_points, used_indices, temp_tolerance=10, cond_tolerance=0.5):
    """
    Find best match for a ground truth point in the extracted points.
    """
    gt_cond = gt_point['Ionic conductivity (S cm-1)']
    gt_comp = normalize_formula(gt_point['Composition'])
    gt_temp_c = 25 # Default RT
    
    best_match = None
    min_cond_diff = float('inf')
    
    for idx, ext in enumerate(ext_points):
        if idx in used_indices:
            continue
            
        # Use normalized values if available
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
            
            # Use broad tolerance for matching, then we filter for accuracy later
            # But the matching itself should be somewhat constrained
            IF_MATCH_TOL = 2.0 # Broader tolerance for initial link
            
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
                    if gt_elems == ext_elems and len(gt_elems) > 1: # Require at least 2 elements to avoid trivial matches
                        is_comp_match = True

                if is_comp_match:
                    if cond_diff < min_cond_diff:
                        min_cond_diff = cond_diff
                        best_match = {
                            "index": idx,
                            "cond_diff": cond_diff,
                            "ext_canon": ext.get('canonical_formula'),
                            "ext_cond": norm_cond,
                            "ext_temp": norm_temp
                        }
        except:
            continue
            
    return best_match

def parse_pipeline_log(log_path):
    """
    Parses a pipeline log file to extract metrics.
    """
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Extract Paper Name (Slug) - More robust regex
    paper_match = re.search(r"\[INFO\] Paper:\s*([^\r\n]+)", content)
    paper_name = paper_match.group(1).strip() if paper_match else os.path.basename(log_path).replace(".log", "").split("_v5_pipeline_")[0]

    # Extract GT and EX counts
    counts_match = re.search(r"Comparing (\d+) GT items vs (\d+) Extracted items...", content)
    total_gt = int(counts_match.group(1)) if counts_match else 0
    extracted_match = re.search(r"Done! Extracted (\d+) points.", content)
    total_ex = int(extracted_match.group(1)) if extracted_match else 0

    # Extract Validation Score (Recall)
    score_match = re.search(r"VALIDATION SCORE: ([\d.]+)%", content)
    recall = float(score_match.group(1)) if score_match else 0.0

    matches = round((recall / 100.0) * total_gt)
    precision = (matches / total_ex) * 100.0 if total_ex > 0 else 0.0

    return {
        "paper": paper_name,
        "total_gt": total_gt,
        "total_ex": total_ex,
        "matches": matches,
        "recall": recall,
        "precision": precision
    }

def main():
    parser = argparse.ArgumentParser(description='Compare model performance with point-to-point accuracy metrics.')
    parser.add_argument('--threshold', type=float, default=0.5, help='Log10 threshold for accuracy (default: 0.5)')
    parser.add_argument('--gt', default='obelix_ground_truth_matches.csv', help='Path to Ground Truth CSV')
    parser.add_argument('--mapping', default='obelix_doi_yields_with_titles_normalized.csv', help='Path to Title-DOI mapping CSV')
    args = parser.parse_args()

    base_dir = "/Users/bourn23/Downloads/general/PageIndex/fetched_papers"
    folders = {
        "Text Only (Flash)": "obelix_parsed_v5_text_only",
        "Vision Only (Flash)": "obelix_parsed_v5_vision_only_gemini_3_flash",
        # "Vision Only (Pro)": "obelix_parsed_v5_vision_only_gemini_3_pro",
        # "Combined (Flash)": "obelix_parsed_v5_combined",
        # "Combined (Flash) + V2": "obelix_parsed_v5_combined_md_v2",
        # "Vision Only (Scaffolding)": "obelix_parsed_v5_vision_only_scaffolding",
        "Combined (Table, Text, Vision)": "obelix_parsed_v5_combined_table_text_vision",
        "Combined v4": "obelix_parsed_v5_v4"
    }

    # Load mappings
    try:
        mapping_df = pd.read_csv(args.mapping)
        slug_to_doi = dict(zip(mapping_df['Title'], mapping_df['DOI']))
    except Exception as e:
        print(f"Warning: Could not load mapping from {args.mapping}: {e}")
        slug_to_doi = {}

    try:
        gt_df = pd.read_csv(args.gt)
    except Exception as e:
        print(f"Error: Could not load GT from {args.gt}: {e}")
        return

    all_results = []
    conductivity_plot_data = []
    point_to_point_data = []  # For scatter and residual plots
    per_paper_results = []  # For per-paper breakdown

    # Collect all unique GT values for the papers we are actually looking at
    # To avoid bias from papers not processed by a specific model, 
    # we'll collect GT data within the model loop but keep unique entries.
    gt_values_collected = set() # (DOI, value) to handle duplicates if any

    for label, folder in folders.items():
        folder_path = os.path.join(base_dir, folder)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} does not exist.")
            continue

        log_files = list(Path(folder_path).glob("*.log"))
        print(f"Processing {label}: Found {len(log_files)} logs.")

        for log in log_files:
            try:
                metrics = parse_pipeline_log(log)
                
                # Accuracy Calculation logic
                accuracy = 0.0
                slug = metrics["paper"]
                doi = slug_to_doi.get(slug)
                
                # Match JSON filename to log
                # Log: nameofpaper_v5_pipeline_TIMESTAMP.log
                # Extracted: nameofpaper_v5_extracted.json
                json_filename = re.sub(r"_pipeline_.*\.log$", "_extracted.json", log.name)
                json_path = log.parent / json_filename
                
                if not json_path.exists():
                    # Fallback pattern check
                    json_path = log.parent / log.name.replace("_pipeline_", "_extracted_").replace(".log", ".json")

                if doi and json_path.exists():
                    with open(json_path, 'r') as f:
                        ext_data = json.load(f)
                        ext_list = ext_data.get('measurements', [])
                    
                    paper_gt = gt_df[gt_df['DOI'].str.contains(re.escape(doi), na=False, case=False)]
                    
                    # Store all GT conductivities for this paper
                    if not paper_gt.empty:
                        for _, row in paper_gt.iterrows():
                            val = row['Ionic conductivity (S cm-1)']
                            try:
                                log_val = np.log10(float(val))
                                if not np.isinf(log_val) and not np.isnan(log_val):
                                    gt_values_collected.add((doi, log_val))
                            except: continue

                    # Store all Extracted conductivities for this paper (Filtered for RT: 20-30C)
                    for ext in ext_list:
                        norm_cond = ext.get('normalized_conductivity')
                        if norm_cond is None:
                            raw_cond = ext.get('raw_conductivity')
                            unit = ext.get('raw_conductivity_unit', 'S/cm')
                            norm_cond = standardize_conductivity(raw_cond, unit)
                        
                        if norm_cond:
                            norm_temp = get_normalized_temperature(ext)
                            # Only include room temperature points (20-30 C)
                            if 20 <= norm_temp <= 30:
                                try:
                                    log_val = np.log10(float(norm_cond))
                                    if not np.isinf(log_val) and not np.isnan(log_val):
                                        conductivity_plot_data.append({
                                            "log_sigma": log_val,
                                            "Source": label
                                        })
                                except: continue

                    if not paper_gt.empty:
                        used_indices = set()
                        accurate_count = 0
                        matched_count = 0
                        paper_errors = []
                        
                        for _, gt_row in paper_gt.iterrows():
                            match = match_points(gt_row, ext_list, used_indices)
                            if match:
                                used_indices.add(match['index'])
                                matched_count += 1
                                
                                gt_cond = gt_row['Ionic conductivity (S cm-1)']
                                ext_cond = match['ext_cond']
                                
                                try:
                                    log_gt = np.log10(float(gt_cond))
                                    log_ext = np.log10(float(ext_cond))
                                    error = log_ext - log_gt
                                    
                                    # Store point-to-point data
                                    point_to_point_data.append({
                                        "config": label,
                                        "paper": slug,
                                        "log_gt": log_gt,
                                        "log_ext": log_ext,
                                        "error": error,
                                        "abs_error": abs(error)
                                    })
                                    
                                    paper_errors.append(abs(error))
                                    
                                    if match['cond_diff'] <= args.threshold:
                                        accurate_count += 1
                                except:
                                    continue
                        
                        accuracy = (accurate_count / matched_count * 100.0) if matched_count > 0 else 0.0
                        
                        # Store per-paper results
                        if paper_errors:
                            per_paper_results.append({
                                "config": label,
                                "paper": slug,
                                "matched_points": matched_count,
                                "accurate_points": accurate_count,
                                "mae": np.mean(paper_errors),
                                "rmse": np.sqrt(np.mean([e**2 for e in paper_errors])),
                                "max_error": max(paper_errors)
                            })
                
                metrics["accuracy"] = accuracy
                metrics["config"] = label
                all_results.append(metrics)
            except Exception as e:
                print(f"Error parsing {log}: {e}")

    if not all_results:
        print("No results found.")
        return

    df = pd.DataFrame(all_results)
    
    # Save to CSV
    csv_path = "model_comparison_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    # Aggregate by config
    summary = df.groupby("config").agg({
        "recall": "mean",
        "precision": "mean",
        "accuracy": "mean",
        "total_gt": "sum",
        "matches": "sum"
    }).reset_index()

    print(f"\n{'='*80}")
    print(f"SUMMARY: Macro Averages (Threshold={args.threshold})")
    print(f"{'='*80}")
    print(summary)
    print(f"{'='*80}\n")

    # ========================================================================
    # PRIMARY ANALYSIS: Point-to-Point Comparison
    # ========================================================================
    
    if point_to_point_data:
        p2p_df = pd.DataFrame(point_to_point_data)
        
        print(f"\n{'='*80}")
        print("PRIMARY ANALYSIS: Point-to-Point Error Metrics")
        print(f"{'='*80}")
        
        # Calculate error statistics by config
        error_stats = p2p_df.groupby('config').agg({
            'abs_error': ['mean', 'std', 'median'],
            'error': lambda x: np.sqrt(np.mean(x**2))  # RMSE
        }).round(4)
        error_stats.columns = ['MAE', 'Std Dev', 'Median AE', 'RMSE']
        print(error_stats)
        
        # Tolerance analysis
        print(f"\n--- Percentage of Points Within Tolerance (log10 units) ---")
        tolerance_levels = [0.3, 0.5, 1.0, 1.5, 2.0]
        tolerance_results = []
        for config in p2p_df['config'].unique():
            config_data = p2p_df[p2p_df['config'] == config]
            row = {'Config': config}
            for tol in tolerance_levels:
                pct = (config_data['abs_error'] <= tol).sum() / len(config_data) * 100
                row[f'±{tol}'] = f"{pct:.1f}%"
            tolerance_results.append(row)
        tolerance_df = pd.DataFrame(tolerance_results)
        print(tolerance_df.to_string(index=False))
        print(f"{'='*80}\n")
        
        # Scatter Plot: GT vs Extracted
        fig, axes = plt.subplots(1, len(folders), figsize=(8*len(folders), 7), squeeze=False)
        axes = axes.flatten()
        
        for idx, config in enumerate(sorted(p2p_df['config'].unique())):
            ax = axes[idx]
            config_data = p2p_df[p2p_df['config'] == config]
            
            ax.scatter(config_data['log_gt'], config_data['log_ext'], alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
            
            # Add y=x reference line
            min_val = min(config_data['log_gt'].min(), config_data['log_ext'].min())
            max_val = max(config_data['log_gt'].max(), config_data['log_ext'].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Match (y=x)')
            
            # Add tolerance bands
            ax.fill_between([min_val, max_val], 
                           [min_val - args.threshold, max_val - args.threshold],
                           [min_val + args.threshold, max_val + args.threshold],
                           alpha=0.2, color='green', label=f'±{args.threshold} log units')
            
            ax.set_xlabel('Ground Truth log₁₀(σ / S cm⁻¹)', fontsize=12)
            ax.set_ylabel('Extracted log₁₀(σ / S cm⁻¹)', fontsize=12)
            ax.set_title(f'{config}\n(n={len(config_data)} points)', fontsize=13, fontweight='bold')
            ax.legend(loc='upper left', fontsize=9)
            ax.grid(True, alpha=0.3, linestyle=':')
            ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        scatter_path = "point_to_point_scatter.png"
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        print(f"✓ Scatter plot (GT vs Extracted) saved to {scatter_path}")
        
        # Residual Plot
        fig, axes = plt.subplots(1, len(folders), figsize=(8*len(folders), 6), squeeze=False)
        axes = axes.flatten()
        
        for idx, config in enumerate(sorted(p2p_df['config'].unique())):
            ax = axes[idx]
            config_data = p2p_df[p2p_df['config'] == config]
            
            ax.scatter(config_data['log_gt'], config_data['error'], alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
            ax.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
            ax.axhline(y=args.threshold, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label=f'+{args.threshold} threshold')
            ax.axhline(y=-args.threshold, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label=f'-{args.threshold} threshold')
            
            ax.set_xlabel('Ground Truth log₁₀(σ / S cm⁻¹)', fontsize=12)
            ax.set_ylabel('Residual (Extracted - GT)', fontsize=12)
            ax.set_title(f'{config}\nMAE={config_data["abs_error"].mean():.3f}', fontsize=13, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3, linestyle=':')
        
        plt.tight_layout()
        residual_path = "point_to_point_residuals.png"
        plt.savefig(residual_path, dpi=300, bbox_inches='tight')
        print(f"✓ Residual plot saved to {residual_path}")
    
    # Per-paper breakdown
    if per_paper_results:
        pp_df = pd.DataFrame(per_paper_results)
        pp_csv_path = "per_paper_performance.csv"
        pp_df.to_csv(pp_csv_path, index=False)
        print(f"✓ Per-paper performance breakdown saved to {pp_csv_path}")
        
        # Visualize per-paper MAE
        fig, ax = plt.subplots(figsize=(14, max(6, len(pp_df) * 0.3)))
        
        # Sort by config and MAE for better visualization
        pp_df_sorted = pp_df.sort_values(['config', 'mae'])
        
        configs = pp_df_sorted['config'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(configs)))
        color_map = {config: colors[i] for i, config in enumerate(configs)}
        
        y_positions = range(len(pp_df_sorted))
        bars = ax.barh(y_positions, pp_df_sorted['mae'], 
                       color=[color_map[c] for c in pp_df_sorted['config']],
                       edgecolor='black', linewidth=0.5)
        
        ax.set_yticks(y_positions)
        ax.set_yticklabels([f"{row['paper'][:40]}... ({row['config'][:15]})" 
                           for _, row in pp_df_sorted.iterrows()], fontsize=8)
        ax.set_xlabel('Mean Absolute Error (log₁₀ units)', fontsize=12)
        ax.set_title('Per-Paper MAE by Model Configuration', fontsize=14, fontweight='bold')
        ax.axvline(x=args.threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({args.threshold})')
        ax.legend()
        ax.grid(True, axis='x', alpha=0.3, linestyle=':')
        
        plt.tight_layout()
        per_paper_path = "per_paper_mae.png"
        plt.savefig(per_paper_path, dpi=300, bbox_inches='tight')
        print(f"✓ Per-paper MAE plot saved to {per_paper_path}")
    
    # Standard bar plots for recall/precision/accuracy
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))

    sns.barplot(data=df, x="config", y="recall", ax=ax1, palette="viridis", errorbar="sd")
    ax1.set_title("Average Recall by Model Configuration", fontsize=14)
    ax1.set_ylabel("Recall (%)")
    ax1.set_ylim(0, 105)

    sns.barplot(data=df, x="config", y="precision", ax=ax2, palette="magma", errorbar="sd")
    ax2.set_title("Average Precision by Model Configuration", fontsize=14)
    ax2.set_ylabel("Precision (%)")
    ax2.set_ylim(0, 105)

    sns.barplot(data=df, x="config", y="accuracy", ax=ax3, palette="rocket", errorbar="sd")
    ax3.set_title(f"Average Accuracy by Model (Threshold={args.threshold})", fontsize=14)
    ax3.set_ylabel("Accuracy (%)")
    ax3.set_ylim(0, 105)

    plt.tight_layout()
    plot_path = "model_comparison_plots.png"
    plt.savefig(plot_path, dpi=300)
    print(f"✓ Summary bar plots saved to {plot_path}")

    # Metric Box Plots (Per-paper distribution)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
    sns.boxplot(data=df, x="config", y="recall", ax=ax1, palette="viridis")
    ax1.set_title("Distribution of Recall by Model", fontsize=14)
    ax1.set_ylabel("Recall (%)")
    ax1.set_ylim(-5, 105)

    sns.boxplot(data=df, x="config", y="precision", ax=ax2, palette="magma")
    ax2.set_title("Distribution of Precision by Model", fontsize=14)
    ax2.set_ylabel("Precision (%)")
    ax2.set_ylim(-5, 105)

    sns.boxplot(data=df, x="config", y="accuracy", ax=ax3, palette="rocket")
    ax3.set_title(f"Distribution of Accuracy by Model (Threshold={args.threshold})", fontsize=14)
    ax3.set_ylabel("Accuracy (%)")
    ax3.set_ylim(-5, 105)

    plt.tight_layout()
    box_plot_path = "model_metrics_boxplots.png"
    plt.savefig(box_plot_path, dpi=300)
    print(f"Box plots of metrics saved to {box_plot_path}")

    # ========================================================================
    # SECONDARY ANALYSIS: Distribution Comparison (Sanity Check)
    # ========================================================================
    
    # Add GT data to distribution plot data
    for doi, log_val in gt_values_collected:
        conductivity_plot_data.append({
            "log_sigma": log_val,
            "Source": "Ground Truth"
        })

    if conductivity_plot_data:
        print(f"\n{'='*80}")
        print("SECONDARY ANALYSIS: Distribution Comparison (Sanity Check)")
        print(f"{'='*80}")
        print("Note: Distribution similarity does NOT guarantee accurate point-to-point predictions.")
        print("This analysis is provided as a supplementary sanity check for systematic bias.")
        print(f"{'='*80}\n")
        
        # Create a separate figure for distribution to keep it legible
        dist_df = pd.DataFrame(conductivity_plot_data)
        
        # Filter out extreme outliers if any to keep plot clean
        q_low = dist_df["log_sigma"].quantile(0.01)
        q_hi  = dist_df["log_sigma"].quantile(0.99)
        dist_df = dist_df[(dist_df["log_sigma"] > q_low - 2) & (dist_df["log_sigma"] < q_hi + 2)]

        # KDE and Boxplot side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        sns.kdeplot(data=dist_df, x="log_sigma", hue="Source", common_norm=False, fill=True, alpha=0.3, ax=ax1)
        ax1.set_title("KDE: Distribution of Room Temp Ionic Conductivities\n(Supplementary Analysis)", fontsize=16)
        ax1.set_xlabel("log₁₀(σ / S cm⁻¹)")
        ax1.set_ylabel("Density")
        ax1.grid(True, linestyle='--', alpha=0.6)

        sns.boxplot(data=dist_df, x="log_sigma", y="Source", palette="Set2", ax=ax2)
        ax2.set_title("Boxplot: Distribution of Room Temp Ionic Conductivities\n(Supplementary Analysis)", fontsize=16)
        ax2.set_xlabel("log₁₀(σ / S cm⁻¹)")
        ax2.set_ylabel("Source")
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        dist_plot_path = "conductivity_distribution_comparison.png"
        plt.savefig(dist_plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Distribution plots (KDE + Box) saved to {dist_plot_path}")

        # Calculate and print Means
        print(f"\n--- Mean Ionic Conductivity (log₁₀ S/cm) ---")
        means = dist_df.groupby("Source")["log_sigma"].mean().sort_values()
        print(means)
        print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
