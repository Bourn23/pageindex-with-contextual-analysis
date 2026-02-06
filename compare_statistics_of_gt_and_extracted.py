import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# --- 1. EXTRACTION LOGIC (Based on your provided code) ---

def get_extracted_df(paths):
    all_data = []
    for path in paths:
        for file_path in glob.glob(os.path.join(path, "*_materials.json")):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                for mat in data.get("materials", []):
                    cond = mat.get("_norm_cond")
                    temp_c = mat.get("_norm_temp")
                    # ... (Your manual parsing logic here) ...
                    if cond and temp_c and cond > 0:
                        all_data.append({"log_cond": np.log10(cond), "temp_c": temp_c})
            except Exception: continue
    df = pd.DataFrame(all_data)
    print("SHAPE OF EXTRACTED DATA: ", df.shape)
    # CRITICAL: Filter to Room Temp to match GT data (e.g., 20-30°C)
    return df[(df['temp_c'] >= 20) & (df['temp_c'] <= 30)]

def get_original_df(csv_path):
    df = pd.read_csv(csv_path)
    cond_col = 'Ionic conductivity (S cm-1)'
    valid_data = df[df[cond_col] > 0].copy()
    valid_data['log_cond'] = np.log10(valid_data[cond_col])
    return valid_data[['log_cond']]

# --- 2. COMPARISON AND PLOTTING ---

def compare_datasets(df_mine, df_gt):
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Density Plot (Comparing Shape)
    sns.kdeplot(df_mine['log_cond'], label=f'Extracted (N={len(df_mine)})', fill=True, ax=ax1)
    sns.kdeplot(df_gt['log_cond'], label=f'Original (N={len(df_gt)})', fill=True, ax=ax1)
    ax1.set_title("Density Comparison (Shape)")
    ax1.legend()

    # ECDF Plot (Comparing Coverage)
    sns.ecdfplot(df_mine['log_cond'], label='Extracted', ax=ax2)
    sns.ecdfplot(df_gt['log_cond'], label='Original', ax=ax2)
    ax2.set_title("Empirical Cumulative Distribution")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("comparison_results.png")
    
    # Statistical Tests
    ks_stat, p_val = stats.ks_2samp(df_mine['log_cond'], df_gt['log_cond'])
    print(f"--- Statistics ---")
    print(f"Extracted Mean: {df_mine['log_cond'].mean():.2f}")
    print(f"Original Mean:  {df_gt['log_cond'].mean():.2f}")
    print(f"KS Test p-value: {p_val:.4e}")
    
    if p_val < 0.05:
        print("Result: Distributions are statistically DIFFERENT.")
    else:
        print("Result: Distributions are statistically SIMILAR.")

# --- 3. RUN ---
if __name__ == "__main__":
    # Update these paths to your actual local paths
    ext_paths = ["./fetched_papers/obelix_parsed", "./fetched_papers/obelix_parsed2"]
    gt_path = "./obelix_data_with_processing_method_filtered.csv"
    
    df_mine = get_extracted_df(ext_paths)
    df_gt = get_original_df(gt_path)
    
    compare_datasets(df_mine, df_gt)