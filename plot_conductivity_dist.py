import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_conductivity_distribution(csv_path, output_path):
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Extract conductivity column
    cond_col = 'Ionic conductivity (S cm-1)'
    if cond_col not in df.columns:
        print(f"Error: {cond_col} not found in CSV.")
        return

    # Filter out non-positive values for log scale
    valid_data = df[df[cond_col] > 0].copy()
    valid_data['log_cond'] = np.log10(valid_data[cond_col])

    # Set up the figure
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.figure(figsize=(12, 7))
    
    # Plot distribution
    sns.histplot(valid_data['log_cond'], bins=40, kde=True, color='#0366d6', alpha=0.6)
    
    # Plot styling
    plt.title('Distribution of Ionic Conductivities (RT / 25 °C)', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel(r'$log_{10}(\sigma / S cm^{-1})$', fontsize=22, color='black')
    plt.ylabel('Count', fontsize=22, color='black')
    
    # Add stats lines or annotations if needed
    mean_log = valid_data['log_cond'].median()
    plt.axvline(mean_log, color='#ff7b72', linestyle='--', label=f'Mean: {mean_log:.2f}', linewidth=2)
    plt.legend(fontsize=18)

    plt.grid(True, linestyle=':', alpha=0.7)
    
    # increase tick label size
    plt.tick_params(axis='both', which='major', labelsize=18)

    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path.replace(".png", ".pdf"), dpi=600, bbox_inches='tight', facecolor='white', transparent=True)
    plt.close()
    print(f"Conductivity distribution plot saved to {output_path}")

if __name__ == "__main__":
    csv_path = "/Users/bourn23/Downloads/general/PageIndex/obelix_data_with_processing_method_filtered.csv"
    output_path = "./conductivity_distribution.png"
    plot_conductivity_distribution(csv_path, output_path)
