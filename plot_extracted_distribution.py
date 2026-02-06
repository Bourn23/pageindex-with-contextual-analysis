import os
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def extract_plotting_data(paths):
    all_data = []
    
    for path in paths:
        search_pattern = os.path.join(path, "*_materials.json")
        for file_path in glob.glob(search_pattern):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                materials = data.get("materials", [])
                for mat in materials:
                    cond = mat.get("_norm_cond")
                    temp_c = mat.get("_norm_temp")
                    
                    # Manual parsing fallback if normalized fields are missing
                    if cond is None and "ionic_conductivity_S_per_cm" in mat:
                        try:
                            # Simple extraction for 1.2 x 10^-3 format
                            val_str = mat["ionic_conductivity_S_per_cm"].split(' ')[0].replace('×', 'x').replace('⁻', '-')
                            if 'x' in val_str:
                                base, exp = val_str.split('x10')
                                cond = float(base) * (10 ** float(exp.replace('^', '')))
                            else:
                                cond = float(val_str)
                        except:
                            continue
                    
                    if temp_c is None and "measurement_temperature" in mat:
                        try:
                            temp_str = mat["measurement_temperature"]
                            if '°C' in temp_str:
                                temp_c = float(temp_str.replace('°C', '').strip())
                            elif 'K' in temp_str:
                                temp_c = float(temp_str.replace('K', '').strip()) - 273.15
                            elif 'room temperature' in temp_str.lower():
                                temp_c = 25.0
                        except:
                            continue
                    
                    if cond is not None and temp_c is not None and cond > 0:
                        all_data.append({
                            "material": mat.get("canonical_formula", "Unknown"),
                            "conductivity": cond,
                            "temp_c": temp_c,
                            "source_file": os.path.basename(file_path)
                        })
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                
    return pd.DataFrame(all_data)

def plot_distribution(df, output_path):
    if df.empty:
        print("No data found for plotting.")
        return

    # Filter for data points near room temperature (20-30°C) to be comparable
    rt_data = df[(df['temp_c'] >= 20) & (df['temp_c'] <= 30)].copy()
    
    if rt_data.empty:
        print("No room temperature data found (20-30°C). Plotting all data.")
        rt_data = df.copy()
    
    # Filter out conductivities > 1e-2 as requested by user
    initial_count = len(rt_data)
    rt_data = rt_data[rt_data['conductivity'] <= 1e-2].copy()
    final_count = len(rt_data)
    
    if initial_count > final_count:
        print(f"Filtered out {initial_count - final_count} points with conductivity > 1e-2.")
    
    rt_data['log_cond'] = np.log10(rt_data['conductivity'])

    # Set up the figure (using style from plot_conductivity_dist.py)
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.figure(figsize=(12, 7))
    
    # Plot distribution
    sns.histplot(rt_data['log_cond'], bins=40, kde=True, color='#28a745', alpha=0.6) # Using green for extracted data
    
    # Plot styling
    plt.title('Distribution of Extracted Ionic Conductivities (RT)', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel(r'$log_{10}(\sigma / S cm^{-1})$', fontsize=22, color='black')
    plt.ylabel('Count', fontsize=22, color='black')
    
    # Adjust x-axis range and ticks as requested
    plt.xlim(-20, 0)
    plt.xticks(np.arange(-18, 0, 2))
    
    # Add stats lines
    mean_log = rt_data['log_cond'].mean()
    plt.axvline(mean_log, color='#ff7b72', linestyle='--', label=f'Mean: {mean_log:.2f}', linewidth=2)
    plt.legend(fontsize=18)

    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tick_params(axis='both', which='major', labelsize=18)

    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white', transparent=True)
    # Also save as PDF for high quality
    plt.savefig(output_path.replace(".png", ".pdf"), dpi=600, bbox_inches='tight', facecolor='white', transparent=True)
    plt.close()
    print(f"Extracted distribution plot saved to {output_path}")

if __name__ == "__main__":
    directories = [
        "/Users/bourn23/Downloads/general/PageIndex/fetched_papers/obelix_parsed",
        "/Users/bourn23/Downloads/general/PageIndex/fetched_papers/obelix_parsed2"
    ]
    df = extract_plotting_data(directories)
    output = "/Users/bourn23/Downloads/general/PageIndex/extracted_conductivity_distribution.png"
    plot_distribution(df, output)
