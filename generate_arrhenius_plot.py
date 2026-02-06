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
                        temp_k = temp_c + 273.15
                        if temp_k > 0:
                            all_data.append({
                                "material": mat.get("canonical_formula", "Unknown"),
                                "conductivity": cond,
                                "temp_c": temp_c,
                                "temp_k": temp_k,
                                "inv_temp": 1000.0 / temp_k,
                                "log_cond": np.log10(cond),
                                "source_file": os.path.basename(file_path)
                            })
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                
    return pd.DataFrame(all_data)

def plot_arrhenius(df, output_path):
    if df.empty:
        print("No data found for plotting.")
        return

    sns.set_theme(style="ticks", context="paper")
    plt.figure(figsize=(12, 10))
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    
    # Group by material to show different colors for main material classes
    # For a presentation, we might want to highlight top materials or just plot all
    
    # 1. Arrhenius Plot (Log Sigma vs 1000/T)
    scatter = sns.scatterplot(
        data=df,
        x='inv_temp',
        y='log_cond',
        alpha=0.6,
        s=100,
        hue='material',
        legend=False # Legend might be too big if many materials
    )
    
    # Add labels for some high performers if needed, or just keep it clean
    plt.title('Arrhenius Plot of Extracted Ionic Conductivities', fontsize=20, fontweight='bold', pad=25)
    plt.xlabel('1000 / T (K⁻¹)', fontsize=16)
    plt.ylabel('log₁₀(σ / S cm⁻¹)', fontsize=16)
    
    # Add a second x-axis for Celsius temperatures
    ax1 = plt.gca()
    ax2 = ax1.twiny()
    
    # Define interesting temperatures in C and convert to 1000/T(K)
    temp_c_ticks = np.array([-100, -50, 0, 25, 50, 100, 200, 300, 400])
    temp_k_ticks = temp_c_ticks + 273.15
    inv_temp_ticks = 1000.0 / temp_k_ticks
    
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(inv_temp_ticks)
    ax2.set_xticklabels([f"{t}°C" for t in temp_c_ticks])
    ax2.set_xlabel('Temperature (°C)', fontsize=14, labelpad=15)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Arrhenius plot saved to {output_path}")

if __name__ == "__main__":
    directories = [
        "/Users/bourn23/Downloads/general/PageIndex/fetched_papers/obelix_parsed",
        "/Users/bourn23/Downloads/general/PageIndex/fetched_papers/obelix_parsed2"
    ]
    df = extract_plotting_data(directories)
    output = "/Users/bourn23/.gemini/antigravity/brain/042df6fd-b9c1-4622-90de-0486d876325b/arrhenius_plot.png"
    plot_arrhenius(df, output)
