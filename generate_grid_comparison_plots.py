import os
import json
import matplotlib.pyplot as plt
import glob

def generate_plots():
    # Find all grid result files
    files = glob.glob("grid_res_gemini_*.json")
    
    # Structure to hold data: {model: {grid: [points]}}
    data = {}
    
    for file_path in files:
        # Extract model and grid info from filename
        # grid_res_gemini_2.0_flash_2x2.json
        parts = os.path.basename(file_path).replace("grid_res_", "").replace(".json", "").split("_")
        
        # This is a bit brittle, let's refine
        grid_label = parts[-1]
        model_name = "_".join(parts[:-1])
        
        with open(file_path, 'r') as f:
            content = json.load(f)
            points = content.get("dataPoints", [])
            
        if model_name not in data:
            data[model_name] = {}
        data[model_name][grid_label] = points

    # Set up the figure
    models = sorted(data.keys())
    num_models = len(models)
    
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 5), sharey=True)
    if num_models == 1:
        axes = [axes]
        
    markers = {'None': 'o', '2x2': 's', '3x3': '^', '4x4': 'D'}
    colors = {'None': 'black', '2x2': 'blue', '3x3': 'green', '4x4': 'red'}
    
    for i, model in enumerate(models):
        ax = axes[i]
        model_data = data[model]
        
        # Sort grids for consistent plotting if possible
        grids = sorted(model_data.keys(), key=lambda x: (x != 'None', x))
        
        for grid in grids:
            points = model_data[grid]
            if not points:
                continue
                
            x_vals = [p['xValue'] for p in points]
            y_vals = [p['yValue'] for p in points]
            
            label = f"Grid: {grid}"
            ax.scatter(x_vals, y_vals, label=label, marker=markers.get(grid, 'x'), color=colors.get(grid), alpha=0.6)
            
        ax.set_title(f"Model: {model}")
        ax.set_xlabel("1000/T (K⁻¹)")
        if i == 0:
            ax.set_ylabel("ln(σT) (S cm⁻¹ K)")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("grid_visual_comparison.png", dpi=300)
    print("Saved grid_visual_comparison.png")

if __name__ == "__main__":
    generate_plots()
