import os
import json
import time
from scifigure_parser import SciFigureParser
from dotenv import load_dotenv

load_dotenv()

def run_grid_comparison():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        print("Please set GEMINI_API_KEY environment variable.")
        return

    models = [
        "gemini-2.0-flash",
        "gemini-2.5-flash",
        "gemini-3-flash-preview"
    ]

    # Grid sizes: (rows, cols)
    # (0, 0) will represent disabled grid
    grid_sizes = [
        (0, 0), # Disabled
        (2, 2), # 2x2
        (3, 3), # 3x3
        (4, 4)  # 4x4
    ]

    sample_image = "/Users/bourn23/Downloads/general/PageIndex/fetched_papers/obelix_md2/10_3389_fenrg_2016_00028/page_7_figure_2.jpeg"
    query = "ionic conductivity"
    
    if not os.path.exists(sample_image):
        print(f"Sample image not found: {sample_image}")
        return

    results = []

    for model_name in models:
        # Pre-detect and crop to decouple grid effect on extraction from detection
        # We'll use the same model to detect and crop once for each model run to be fair
        try:
            temp_parser = SciFigureParser(api_key=api_key, model_name=model_name)
            box = temp_parser.detect_subplot(sample_image, query)
            cropped_path = temp_parser.crop_image(sample_image, box, padding=80)
        except Exception as e:
            print(f"Initial setup failed for model {model_name}: {e}")
            continue

        for rows, cols in grid_sizes:
            grid_label = f"{rows}x{cols}" if rows > 0 else "None"
            print(f"\n" + "="*50)
            print(f"Testing Model: {model_name} | Grid: {grid_label}")
            print("="*50)
            
            try:
                # Initialize parser (debug set to True for visual verification)
                parser = SciFigureParser(api_key=api_key, model_name=model_name, debug=True)
                
                grid_config = {"enabled": rows > 0, "rows": rows, "cols": cols}
                
                start_time = time.time()
                
                # Extraction
                print(f"Running data extraction with grid {grid_label}...")
                result = parser.extract_data(cropped_path, grid_config=grid_config)
                
                elapsed_time = time.time() - start_time
                
                output_file = f"grid_res_{model_name.replace('-', '_')}_{grid_label}.json"
                with open(output_file, "w") as f:
                    json.dump(result, f, indent=2)
                
                res_entry = {
                    "model": model_name,
                    "grid": grid_label,
                    "status": "success",
                    "elapsed_time": elapsed_time,
                    "data_points_count": len(result.get("dataPoints", [])),
                    "title": result.get("title"),
                    "output_file": output_file
                }
                results.append(res_entry)
                
                print(f"Completed in {elapsed_time:.2f}s. Points: {len(result.get('dataPoints', []))}")
                
            except Exception as e:
                print(f"Error testing Model {model_name} with Grid {grid_label}: {e}")
                results.append({
                    "model": model_name,
                    "grid": grid_label,
                    "status": "error",
                    "error": str(e)
                })

    # Final Summary Table
    print("\n" + "#"*70)
    print("GRID SIZE EFFECT COMPARISON SUMMARY")
    print("#"*70)
    print(f"{'Model':<25} | {'Grid':<10} | {'Status':<10} | {'Time (s)':<10} | {'Points':<10}")
    print("-" * 75)
    for res in results:
        if res["status"] == "success":
            print(f"{res['model']:<25} | {res['grid']:<10} | {res['status']:<10} | {res['elapsed_time']:<10.2f} | {res['data_points_count']:<10}")
        else:
            print(f"{res['model']:<25} | {res['grid']:<10} | {res['status']:<10} | {'N/A':<10} | {'N/A':<10}")
    
    with open("grid_comparison_report.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull report saved to grid_comparison_report.json")

if __name__ == "__main__":
    run_grid_comparison()
