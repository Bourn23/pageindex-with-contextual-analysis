import os
import json
import time
from scifigure_parser import SciFigureParser

from dotenv import load_dotenv
load_dotenv()

def run_comparison():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        print("Please set GEMINI_API_KEY environment variable.")
        return

    models = [
        "gemini-2.0-flash",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-3-flash-preview",
        "gemini-3-pro-preview"
    ]

    sample_image = "/Users/bourn23/Downloads/general/PageIndex/fetched_papers/obelix_md2/10_3389_fenrg_2016_00028/page_7_figure_2.jpeg"
    query = "ionic conductivity"
    
    if not os.path.exists(sample_image):
        print(f"Sample image not found: {sample_image}")
        return

    results = {}

    for model_name in models:
        print(f"\n" + "="*50)
        print(f"Testing model: {model_name}")
        print("="*50)
        
        try:
            # Initialize parser with debug=True to get visualizations for each model
            parser = SciFigureParser(api_key=api_key, model_name=model_name, debug=True)
            
            start_time = time.time()
            
            # 1. Detection
            print(f"Running detection for '{query}'...")
            box = parser.detect_subplot(sample_image, query)
            
            # 2. Cropping
            cropped_path = parser.crop_image(sample_image, box, padding=80)
            
            # 3. Extraction
            print("Running data extraction...")
            result = parser.extract_data(cropped_path, grid_config={"enabled": True, "rows": 2, "cols": 2})
            
            elapsed_time = time.time() - start_time
            
            # Save model-specific results
            output_file = f"result_{model_name.replace('-', '_')}.json"
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)
            
            results[model_name] = {
                "status": "success",
                "elapsed_time": elapsed_time,
                "data_points_count": len(result.get("dataPoints", [])),
                "title": result.get("title"),
                "output_file": output_file
            }
            
            print(f"Model {model_name} completed in {elapsed_time:.2f}s. Extracted {len(result.get('dataPoints', []))} points.")
            
        except Exception as e:
            print(f"Error testing model {model_name}: {e}")
            results[model_name] = {
                "status": "error",
                "error": str(e)
            }

    # Final Summary
    print("\n" + "#"*50)
    print("COMPARISON SUMMARY")
    print("#"*50)
    print(f"{'Model':<30} | {'Status':<10} | {'Time (s)':<10} | {'Points':<10}")
    print("-" * 65)
    for model, data in results.items():
        if data["status"] == "success":
            print(f"{model:<30} | {data['status']:<10} | {data['elapsed_time']:<10.2f} | {data['data_points_count']:<10}")
        else:
            print(f"{model:<30} | {data['status']:<10} | {'N/A':<10} | {'N/A':<10}")
    
    with open("comparison_report.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull comparison report saved to comparison_report.json")

if __name__ == "__main__":
    run_comparison()
