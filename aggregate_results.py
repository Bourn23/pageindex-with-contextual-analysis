import os
import json
import glob

def aggregate_results(paths, output_file):
    results = []
    
    for path in paths:
        search_pattern = os.path.join(path, "*_metrics.json")
        files = glob.glob(search_pattern)
        print(f"Searching in {path}: Found {len(files)} files")
        
        for file_path in files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Extract filename base as title
                filename = os.path.basename(file_path)
                title = filename.replace("_structure_materials_metrics.json", "").replace("_", " ").title()
                
                metrics = data.get("metrics", {})
                
                results.append({
                    "title": title,
                    "recall": metrics.get("recall", 0),
                    "precision": metrics.get("precision", 0),
                    "f1": metrics.get("f1", 0),
                    "avg_log10_error": metrics.get("avg_log10_error", 0),
                    "numeric_accuracy": metrics.get("numeric_accuracy_pct", 0),
                    "total_gt": metrics.get("total_gt", 0),
                    "total_ex": metrics.get("total_ex", 0),
                    "source_dir": os.path.basename(path)
                })
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    # Calculate overall averages
    if results:
        summary = {
            "avg_recall": sum(r["recall"] for r in results) / len(results),
            "avg_precision": sum(r["precision"] for r in results) / len(results),
            "avg_f1": sum(r["f1"] for r in results) / len(results),
            "total_papers": len(results),
            "papers": results
        }
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Aggregated results saved to {output_file}")
    else:
        print("No results found to aggregate.")

if __name__ == "__main__":
    directories = [
        "/Users/bourn23/Downloads/general/PageIndex/fetched_papers/obelix_parsed",
        "/Users/bourn23/Downloads/general/PageIndex/fetched_papers/obelix_parsed2"
    ]
    output = "/Users/bourn23/Downloads/general/PageIndex/results_aggregated.json"
    aggregate_results(directories, output)
