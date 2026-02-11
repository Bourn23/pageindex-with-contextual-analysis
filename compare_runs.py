import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def parse_validation_report(file_path):
    """
    Parses a validation report text file.
    Returns a list of dictionaries with status for each GT item.
    """
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('GT-'):
                # Format: GT-ID | STATUS | MESSAGE
                parts = line.split('|')
                if len(parts) >= 2:
                    gt_id = parts[0].strip()
                    status_part = parts[1].strip()
                    
                    is_match = "✅ MATCH" in status_part
                    results.append({
                        "gt_id": gt_id,
                        "is_match": is_match
                    })
    return results

def get_run_statistics(directory):
    """
    Aggregates statistics for all validation reports in a directory.
    """
    path = Path(directory)
    if not path.exists():
        print(f"Warning: Directory {directory} does not exist.")
        return None

    reports = list(path.glob("*_extracted_validation_report.txt"))
    all_data = []
    
    for report in reports:
        paper_name = report.name.replace("_v5_extracted_validation_report.txt", "")
        items = parse_validation_report(report)
        
        if not items:
            continue
            
        matches = sum(1 for item in items if item['is_match'])
        total = len(items)
        accuracy = (matches / total) * 100 if total > 0 else 0
        
        all_data.append({
            "paper": paper_name,
            "matches": matches,
            "total_gt": total,
            "accuracy": accuracy
        })
        
    return pd.DataFrame(all_data)

def main():
    test1_dir = "/Users/bourn23/Downloads/general/PageIndex/fetched_papers/obelix_parsed_v5_test1"
    test2_dir = "/Users/bourn23/Downloads/general/PageIndex/fetched_papers/obelix_parsed_v5"
    
    print("Processing Run 1 (Test 1)...")
    df1 = get_run_statistics(test1_dir)
    if df1 is not None:
        df1['run'] = 'Run 1 (Test 1)'
        
    print("Processing Run 2 (Test 2)...")
    df2 = get_run_statistics(test2_dir)
    if df2 is not None:
        df2['run'] = 'Run 2 (Test 2)'
        
    if df1 is None or df2 is None:
        print("Error: Could not process one or both directories.")
        return

    # Combine data
    df_combined = pd.concat([df1, df2], ignore_index=True)
    
    # Calculate aggregate stats
    summary = df_combined.groupby('run').agg({
        'matches': 'sum',
        'total_gt': 'sum'
    }).reset_index()
    summary['accuracy'] = (summary['matches'] / summary['total_gt']) * 100
    
    print("\nAggregate Statistics:")
    print(summary)
    
    # Save results to CSV
    df_combined.to_csv("run_comparison_results.csv", index=False)
    print("\nDetailed results saved to run_comparison_results.csv")
    
    # Plotting
    sns.set_theme(style="whitegrid")
    
    # 1. Overall Accuracy Plot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=summary, x='run', y='accuracy', palette="viridis")
    plt.title('Overall Extraction Accuracy Comparison', fontsize=16)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xlabel('Run', fontsize=12)
    plt.ylim(0, 105)
    
    # Add values on top of bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 9), 
                    textcoords='offset points',
                    fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("overall_accuracy_comparison.png", dpi=300)
    print("Saved overall_accuracy_comparison.png")
    
    # 2. Per-Paper Comparison Plot
    # We want to see how each paper changed between runs
    # Pivot for easier plotting of deltas
    pivot_df = df_combined.pivot(index='paper', columns='run', values='accuracy').reset_index()
    pivot_df['delta'] = pivot_df['Run 2 (Test 2)'] - pivot_df['Run 1 (Test 1)']
    
    plt.figure(figsize=(12, max(6, len(pivot_df) * 0.4)))
    pivot_df_sorted = pivot_df.sort_values('delta', ascending=False)
    
    # Create a divergent color palette for delta
    colors = ['green' if x >= 0 else 'red' for x in pivot_df_sorted['delta']]
    
    ax = sns.barplot(data=pivot_df_sorted, y='paper', x='delta', palette=colors)
    plt.title('Accuracy Change per Paper (Run 2 - Run 1)', fontsize=16)
    plt.xlabel('Change in Accuracy (percentage points)', fontsize=12)
    plt.ylabel('Paper', fontsize=12)
    
    plt.tight_layout()
    plt.savefig("per_paper_accuracy_change.png", dpi=300)
    print("Saved per_paper_accuracy_change.png")
    
    # 3. Accuracy Scatter Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=pivot_df, x='Run 1 (Test 1)', y='Run 2 (Test 2)', s=100, alpha=0.7)
    
    # Add y=x line
    lims = [0, 105]
    plt.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='No Change')
    
    plt.title('Accuracy: Run 1 vs Run 2', fontsize=16)
    plt.xlabel('Run 1 Accuracy (%)', fontsize=12)
    plt.ylabel('Run 2 Accuracy (%)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig("run_comparison_scatter.png", dpi=300)
    print("Saved run_comparison_scatter.png")

if __name__ == "__main__":
    main()
