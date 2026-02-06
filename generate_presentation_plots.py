import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def generate_plots(data_path, output_dir):
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data['papers'])
    
    # Set the style
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['axes.labelcolor'] = 'black'
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'
    
    # 1. Precision vs Recall Scatter Plot
    plt.figure(figsize=(10, 8))
    scatter = sns.scatterplot(
        data=df, 
        x='recall', 
        y='precision', 
        size='total_gt', 
        hue='f1', 
        palette='viridis', 
        sizes=(50, 500),
        alpha=0.7
    )
    plt.title('Extraction Performance: Precision vs. Recall', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Recall (%)', fontsize=14)
    plt.ylabel('Precision (%)', fontsize=14)
    plt.xlim(-5, 105)
    plt.ylim(-5, 105)
    plt.axline((0, 0), slope=1, color='gray', linestyle='--', alpha=0.3, label='Ideal Balanced')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='F1 Score / GT Count')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/precision_vs_recall_scatter.png", dpi=300)
    plt.close()

    # 2. Metric Distributions (Boxplot + Swarmplot)
    metrics_melted = df.melt(value_vars=['recall', 'precision', 'numeric_accuracy'], var_name='Metric', value_name='Value')
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=metrics_melted, x='Metric', y='Value', palette='pastel', showfliers=False)
    sns.swarmplot(data=metrics_melted, x='Metric', y='Value', color='.25', alpha=0.6)
    plt.title('Distribution of Performance Metrics', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Score (%)', fontsize=14)
    plt.xlabel('', fontsize=14)
    # Fix x-axis labels
    plt.xticks([0, 1, 2], ['Recall', 'Precision', 'Numeric Accuracy'])
    plt.ylim(-5, 105)

    # increase tick label size
    plt.tick_params(axis='both', which='major', labelsize=18)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/metric_distributions.png", dpi=300)
    plt.close()

    # 3. Correlation: GT Size vs Performance
    plt.figure(figsize=(10, 6))
    sns.regplot(data=df, x='total_gt', y='f1', scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    plt.title('Impact of Ground Truth Complexity on F1 Score', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Number of Ground Truth Entries', fontsize=14)
    plt.ylabel('F1 Score', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/complexity_vs_f1.png", dpi=300)
    plt.close()

    print(f"Scientific plots generated in {output_dir}")

if __name__ == "__main__":
    data_path = "/Users/bourn23/Downloads/general/PageIndex/results_aggregated.json"
    output_dir = "/Users/bourn23/.gemini/antigravity/brain/042df6fd-b9c1-4622-90de-0486d876325b"
    generate_plots(data_path, output_dir)
