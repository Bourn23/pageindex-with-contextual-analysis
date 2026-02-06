import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_sample_table_figure(csv_path, output_path):
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Select and rename columns as requested
    cols = ['Composition', 'Ionic conductivity (S cm-1)', 'DOI']
    sample_df = df[cols].head(10).copy() # Take top 10 as sample
    
    # Rename for better display
    sample_df.columns = ['Material Composition', 'Ionic Cond. (S/cm)', 'Source DOI']
    
    # Clean up DOI (just take first if multiple)
    sample_df['Source DOI'] = sample_df['Source DOI'].apply(lambda x: str(x).split('|')[0] if pd.notnull(x) else "")
    
    # Format Conductivity in scientific notation
    sample_df['Ionic Cond. (S/cm)'] = sample_df['Ionic Cond. (S/cm)'].apply(lambda x: f"{x:.2e}")

    # Set up the figure
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    ax.axis('tight')

    # Create the table
    table = ax.table(
        cellText=sample_df.values,
        colLabels=sample_df.columns,
        loc='center',
        cellLoc='left',
        colColours=['#f3f4f6']*3
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.5) # Scale height for better readability

    # Style cells
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#d1d5db')
        if row == 0:
            cell.set_text_props(weight='bold', color='#0366d6')
        else:
            cell.set_text_props(color='#24292e')
        cell.set_linewidth(1.0)

    plt.title('Sample of Original OBELiX Ground Truth Data', fontsize=18, fontweight='bold', color='#0366d6', pad=30)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=False, facecolor='white')
    plt.close()
    print(f"Sample data table figure saved to {output_path}")

if __name__ == "__main__":
    csv_path = "/Users/bourn23/Downloads/general/PageIndex/OBELiX/data/processed.csv"
    output_path = "/Users/bourn23/.gemini/antigravity/brain/042df6fd-b9c1-4622-90de-0486d876325b/original_data_sample.png"
    generate_sample_table_figure(csv_path, output_path)
