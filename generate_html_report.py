import pandas as pd
from pathlib import Path

def generate_html_report():
    csv_path = Path("evaluation_report.csv")
    if not csv_path.exists():
        print("Error: evaluation_report.csv not found.")
        return

    df = pd.read_csv(csv_path)
    
    # Sort for better readability: Status first (MISSING at top?), or maybe Found first?
    # Let's sort by DOI then Status
    df = df.sort_values(by=['DOI', 'Status'])

    html_content = """
    <html>
    <head>
        <title>OBELiX Pipeline Evaluation Report</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 20px; background: #f9f9f9; }}
            h1 {{ color: #333; }}
            .summary {{ margin-bottom: 20px; padding: 15px; background: #fff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            table {{ width: 100%; border-collapse: collapse; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.1); font-size: 14px; }}
            th {{ background: #f4f4f4; text-align: left; padding: 10px; border-bottom: 2px solid #ddd; position: sticky; top: 0; }}
            td {{ padding: 10px; border-bottom: 1px solid #eee; vertical-align: top; }}
            tr:hover {{ background-color: #f8f8f8; }}
            .status-found {{ color: #2e7d32; font-weight: bold; }}
            .status-missing {{ color: #d32f2f; font-weight: bold; }}
            .accuracy-good {{ color: #2e7d32; }}
            .accuracy-ok {{ color: #f57c00; }}
            .accuracy-bad {{ color: #d32f2f; }}
            .meta {{ font-size: 12px; color: #666; display: block; margin-top: 4px; }}
            .comp-raw {{ font-family: monospace; color: #555; background: #f0f0f0; padding: 2px 4px; border-radius: 3px; }}
        </style>
    </head>
    <body>
        <h1>OBELiX Extraction Pipeline Evaluation</h1>
        
        <div class="summary">
            <p><strong>Total Points:</strong> {total}</p>
            <p><strong>Matches Found:</strong> {found} ({recall:.1f}%)</p>
            <p><strong>Average Log Error:</strong> {avg_error:.4f}</p>
        </div>

        <table>
            <thead>
                <tr>
                    <th>DOI</th>
                    <th>Status</th>
                    <th>Ground Truth (Composition / Conductivity)</th>
                    <th>Extracted Match (Name / Conductivity)</th>
                    <th>Log Error</th>
                    <th>Metadata & Context</th>
                </tr>
            </thead>
            <tbody>
    """

    total = len(df)
    found = len(df[df['Status'] == 'FOUND'])
    recall = (found / total * 100) if total > 0 else 0
    avg_error = df[df['Status'] == 'FOUND']['Log_Error'].mean()

    html_content = html_content.format(total=total, found=found, recall=recall, avg_error=avg_error)

    for _, row in df.iterrows():
        status_class = "status-found" if row['Status'] == 'FOUND' else "status-missing"
        
        # Format GT
        gt_cell = f"<strong>{row['GT_Comp']}</strong><br>{row['GT_Cond']} S/cm"
        
        # Format Extraction
        if row['Status'] == 'FOUND':
            ext_comp = str(row['Ext_Comp_Raw']).replace("nan", "N/A")
            ext_cell = f"<span class='comp-raw'>{ext_comp}</span><br>{row['Ext_Cond']} S/cm"
            
            # Error Color
            err = float(row['Log_Error'])
            err_class = "accuracy-good" if err < 0.1 else ("accuracy-ok" if err < 0.3 else "accuracy-bad")
            err_cell = f"<span class='{err_class}'>{err:.4f}</span>"
            
            # Metadata
            meta = f"Temp: {row.get('Ext_Temp', 'N/A')}<br>Desc: {str(row.get('Ext_Desc', ''))[:100]}..."
        else:
            ext_cell = "<span style='color: #999;'>No Match Found</span>"
            err_cell = "-"
            meta = "-"

        html_content += f"""
            <tr>
                <td><small>{row['DOI']}</small></td>
                <td class="{status_class}">{row['Status']}</td>
                <td>{gt_cell}</td>
                <td>{ext_cell}</td>
                <td>{err_cell}</td>
                <td class="meta">{meta}</td>
            </tr>
        """

    html_content += """
            </tbody>
        </table>
    </body>
    </html>
    """

    output_path = "evaluation_report.html"
    with open(output_path, "w") as f:
        f.write(html_content)
    
    print(f"[++] HTML report generated: {output_path}")

if __name__ == "__main__":
    generate_html_report()
