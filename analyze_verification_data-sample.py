#!/usr/bin/env python3
"""
Analyze verification data comparing automatic workflow vs manual verification.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('master_conductivity_data_downselected_papers_new_jy2-csv.csv')

print("=" * 80)
print("VERIFICATION DATA ANALYSIS")
print("=" * 80)
print(f"\nTotal entries: {len(df)}")
print(f"\nColumns: {df.columns.tolist()}")

# Analyze the 'verified ionic conductivity' column
verified_col = 'verified ionic conductivity'
auto_col = 'ionic_conductivity_S_per_cm'

print("\n" + "=" * 80)
print("VERIFICATION STATUS BREAKDOWN")
print("=" * 80)

# Categorize verification results
def categorize_verification(value):
    """Categorize verification status."""
    if pd.isna(value):
        return 'missing feedback'
    value_str = str(value).strip().lower()
    
    if value_str == '' or value_str == 'nan':
        return 'missing feedback'
    elif 'no data exists' in value_str:
        return 'no_data_exists'
    elif value_str == 'good':
        return 'good'
    else:
        # Try to parse as numeric
        try:
            float(value)
            return 'bad'
        except:
            # Check for range indicators
            if '<' in value_str or '>' in value_str or 'to' in value_str or 'didn\'t include' in value_str:
                return 'range_or_note'
            return 'other'

df['verification_category'] = df[verified_col].apply(categorize_verification)

# Count by category
category_counts = df['verification_category'].value_counts()
print("\nVerification Categories:")
for cat, count in category_counts.items():
    pct = (count / len(df)) * 100
    print(f"  {cat:20s}: {count:4d} ({pct:5.1f}%)")

# Detailed analysis for numeric comparisons
print("\n" + "=" * 80)
print("NUMERIC COMPARISON ANALYSIS")
print("=" * 80)

# Extract numeric verified values
def parse_verified_value(value):
    """Parse verified value to numeric."""
    if pd.isna(value):
        return np.nan
    value_str = str(value).strip().lower()
    
    # Handle 'good' - means automatic value is correct
    if value_str == 'good':
        return 'GOOD'
    
    # Handle 'no data exists'
    if 'no data exists' in value_str:
        return 'NO_DATA'
    
    # Try to parse as numeric
    try:
        return float(value)
    except:
        return np.nan

df['verified_numeric'] = df[verified_col].apply(parse_verified_value)

# Separate into categories for analysis
numeric_mask = df['verification_category'] == 'bad'
good_mask = df['verification_category'] == 'good'
no_data_mask = df['verification_category'] == 'no_data_exists'
missing_mask = df['verification_category'] == 'missing feedback'

print(f"\nNumeric comparisons available: {numeric_mask.sum()}")
print(f"Marked as 'good': {good_mask.sum()}")
print(f"No data exists in paper: {no_data_mask.sum()}")
print(f"Missing verification: {missing_mask.sum()}")

# Analyze numeric comparisons
if numeric_mask.sum() > 0:
    numeric_df = df[numeric_mask].copy()
    numeric_df['verified_numeric'] = pd.to_numeric(numeric_df['verified_numeric'], errors='coerce')
    numeric_df['auto_numeric'] = pd.to_numeric(numeric_df[auto_col], errors='coerce')
    
    # Remove any rows where conversion failed
    valid_mask = numeric_df['verified_numeric'].notna() & numeric_df['auto_numeric'].notna()
    numeric_df = numeric_df[valid_mask]
    
    print(f"\nValid numeric comparisons: {len(numeric_df)}")
    
    if len(numeric_df) > 0:
        # Calculate differences
        numeric_df['absolute_diff'] = numeric_df['auto_numeric'] - numeric_df['verified_numeric']
        numeric_df['relative_diff'] = (numeric_df['absolute_diff'] / numeric_df['verified_numeric']) * 100
        numeric_df['log_auto'] = np.log10(numeric_df['auto_numeric'])
        numeric_df['log_verified'] = np.log10(numeric_df['verified_numeric'])
        numeric_df['log_diff'] = numeric_df['log_auto'] - numeric_df['log_verified']
        
        # Calculate accuracy metrics
        numeric_df['within_10pct'] = np.abs(numeric_df['relative_diff']) <= 10
        numeric_df['within_50pct'] = np.abs(numeric_df['relative_diff']) <= 50
        numeric_df['within_1_order'] = np.abs(numeric_df['log_diff']) <= 1
        numeric_df['within_2_orders'] = np.abs(numeric_df['log_diff']) <= 2
        
        print("\n" + "-" * 80)
        print("ACCURACY METRICS")
        print("-" * 80)
        print(f"Within 10% of verified value:  {numeric_df['within_10pct'].sum():3d} ({numeric_df['within_10pct'].mean()*100:5.1f}%)")
        print(f"Within 50% of verified value:  {numeric_df['within_50pct'].sum():3d} ({numeric_df['within_50pct'].mean()*100:5.1f}%)")
        print(f"Within 1 order of magnitude:   {numeric_df['within_1_order'].sum():3d} ({numeric_df['within_1_order'].mean()*100:5.1f}%)")
        print(f"Within 2 orders of magnitude:  {numeric_df['within_2_orders'].sum():3d} ({numeric_df['within_2_orders'].mean()*100:5.1f}%)")
        
        print("\n" + "-" * 80)
        print("ERROR STATISTICS")
        print("-" * 80)
        print(f"Mean absolute error:           {numeric_df['absolute_diff'].mean():.2e} S/cm")
        print(f"Median absolute error:         {numeric_df['absolute_diff'].median():.2e} S/cm")
        print(f"Mean relative error:           {numeric_df['relative_diff'].mean():.1f}%")
        print(f"Median relative error:         {numeric_df['relative_diff'].median():.1f}%")
        print(f"Mean log10 difference:         {numeric_df['log_diff'].mean():.2f}")
        print(f"Median log10 difference:       {numeric_df['log_diff'].median():.2f}")
        
        # Show worst cases
        print("\n" + "-" * 80)
        print("LARGEST DISCREPANCIES (Top 5)")
        print("-" * 80)
        worst = numeric_df.nlargest(5, 'log_diff')[['acronym', 'auto_numeric', 'verified_numeric', 'log_diff', 'relative_diff']]
        for idx, row in worst.iterrows():
            print(f"\n{row['acronym']}")
            print(f"  Automatic:  {row['auto_numeric']:.2e} S/cm")
            print(f"  Verified:   {row['verified_numeric']:.2e} S/cm")
            print(f"  Log diff:   {row['log_diff']:.2f} orders of magnitude")
            print(f"  Rel diff:   {row['relative_diff']:.1f}%")
        
        # Save detailed comparison
        output_cols = ['paper_title', 'acronym', 'material_class', 'temperature', 
                      'auto_numeric', 'verified_numeric', 'absolute_diff', 'relative_diff', 
                      'log_diff', 'verification notes']
        numeric_df[output_cols].to_csv('verification_numeric_comparison.csv', index=False)
        print(f"\n✓ Saved detailed comparison to: verification_numeric_comparison.csv")

# Summary statistics
print("\n" + "=" * 80)
print("OVERALL SUMMARY")
print("=" * 80)

total = len(df)
correct_good = good_mask.sum()
correct_numeric = numeric_mask.sum() if numeric_mask.sum() > 0 else 0
incorrect_no_data = no_data_mask.sum()
missing_verification = missing_mask.sum()

print(f"\nTotal entries analyzed:        {total}")
print(f"\nCorrect (marked 'good'):       {correct_good:3d} ({correct_good/total*100:5.1f}%)")
print(f"Numeric values to verify:      {correct_numeric:3d} ({correct_numeric/total*100:5.1f}%)")
print(f"Incorrect (no data exists):    {incorrect_no_data:3d} ({incorrect_no_data/total*100:5.1f}%)")
print(f"Missing verification:          {missing_verification:3d} ({missing_verification/total*100:5.1f}%)")

# Create visualizations
print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Automatic Workflow vs Manual Verification Analysis', fontsize=16, fontweight='bold')

# 1. Verification category breakdown
ax1 = axes[0, 0]
category_counts.plot(kind='bar', ax=ax1, color='steelblue')
ax1.set_title('Verification Status Breakdown')
ax1.set_xlabel('Category')
ax1.set_ylabel('Count')
ax1.tick_params(axis='x', rotation=45)
for i, v in enumerate(category_counts.values):
    ax1.text(i, v + 0.5, str(v), ha='center', va='bottom')

# 2. Pie chart of main categories
ax2 = axes[0, 1]
main_categories = {
    'Correct (good)': correct_good,
    'Needs verification': correct_numeric,
    'Incorrect (no data)': incorrect_no_data,
    'Missing': missing_verification
}
colors = ['#2ecc71', '#f39c12', '#e74c3c', '#95a5a6']
ax2.pie(main_categories.values(), labels=main_categories.keys(), autopct='%1.1f%%',
        colors=colors, startangle=90)
ax2.set_title('Overall Verification Status')

# 3. Scatter plot of automatic vs verified (if numeric data available)
ax3 = axes[1, 0]
if numeric_mask.sum() > 0 and len(numeric_df) > 0:
    ax3.scatter(numeric_df['log_verified'], numeric_df['log_auto'], alpha=0.6, s=50)
    
    # Add diagonal line (perfect agreement)
    min_val = min(numeric_df['log_verified'].min(), numeric_df['log_auto'].min())
    max_val = max(numeric_df['log_verified'].max(), numeric_df['log_auto'].max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect agreement', linewidth=2)
    
    # Add ±1 order of magnitude lines
    ax3.plot([min_val, max_val], [min_val+1, max_val+1], 'g--', alpha=0.5, label='±1 order of magnitude')
    ax3.plot([min_val, max_val], [min_val-1, max_val-1], 'g--', alpha=0.5)
    
    ax3.set_xlabel('Verified (log10 S/cm)')
    ax3.set_ylabel('Automatic (log10 S/cm)')
    ax3.set_title('Automatic vs Verified Values')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
else:
    ax3.text(0.5, 0.5, 'No numeric\ncomparisons available', 
             ha='center', va='center', fontsize=12, transform=ax3.transAxes)
    ax3.set_title('Automatic vs Verified Values')

# 4. Error distribution
ax4 = axes[1, 1]
if numeric_mask.sum() > 0 and len(numeric_df) > 0:
    ax4.hist(numeric_df['log_diff'], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax4.axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect agreement')
    ax4.axvline(1, color='green', linestyle='--', linewidth=1, alpha=0.5, label='±1 order')
    ax4.axvline(-1, color='green', linestyle='--', linewidth=1, alpha=0.5)
    ax4.set_xlabel('Log10 Difference (orders of magnitude)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Error Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
else:
    ax4.text(0.5, 0.5, 'No numeric\ncomparisons available', 
             ha='center', va='center', fontsize=12, transform=ax4.transAxes)
    ax4.set_title('Error Distribution')

plt.tight_layout()
plt.savefig('verification_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved visualization to: verification_analysis.png")

# Save summary report
with open('verification_summary_report.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("VERIFICATION DATA ANALYSIS SUMMARY\n")
    f.write("=" * 80 + "\n\n")
    
    f.write(f"Total entries analyzed: {total}\n\n")
    
    f.write("VERIFICATION STATUS:\n")
    f.write(f"  Correct (marked 'good'):       {correct_good:3d} ({correct_good/total*100:5.1f}%)\n")
    f.write(f"  Numeric values to verify:      {correct_numeric:3d} ({correct_numeric/total*100:5.1f}%)\n")
    f.write(f"  Incorrect (no data exists):    {incorrect_no_data:3d} ({incorrect_no_data/total*100:5.1f}%)\n")
    f.write(f"  Missing verification:          {missing_verification:3d} ({missing_verification/total*100:5.1f}%)\n\n")
    
    if numeric_mask.sum() > 0 and len(numeric_df) > 0:
        f.write("ACCURACY METRICS (for numeric comparisons):\n")
        f.write(f"  Within 10% of verified value:  {numeric_df['within_10pct'].sum():3d} ({numeric_df['within_10pct'].mean()*100:5.1f}%)\n")
        f.write(f"  Within 50% of verified value:  {numeric_df['within_50pct'].sum():3d} ({numeric_df['within_50pct'].mean()*100:5.1f}%)\n")
        f.write(f"  Within 1 order of magnitude:   {numeric_df['within_1_order'].sum():3d} ({numeric_df['within_1_order'].mean()*100:5.1f}%)\n")
        f.write(f"  Within 2 orders of magnitude:  {numeric_df['within_2_orders'].sum():3d} ({numeric_df['within_2_orders'].mean()*100:5.1f}%)\n\n")
        
        f.write("ERROR STATISTICS:\n")
        f.write(f"  Mean log10 difference:         {numeric_df['log_diff'].mean():.2f} orders of magnitude\n")
        f.write(f"  Median log10 difference:       {numeric_df['log_diff'].median():.2f} orders of magnitude\n")

print("✓ Saved summary report to: verification_summary_report.txt")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print("\nGenerated files:")
print("  - verification_analysis.png")
print("  - verification_summary_report.txt")
if numeric_mask.sum() > 0:
    print("  - verification_numeric_comparison.csv")
