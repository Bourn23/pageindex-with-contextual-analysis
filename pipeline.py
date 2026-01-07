#!/usr/bin/env python3
"""
Full Pipeline for Ionic Conductivity Data Extraction and Visualization.
Combines tree structure parsing, data extraction, and dual visualization.

Usage:
    python pipeline.py path/to/file.md
    python pipeline.py path/to/file.md --asset_dir path/to/assets
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
def run_command(command):
    """Run a shell command and print its output."""
    print(f"\n>>> Running: {' '.join(command)}")
    result = subprocess.run(command, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"Error: Command failed with return code {result.returncode}")
        # sys.exit(1) # Don't exit, maybe some steps failed but others can continue
    return result.returncode == 0


def process_file(md_path, asset_dir_override, model, batch_size, results_dir):
    """Run the full pipeline for a single markdown file."""
    md_path = Path(md_path)
    if not md_path.exists():
        print(f"Error: Markdown file not found: {md_path}")
        return False
        
    # Determine asset directory
    if asset_dir_override:
        asset_dir = Path(asset_dir_override)
    else:
        # Defaults to the directory where the markdown file is located
        asset_dir = md_path.parent
        
    basename = md_path.stem
    tree_json = results_dir / f"{basename}_structure.json"
    materials_json = results_dir / f"{basename}_materials.json"
    
    print("\n" + "=" * 80)
    print(f"PROCESSING: {md_path.name}")
    print(f"Asset Directory: {asset_dir}")
    print("=" * 80)
    
    # 1. Convert to tree structure
    print(f"\n[1/4] Parsing {md_path.name}...")
    step1_cmd = [sys.executable, "markdown_v3.py", str(md_path), "--output", str(tree_json)]
    if not run_command(step1_cmd):
        print(f"Stopping processing for {md_path.name} due to error in Step 1.")
        return False
        
    # 2. Run extraction
    print(f"\n[2/4] Extracting data from {md_path.name}...")
    step2_cmd = [
        sys.executable, "run_extraction_md.py", 
        str(tree_json), 
        "--asset_dir", str(asset_dir),
        "--output", str(materials_json),
        "--model", model,
        "--batch-size", str(batch_size)
    ]
    if not run_command(step2_cmd):
        print(f"Warning: Extraction step for {md_path.name} had issues.")
        
    # 3. Visualize tree
    print(f"\n[3/4] Generating tree visualization for {md_path.name}...")
    step3_cmd = [sys.executable, "visualize_markdown_v3_fixed.py", str(tree_json)]
    run_command(step3_cmd)
    
    # 4. Visualize materials
    print(f"\n[4/4] Generating materials visualization for {md_path.name}...")
    if materials_json.exists():
        step4_cmd = [sys.executable, "visualize_materials.py", str(materials_json)]
        run_command(step4_cmd)
    else:
        print(f"Skipping Step 4 for {md_path.name}: Materials JSON not found.")
        
    return True

def main():
    parser = argparse.ArgumentParser(description='Ionic Conductivity Extraction Pipeline')
    parser.add_argument('input_path', help='Path to the input markdown file or directory containing markdowns')
    parser.add_argument('--asset_dir', help='Explicitly set asset directory (only applies to single file mode)')
    parser.add_argument('--model', default='gemini-2.5-flash-lite', help='LLM model for extraction')
    parser.add_argument('--batch-size', type=int, default=7, help='Batch size for API calls')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: Path not found: {input_path}")
        sys.exit(1)
        
    # Ensure results directory exists
    results_dir = Path("./parsed_downselectedpapers")
    results_dir.mkdir(exist_ok=True)
    
    # Collect files to process
    files_to_process = []
    if input_path.is_file():
        if input_path.suffix.lower() == '.md':
            files_to_process.append(input_path)
        else:
            print(f"Error: {input_path} is not a markdown file.")
            sys.exit(1)
    else:
        # Recursively find all markdown files
        print(f"Searching for markdown files in: {input_path}")
        files_to_process = sorted(list(input_path.rglob("*.md")))
        print(f"Found {len(files_to_process)} markdown files.")

    if not files_to_process:
        print("No markdown files found to process.")
        return

    # Process each file
    success_count = 0
    for md_file in files_to_process:
        if process_file(md_file, args.asset_dir if input_path.is_file() else None, 
                       args.model, args.batch_size, results_dir):
            success_count += 1
            
    print("\n" + "=" * 80)
    print(f"BATCH PROCESSING COMPLETE: {success_count}/{len(files_to_process)} files processed successfully.")
    print("=" * 80)
    print(f"All results are in: {results_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()
