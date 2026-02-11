#!/usr/bin/env python3
"""
Batch Processor for Extraction Pipeline
=======================================

This script manages the batch execution of the extraction pipeline across multiple papers.
It provides:
1. State management (checkpointing) via `batch_status.json`.
2. Resume capability (skips already completed folders).
3. robust error handling and logging.
4. Progress tracking.

Usage:
    python batch_processor.py <parent_directory> [--retry-failed]

"""

import os
import sys
import json
import argparse
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed
    def tqdm(iterable, desc="", unit=""):
        print(f"Processing {len(iterable)} items...")
        return iterable

# Configuration
STATUS_FILE_NAME = "batch_status.json"
PIPELINE_SCRIPT = "./run_pipeline_v5.sh"

def load_status(status_path: Path) -> Dict[str, Any]:
    if status_path.exists():
        try:
            with open(status_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Corrupt status file at {status_path}. Starting fresh.")
            return {}
    return {}

def save_status(status_path: Path, status_data: Dict[str, Any]):
    with open(status_path, 'w') as f:
        json.dump(status_data, f, indent=2)

def run_batch(parent_dir: str, retry_failed: bool = False, pipeline_script: str = "./run_pipeline_v5.sh"):
    parent_path = Path(parent_dir).resolve()
    if not parent_path.exists():
        print(f"Error: Directory not found: {parent_path}")
        sys.exit(1)
        
    pipeline_script_path = Path(pipeline_script).resolve()
    
    # 1. Initialize Status Tracking
    status_path = parent_path / STATUS_FILE_NAME
    status_data = load_status(status_path)
    
    print(f"Loaded status for {len(status_data)} papers.")

    # 2. Discover Papers (Subdirectories)
    # Filter for directories that look like valid paper folders (contain .md? or just any dir?)
    # The shell script checked for .md files, we should do the same to be safe.
    all_subdirs = [d for d in parent_path.iterdir() if d.is_dir()]
    valid_papers = []
    
    print("Scanning directories...")
    for d in all_subdirs:
        # Check if it contains an MD file
        md_files = list(d.glob("*.md"))
        if md_files:
            valid_papers.append(d)
        else:
            # print(f"Skipping {d.name} (No .md file)")
            pass

    print(f"Found {len(valid_papers)} valid paper directories.\n")

    # 3. Processing Loop
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    success_count = 0
    
    # Sort for consistent order
    valid_papers.sort(key=lambda p: p.name)
    
    if not valid_papers:
        print("No valid papers found (must contain .md files). Exiting.")
        return

    pbar = tqdm(valid_papers, desc="Processing Papers", unit="paper")
    
    for paper_dir in pbar:
        paper_name = paper_dir.name
        
        # Check existing status
        current_status = status_data.get(paper_name, {}).get("status")
        
        if current_status == "completed":
            skipped_count += 1
            # pbar.set_postfix(status="Skip", paper=paper_name[:20])
            continue
            
        if current_status == "failed" and not retry_failed:
             skipped_count += 1
             # pbar.set_postfix(status="SkipFail", paper=paper_name[:20])
             continue

        # Prepare to run
        # pbar.set_postfix(status="Running", paper=paper_name[:20])
        
        start_time = datetime.now().isoformat()
        
        # Execute Pipeline
        # We call the shell script with the absolute path of the paper directory
        cmd = [str(pipeline_script_path), str(paper_dir)]
        
        try:
            # Run existing shell script
            # We capture output to avoid cluttering the progress bar, 
            # but maybe we should redirect it to a per-paper log file?
            # The shell script already logs to a file in RESULTS_DIR, so we can just capture stdout/stderr here to keep the UI clean.
            result = subprocess.run(
                cmd,
                check=False,
                start_new_session=True, # Detach somewhat
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            end_time = datetime.now().isoformat()
            
            # Determine Success/Failure
            # The shell script returns exit code 0 on success, non-zero on failure.
            if result.returncode == 0:
                new_status = "completed"
                success_count += 1
            else:
                new_status = "failed"
                failed_count += 1
                
            # Log output to status (optional: can be large, maybe just save the last few lines or path to log)
            # For compactness, we'll store the exit code and maybe the last error line if failed.
            
            status_entry = {
                "status": new_status,
                "last_run": end_time,
                "duration_seconds": (datetime.fromisoformat(end_time) - datetime.fromisoformat(start_time)).total_seconds(),
                "exit_code": result.returncode,
                "log_file": f"See {paper_dir}/...log" # We don't know the exact log name from here easily without parsing
            }
            
            if new_status == "failed":
                # Try to capture the last error message
                lines = result.stdout.strip().split('\n')
                last_lines = lines[-5:] if lines else []
                status_entry["error_preview"] = "\n".join(last_lines)
                
            status_data[paper_name] = status_entry
            save_status(status_path, status_data) # Save immediately
            
            processed_count += 1
            
        except Exception as e:
            # System error (e.g. script not found)
            print(f"\nCritical Error running {paper_name}: {e}")
            status_data[paper_name] = {
                "status": "system_error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            save_status(status_path, status_data)
            failed_count += 1

    # 4. Final Report
    print("\n" + "="*40)
    print("BATCH PROCESSING COMPLETE")
    print("="*40)
    print(f"Total Papers: {len(valid_papers)}")
    print(f"Processed:    {processed_count}")
    print(f"Success:      {success_count}")
    print(f"Failed:       {failed_count}")
    print(f"Skipped:      {skipped_count}")
    print("="*40)
    print(f"Status saved to: {status_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process papers for extraction.")
    parser.add_argument("parent_dir", help="Path to the parent directory containing paper subfolders.")
    parser.add_argument("--retry-failed", action="store_true", help="Retry papers that are marked as failed.")
    parser.add_argument("--pipeline-script", default="./run_pipeline_v5.sh", help="Path to the pipeline script to run.")
    
    args = parser.parse_args()
    
    run_batch(args.parent_dir, args.retry_failed, args.pipeline_script)
