"""
Fetch scientific papers by DOI.
usage: python fetch_paper.py "10.1021/acs.chemmater.0c04150" --email "mail@gmail.com"
"""

import sys
import os
import argparse
from pathlib import Path
import json

# Placeholder for user's suggested libraries
try:
    from unpywall import Unpywall
    from unpywall.utils import UnpywallCredentials
except ImportError:
    Unpywall = None

try:
    import PyPaperBot
except ImportError:
    PyPaperBot = None

def fetch_with_unpywall(doi, email, output_dir):
    if not Unpywall:
        print("[-] unpywall not installed.")
        return False
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"[*] Trying unpywall for DOI: {doi}...")
    try:
        UnpywallCredentials(email)
        # Check if OA version exists
        df = Unpywall.doi(dois=[doi])
        if df.empty or not df.iloc[0].get('is_oa'):
            print(f"[-] No Open Access version found via unpywall for {doi}")
            return False
            
        filename = f"{doi.replace('/', '_')}.pdf"
        filepath = output_dir / filename
        Unpywall.download_pdf_file(doi=doi, filename=str(filepath))
        print(f"[++] Successfully downloaded PDF to: {filepath}")
        return True
    except Exception as e:
        print(f"[!] unpywall error: {e}")
        return False

def fetch_with_pypaperbot(doi, output_dir):
    print(f"[*] Trying PyPaperBot for DOI: {doi}...")
    import subprocess
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Run PyPaperBot as a subprocess to keep it simple
        cmd = [
            sys.executable, "-m", "PyPaperBot",
            f"--doi={doi}",
            f"--dwn-dir={output_dir}"
        ]
        subprocess.run(cmd, check=True)
        print(f"[++] PyPaperBot finished processing {doi}")
        return True
    except Exception as e:
        print(f"[!] PyPaperBot error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Fetch scientific papers by DOI.")
    parser.add_argument("doi", help="The DOI to fetch")
    parser.add_argument("--email", default="user@example.com", help="Email for unpywall (required)")
    parser.add_argument("--output", default="fetched_papers", help="Output directory")
    parser.add_argument("--method", choices=["unpywall", "pypaperbot", "both"], default="both", help="Fetch method")
    
    args = parser.parse_args()
    
    success = False
    if args.method in ["unpywall", "both"]:
        success = fetch_with_unpywall(args.doi, args.email, args.output)
        
    if not success and args.method in ["pypaperbot", "both"]:
        success = fetch_with_pypaperbot(args.doi, args.output)
        
    if success:
        print(f"[***] Search complete for {args.doi}")
    else:
        print(f"[!!!] Failed to fetch {args.doi} with available methods.")

if __name__ == "__main__":
    main()
