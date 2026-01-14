import sys
import os
from habanero import Crossref
import trafilatura
import json
from pathlib import Path

def fetch_paper_text(doi):
    """
    Fetches metadata and primary text for a given DOI.
    Uses Crossref for metadata and Trafilatura for text extraction.
    """
    cr = Crossref()
    print(f"[*] Resolving DOI: {doi}")
    
    try:
        # 1. Get Metadata from Crossref
        res = cr.works(ids=doi)
        metadata = res['message']
        title = metadata.get('title', ['Unknown Title'])[0]
        url = metadata.get('resource', {}).get('primary', {}).get('URL')
        
        if not url:
            # Fallback to doi.org
            url = f"https://doi.org/{doi}"
        
        print(f"[+] Found Title: {title}")
        print(f"[*] Target URL: {url}")
        
        # 2. Fetch and Extract Text
        print(f"[*] Attempting to fetch content from {url}...")
        downloaded = trafilatura.fetch_url(url)
        
        if not downloaded:
            print(f"[-] Could not fetch content from {url}")
            return None
            
        # Extract metadata and text in a structured format
        # Trafilatura can output markdown-like text
        text = trafilatura.extract(downloaded, include_comments=False, include_tables=True, output_format='txt')
        
        if not text:
            print(f"[-] Could not extract meaningful text from the page.")
            return None
            
        return {
            "doi": doi,
            "title": title,
            "url": url,
            "text": text,
            "metadata": metadata
        }
        
    except Exception as e:
        print(f"[!] Error fetching DOI {doi}: {e}")
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_paper_text.py <DOI>")
        # Default to the high-yield paper we found
        doi = "10.1149/1.2086597"
        print(f"No DOI provided. Defaulting to: {doi}")
    else:
        doi = sys.argv[1]
        
    result = fetch_paper_text(doi)
    
    if result:
        # Create output directory
        output_dir = Path("fetched_papers")
        output_dir.mkdir(exist_ok=True)
        
        # Clean title for filename
        safe_title = "".join([c if c.isalnum() else "_" for c in result['title'][:50]])
        filename = output_dir / f"{safe_title}.md"
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"# {result['title']}\n\n")
            f.write(f"**DOI**: {result['doi']}\n")
            f.write(f"**URL**: {result['url']}\n\n")
            f.write("## CONTENT\n\n")
            f.write(result['text'])
            
        print(f"[++] Successfully saved text to: {filename}")
    else:
        print("[-] Fetching failed.")

if __name__ == "__main__":
    main()
