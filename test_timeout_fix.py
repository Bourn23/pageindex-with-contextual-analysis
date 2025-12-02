#!/usr/bin/env python3
"""
Quick test to verify timeout and progress indicators are working.
"""

import asyncio
import sys
from pathlib import Path

# Add PageIndex to path
sys.path.insert(0, str(Path(__file__).parent))

from pageindex.markdown_processor_v2 import process_markdown_v2
from pageindex.llm_client import get_llm_client

async def main():
    md_file = "tests/markdowns/New Insights into the Compositional Dependence of Li-Ion Transport in polymer-ceramic composite electrolytes/New Insights into the Compositional Dependence of Li-Ion Transport in polymer-ceramic composite electrolytes.md"
    
    print("Testing timeout fix...")
    print(f"Processing: {md_file}")
    print()
    
    # Initialize LLM client
    llm_client = get_llm_client(provider='gemini', model='gemini-2.5-flash-lite')
    
    # Process with keywords granularity
    result = await process_markdown_v2(
        md_file,
        granularity='keywords',
        llm_client=llm_client
    )
    
    print("\n✓ Processing completed!")
    print(f"  Sections: {len(result.get('structure', []))}")
    
    return 0

if __name__ == '__main__':
    exit(asyncio.run(main()))
