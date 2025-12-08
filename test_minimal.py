#!/usr/bin/env python3
import asyncio
import logging

# Setup logging FIRST
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(message)s',
    datefmt='%H:%M:%S'
)

async def main():
    print('=== Starting test ===')
    
    print('1. Importing...')
    from pageindex.llm_client import get_llm_client
    from pageindex.markdown_processor_v2 import process_markdown_v2
    
    print('2. Getting LLM client...')
    llm_client = get_llm_client(model='gemini-2.5-flash-lite')
    print(f'   LLM client: {llm_client.provider}')
    
    print('3. Processing markdown (coarse mode)...')
    md_file = 'tests/markdowns/New Insights into the Compositional Dependence of Li-Ion Transport in polymer-ceramic composite electrolytes/New Insights into the Compositional Dependence of Li-Ion Transport in polymer-ceramic composite electrolytes.md'
    
    result = await process_markdown_v2(
        md_file,
        granularity='coarse',
        llm_client=llm_client,
        keyword_level='fine'
    )
    
    print(f'4. Success! Got {len(result.get("structure", []))} sections')
    return 0

if __name__ == '__main__':
    exit(asyncio.run(main()))
