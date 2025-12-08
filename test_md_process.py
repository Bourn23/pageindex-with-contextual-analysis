#!/usr/bin/env python3
import asyncio
import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(message)s',
    datefmt='%H:%M:%S'
)

async def test():
    print('Starting markdown processing test...')
    
    from pageindex.llm_client import get_llm_client
    from pageindex.markdown_processor_v2 import process_markdown_v2
    
    md_file = 'tests/markdowns/New Insights into the Compositional Dependence of Li-Ion Transport in polymer-ceramic composite electrolytes/New Insights into the Compositional Dependence of Li-Ion Transport in polymer-ceramic composite electrolytes.md'
    
    print('Initializing LLM client...')
    llm_client = get_llm_client(model='gemini-2.5-flash-lite')
    print(f'LLM client initialized: {llm_client.provider}')
    
    print('Starting process_markdown_v2...')
    try:
        result = await asyncio.wait_for(
            process_markdown_v2(
                md_file,
                granularity='coarse',  # Start with coarse to test parsing only
                llm_client=llm_client,
                keyword_level='fine'
            ),
            timeout=30.0
        )
        print(f'Success! Got {len(result.get("structure", []))} top-level nodes')
    except asyncio.TimeoutError:
        print('TIMEOUT after 30 seconds!')
        sys.exit(1)
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)

asyncio.run(test())
