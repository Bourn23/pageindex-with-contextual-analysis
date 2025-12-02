import asyncio
import os
import sys
import logging
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pageindex.markdown_processor_v2 import process_markdown_v2
from pageindex.llm_client import LLMClient

async def test_keywords():
    print("Testing Keyword Extraction...")
    
    # Path to the specific example file
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_file_path = os.path.join(
        base_dir, 
        "tests/markdowns/New Insights into the Compositional Dependence of Li-Ion Transport in polymer-ceramic composite electrolytes/New Insights into the Compositional Dependence of Li-Ion Transport in polymer-ceramic composite electrolytes.md"
    )
    
    # Initialize Real LLM Client
    try:
        llm_client = LLMClient(provider='gemini', model='gemini-2.5-flash-lite')
    except Exception as e:
        print(f"Failed to initialize LLM Client: {e}")
        return

    try:
        # Test with 'keywords' granularity
        print(f"\nProcessing file with granularity='keywords'...")
        result = await process_markdown_v2(test_file_path, granularity='keywords', llm_client=llm_client)
        
        structure = result['structure']
        
        # Count keywords
        keyword_count = 0
        
        def traverse(nodes):
            nonlocal keyword_count
            for node in nodes:
                if node.get('node_type') == 'keyword':
                    keyword_count += 1
                
                if 'nodes' in node:
                    traverse(node['nodes'])
        
        traverse(structure)
        
        print(f"\nTotal Keywords Found: {keyword_count}")
        
        if keyword_count > 0:
            print("SUCCESS: Keywords found.")
        else:
            print("FAILURE: No keywords found.")
            
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Suppress noisy logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(test_keywords())
