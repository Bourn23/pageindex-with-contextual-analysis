import asyncio
import os
import sys
import json
import logging
from unittest.mock import MagicMock

# Mock dependencies if missing
try:
    import tiktoken
except ImportError:
    mock_tiktoken = MagicMock()
    mock_tiktoken.encoding_for_model.return_value.encode.return_value = [1, 2, 3] # Mock tokens
    sys.modules['tiktoken'] = mock_tiktoken

try:
    import pymupdf
except ImportError:
    sys.modules['pymupdf'] = MagicMock()

try:
    import PyPDF2
except ImportError:
    sys.modules['PyPDF2'] = MagicMock()

try:
    import dotenv
except ImportError:
    sys.modules['dotenv'] = MagicMock()

try:
    import yaml
except ImportError:
    sys.modules['yaml'] = MagicMock()

try:
    import google.genai
except ImportError:
    print("Warning: google-genai not found")

try:
    from PIL import Image
except ImportError:
    sys.modules['PIL'] = MagicMock()
    sys.modules['PIL.Image'] = MagicMock()

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pageindex.markdown_processor_v2 import process_markdown_v2
from pageindex.llm_client import LLMClient

async def test_markdown_v2_real_llm():
    print("Testing Markdown Processor V2 with Real LLM (Gemini)...")
    
    # Path to the specific example file
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_file_path = os.path.join(
        base_dir, 
        "tests/markdowns/New Insights into the Compositional Dependence of Li-Ion Transport in polymer-ceramic composite electrolytes/New Insights into the Compositional Dependence of Li-Ion Transport in polymer-ceramic composite electrolytes.md"
    )
    
    if not os.path.exists(test_file_path):
        print(f"Error: Test file not found at {test_file_path}")
        return

    # Initialize Real LLM Client
    try:
        # Explicitly use gemini provider
        llm_client = LLMClient(provider='gemini', model='gemini-2.5-flash-lite')
        print(f"LLM Client initialized: Provider={llm_client.provider}")
    except Exception as e:
        print(f"Failed to initialize LLM Client: {e}")
        return

    try:
        # Test: Granular Features with Real LLM
        # We use 'medium' to trigger semantic subdivision
        print(f"\nProcessing file: {os.path.basename(test_file_path)}")
        
        result = await process_markdown_v2(test_file_path, granularity='medium', llm_client=llm_client)
        
        structure = result['structure']
        print(f"Doc Name: {result['doc_name']}")
        print(f"Root Nodes: {len(structure)}")
        
        # Verify semantic units were created
        found_semantic_units = 0
        
        def traverse(nodes):
            nonlocal found_semantic_units
            for node in nodes:
                if node.get('node_type') == 'semantic_unit':
                    found_semantic_units += 1
                    print(f"Found Semantic Unit: {node['title']}")
                    # print(f"  - Summary: {node.get('summary', '')[:50]}...")
                
                if 'nodes' in node:
                    traverse(node['nodes'])
        
        traverse(structure)
        
        print(f"\nTotal Semantic Units Found: {found_semantic_units}")
        
        if found_semantic_units > 0:
            print("\nVerification passed! Semantic analysis is working.")
        else:
            print("\nWarning: No semantic units found. This might be due to short sections or LLM response.")
            
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.DEBUG)
    # Filter noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    asyncio.run(test_markdown_v2_real_llm())
