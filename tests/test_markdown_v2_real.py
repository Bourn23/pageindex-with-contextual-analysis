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
    import google.genai
except ImportError:
    sys.modules['google'] = MagicMock()
    sys.modules['google.genai'] = MagicMock()
    sys.modules['google.genai.types'] = MagicMock()

try:
    import dotenv
except ImportError:
    sys.modules['dotenv'] = MagicMock()

try:
    from PIL import Image
except ImportError:
    sys.modules['PIL'] = MagicMock()
    sys.modules['PIL.Image'] = MagicMock()

try:
    import yaml
except ImportError:
    sys.modules['yaml'] = MagicMock()

try:
    import openai
except ImportError:
    sys.modules['openai'] = MagicMock()

try:
    import pydantic
except ImportError:
    mock_pydantic = MagicMock()
    mock_pydantic.BaseModel = MagicMock
    mock_pydantic.Field = MagicMock
    sys.modules['pydantic'] = mock_pydantic

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pageindex.markdown_processor_v2 import process_markdown_v2

# Mock LLM Client
class MockLLMClient:
    def __init__(self):
        self.provider = 'mock'
    
    def chat_completion(self, **kwargs):
        return "Mock summary"

async def test_markdown_v2_real_file():
    print("Testing Markdown Processor V2 with Real File...")
    
    # Path to the specific example file
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_file_path = os.path.join(
        base_dir, 
        "tests/markdowns/New Insights into the Compositional Dependence of Li-Ion Transport in polymer-ceramic composite electrolytes/New Insights into the Compositional Dependence of Li-Ion Transport in polymer-ceramic composite electrolytes.md"
    )
    
    if not os.path.exists(test_file_path):
        print(f"Error: Test file not found at {test_file_path}")
        return

    try:
        # Test: Granular Features with Real File
        print(f"\nProcessing file: {os.path.basename(test_file_path)}")
        mock_llm = MockLLMClient()
        
        result = await process_markdown_v2(test_file_path, granularity='medium', llm_client=mock_llm)
        
        structure = result['structure']
        print(f"Doc Name: {result['doc_name']}")
        print(f"Root Nodes: {len(structure)}")
        
        # Verify we found some figures
        found_figures = 0
        found_tables = 0
        
        def traverse(nodes):
            nonlocal found_figures, found_tables
            for node in nodes:
                if node.get('node_type') == 'figure':
                    found_figures += 1
                    print(f"Found Figure: {node['title']}")
                    # Check if image path was resolved
                    if 'metadata' in node:
                        print(f"  - Path: {node['metadata'].get('full_path')}")
                        print(f"  - Exists: {node['metadata'].get('exists')}")
                elif node.get('node_type') == 'table':
                    found_tables += 1
                
                if 'nodes' in node:
                    traverse(node['nodes'])
        
        traverse(structure)
        
        print(f"\nTotal Figures Found: {found_figures}")
        print(f"Total Tables Found: {found_tables}")
        
        assert found_figures > 0, "No figures found in the real file!"
        
        print("\nVerification passed!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_markdown_v2_real_file())
