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

from pageindex.markdown_processor_v2 import process_markdown_v2, MarkdownParser

# Mock LLM Client
class MockLLMClient:
    def __init__(self):
        self.provider = 'mock'
    
    def chat_completion(self, **kwargs):
        return "Mock summary"

async def test_markdown_v2():
    print("Testing Markdown Processor V2...")
    
    # Create sample markdown
    sample_md = """
# Main Title

Introduction text here.

## Section 1

This is section 1.

![Figure 1: A sample chart](http://example.com/chart.png)

### Subsection 1.1

Deep nested content.

| Header 1 | Header 2 |
|----------|----------|
| Row 1    | Data 1   |
| Row 2    | Data 2   |

## Section 2

Conclusion text.
    """.strip()
    
    test_file = "test_sample.md"
    with open(test_file, "w") as f:
        f.write(sample_md)
        
    try:
        # Test 1: Coarse Granularity (Parser only)
        print("\nTest 1: Coarse Granularity")
        result = await process_markdown_v2(test_file, granularity='coarse')
        
        structure = result['structure']
        print(f"Doc Name: {result['doc_name']}")
        print(f"Root Nodes: {len(structure)}")
        
        # Verify structure
        assert len(structure) == 1
        assert structure[0]['title'] == "Main Title"
        assert len(structure[0]['nodes']) == 2 # Section 1, Section 2
        
        sec1 = structure[0]['nodes'][0]
        assert sec1['title'] == "Section 1"
        assert len(sec1['nodes']) == 1 # Subsection 1.1
        
        print("Structure verification passed!")
        
        # Test 2: Granular Features (Mocked)
        print("\nTest 2: Granular Features (Figures/Tables)")
        mock_llm = MockLLMClient()
        
        # We use 'medium' to trigger granular integration
        # But since our mock LLM doesn't return valid JSON for SemanticAnalyzer,
        # SemanticAnalyzer might fail gracefully or return empty.
        # However, Figure/Table extractors should still work as they are regex-based
        # and use LLM only for summaries (which we mocked).
        
        result_granular = await process_markdown_v2(test_file, granularity='medium', llm_client=mock_llm)
        
        # Check for figures/tables in Section 1
        sec1_granular = result_granular['structure'][0]['nodes'][0]
        
        # The extractors add nodes to the 'nodes' list
        # Section 1 has 1 subsection + 1 figure (from text)
        # Wait, my implementation adds them to 'nodes'.
        # Let's check the children of Section 1.
        
        print("Checking children of Section 1:")
        for child in sec1_granular['nodes']:
            print(f" - Type: {child.get('node_type', 'section')}, Title: {child.get('title')}")
            
        # We expect: Subsection 1.1, Figure
        # And Subsection 1.1 should have a Table
        
        found_figure = False
        for child in sec1_granular['nodes']:
            if child.get('node_type') == 'figure':
                found_figure = True
                assert "A sample chart" in child['title']
        
        assert found_figure, "Figure not found in Section 1"
        
        # Check Subsection 1.1 for table
        subsec = [n for n in sec1_granular['nodes'] if n.get('title') == "Subsection 1.1"][0]
        found_table = False
        for child in subsec.get('nodes', []):
            if child.get('node_type') == 'table':
                found_table = True
        
        assert found_table, "Table not found in Subsection 1.1"
        
        print("Granular feature verification passed!")
        
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)

if __name__ == "__main__":
    asyncio.run(test_markdown_v2())
