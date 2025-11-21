"""
Example: Using PageIndex with Google Gemini

This example shows how to use PageIndex with Gemini instead of OpenAI.
"""
import os
from pageindex import page_index_main, config
from pageindex.llm_client import set_llm_provider

# Method 1: Auto-detection (recommended)
# Just set GEMINI_API_KEY in your .env file and it will auto-detect
print("Method 1: Auto-detection from environment variables")
print("=" * 60)

# Check if Gemini API key is available
if os.getenv("GEMINI_API_KEY"):
    print("✓ GEMINI_API_KEY found - will use Gemini automatically")
    
    # Configure options with Gemini model
    opt = config(
        model='gemini-1.5-flash',  # Fast Gemini model
        toc_check_page_num=20,
        max_page_num_each_node=10,
        max_token_num_each_node=20000,
        if_add_node_id='yes',
        if_add_node_summary='yes',
        if_add_doc_description='no',
        if_add_node_text='no'
    )
    
    # Process a PDF (replace with your actual PDF path)
    # result = page_index_main('your_document.pdf', opt)
    print("Ready to process PDFs with Gemini!")
    
else:
    print("✗ GEMINI_API_KEY not found in environment")
    print("  Add it to your .env file:")
    print("  GEMINI_API_KEY=your_api_key_here")

print()

# Method 2: Explicit provider selection
print("Method 2: Explicit provider selection")
print("=" * 60)

# Manually set provider to Gemini
# set_llm_provider('gemini')  # Uses GEMINI_API_KEY from env
# or with explicit API key:
# set_llm_provider('gemini', api_key='your_key_here')

print("You can explicitly set the provider using:")
print("  from pageindex.llm_client import set_llm_provider")
print("  set_llm_provider('gemini')")
print()

# Method 3: Using different Gemini models
print("Method 3: Available Gemini models")
print("=" * 60)
print("- gemini-1.5-flash      (Fast, efficient)")
print("- gemini-1.5-pro        (More capable)")
print("- gemini-2.0-flash-exp  (Experimental)")
print()

# Example configuration for different use cases
print("Example configurations:")
print("-" * 60)

print("\n1. Fast processing (recommended for testing):")
fast_config = config(
    model='gemini-1.5-flash',
    if_add_node_summary='no',  # Skip summaries for speed
    if_add_doc_description='no'
)
print(f"   model: {fast_config.model}")

print("\n2. High quality (recommended for production):")
quality_config = config(
    model='gemini-1.5-pro',
    if_add_node_summary='yes',
    if_add_doc_description='yes'
)
print(f"   model: {quality_config.model}")

print("\n3. Balanced (good default):")
balanced_config = config(
    model='gemini-1.5-flash',
    if_add_node_summary='yes',
    if_add_doc_description='no'
)
print(f"   model: {balanced_config.model}")

print("\n" + "=" * 60)
print("Setup complete! You're ready to use PageIndex with Gemini.")
print("=" * 60)
