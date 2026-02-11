
import asyncio
import os
import sys
from scifigure_parser import SciFigureParser, FigureAnalysis

from dotenv import load_dotenv

load_dotenv()

# Mock API Key if not present (the script will fail at the API call, but we want to see the schema construction)
if not os.environ.get("GEMINI_API_KEY"):
    print("Warning: GEMINI_API_KEY not set. API calls will fail.")

def check_schema_for_additional_properties():
    print("Checking Pydantic generated schema for FigureAnalysis...")
    schema = FigureAnalysis.model_json_schema()
    import json
    print(json.dumps(schema, indent=2))
    
    if "additionalProperties" in str(schema):
        print("\nFAILURE: 'additionalProperties' found in schema string representation.")
        return True
    else:
         # Pydantic v2 might nest it, so a simple string check is a good heuristic
        print("\nSUCCESS: 'additionalProperties' not found (or at least not obvious).")
        return False

async def run_reproduction():
    # Attempt to actually call the API with the schema if a key is present
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Skipping API call reproduction due to missing API key.")
        return

    parser = SciFigureParser(api_key=api_key)
    
    # We need a dummy image
    img_path = "dummy_test_image.jpg"
    from PIL import Image
    if not os.path.exists(img_path):
        img = Image.new('RGB', (100, 100), color = 'white')
        img.save(img_path)
    
    print(f"\nAttempting to call detect_subplot_async with {img_path}...")
    try:
        await parser.detect_subplot_async(img_path, "test query")
        print("Call successful (unexpected if bug exists).")
    except Exception as e:
        print(f"\nCaught expected exception: {e}")
        if "additionalProperties" in str(e):
             print("CONFIRMED: Error message contains 'additionalProperties'.")
        else:
             print("WARNING: Error message does NOT contain 'additionalProperties'.")

if __name__ == "__main__":
    has_issue = check_schema_for_additional_properties()
    
    # Run the async test
    try:
        asyncio.run(run_reproduction())
    except Exception as e:
        print(f"Runtime error: {e}")
