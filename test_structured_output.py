"""
Simple test to verify Gemini structured output is working correctly.
"""
import os
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List

# Test models
class Ingredient(BaseModel):
    name: str = Field(description="Name of the ingredient.")
    quantity: str = Field(description="Quantity of the ingredient, including units.")

class Recipe(BaseModel):
    recipe_name: str = Field(description="The name of the recipe.")
    ingredients: List[Ingredient]
    instructions: List[str]

# Initialize client
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("ERROR: GEMINI_API_KEY not set")
    exit(1)

client = genai.Client(api_key=api_key)

prompt = """
Please extract the recipe from the following text.
The user wants to make delicious chocolate chip cookies. They need 2 and 1/4 cups of all-purpose flour, 
1 teaspoon of baking soda, 1 cup of unsalted butter (softened), and 2 cups of semisweet chocolate chips.
First, preheat the oven to 375°F. Then, mix the dry ingredients. Finally, stir in the chocolate chips.
"""

print("Testing Gemini structured output...")
print("=" * 70)

try:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=Recipe.model_json_schema(),
        ),
    )
    
    print("Raw response text:")
    print(response.text)
    print("\n" + "=" * 70)
    
    # Parse with Pydantic - no cleanup needed!
    recipe = Recipe.model_validate_json(response.text)
    
    print("\nParsed recipe:")
    print(f"Name: {recipe.recipe_name}")
    print(f"Ingredients: {len(recipe.ingredients)}")
    for ing in recipe.ingredients:
        print(f"  - {ing.quantity} {ing.name}")
    print(f"Instructions: {len(recipe.instructions)} steps")
    
    print("\n" + "=" * 70)
    print("✓ Structured output test PASSED!")
    print("  - Response was valid JSON")
    print("  - Response matched the schema")
    print("  - No cleanup or parsing errors")
    
except Exception as e:
    print(f"\n✗ Test FAILED: {e}")
    import traceback
    traceback.print_exc()
