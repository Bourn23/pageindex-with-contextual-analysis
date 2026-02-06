import os
from google import genai

def list_models():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY")
    client = genai.Client(api_key=api_key)
    for model in client.models.list():
        print(model.name)

if __name__ == "__main__":
    list_models()
