import os
import google.genai as genai
from google.genai import types
from dotenv import load_dotenv
import base64

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"), http_options={'api_version': 'v1alpha'})
model = 'gemini-3-flash-preview'
# model = 'gemini-2.5-flash'
prompt_text = "Tell a five sentence description of the image."
raw_parts = [types.Part(text=prompt_text)]

# also add an image
img_dir = './fetched_papers/obelix_md/Effect of Si substitution on the structural and transport properties of superionic Li-argyrodites/_page_5_Figure_5.jpeg'
# read them as bytes
with open(img_dir, 'rb') as f:
    image_bytes = f.read()
image_b64 = base64.b64encode(image_bytes).decode('utf-8')
mime_type = "image/png" if img_dir.endswith('.png') else "image/jpeg"

image_part = types.Part(
                    inline_data=types.Blob(
                        mime_type=mime_type,
                        data=image_b64
                    ),
                    # Ensure your self.client was initialized with api_version='v1alpha'
                    media_resolution={"level": "media_resolution_high"}
                )

parts = raw_parts + [image_part]
contents = [
                types.Content(
                    parts = parts
                )
            ]
config = types.GenerateContentConfig(
    temperature=1.0,
    response_mime_type="text/plain",
)

response = client.models.generate_content(model=model, contents=contents, config=config)

print(response)
# Access token counts
usage = response.usage_metadata
input_tokens = usage.prompt_token_count
output_tokens = usage.candidates_token_count
thought_tokens = usage.thoughts_token_count
total_tokens = usage.total_token_count
print('length of token details: ', len(usage.prompt_tokens_details))

print(f"Input: {input_tokens}, Output: {output_tokens}, Thoughts: {thought_tokens}, Total: {total_tokens}")