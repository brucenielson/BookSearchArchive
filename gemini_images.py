from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import base64
from generator_model import get_secret
import google.generativeai as genai2

secret: str = get_secret(r'D:\Documents\Secrets\gemini_secret.txt')
genai2.configure(api_key=secret)

client = genai.Client(api_key=secret)

contents = ('Hi, can you create a 3d rendered image of a pig '
            'with wings and a top hat flying over a happy '
            'futuristic scifi city with lots of greenery?')

response = client.models.generate_content(
    model="gemini-2.0-flash-exp-image-generation",
    contents=contents,
    config=types.GenerateContentConfig(
      response_modalities=['TEXT', 'IMAGE']
    )
)

for part in response.candidates[0].content.parts:
    if part.text is not None:
        print(part.text)
    elif part.inline_data is not None:
        image = Image.open(BytesIO((part.inline_data.data)))
        image.save('gemini-native-image.png')
        image.show()
