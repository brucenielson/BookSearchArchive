from pathlib import Path
from huggingface_hub import InferenceClient
import generator_model as gen


# Load secret
hf_secret: str = gen.get_secret(r'D:\Documents\Secrets\huggingface_secret.txt')

# Initialize the InferenceClient
client = InferenceClient(api_key=hf_secret)

# Test sentence to convert to speech
text = "Hello, welcome to the world of text to speech!"

# Specify the model to use (smaller version)
model = "suno/bark-small"

# Generate audio from the text using the specified model
audio = client.text_to_speech(text) #, model=model)

# Save the audio to a file
audio_file = Path("test_sentence.flac")
audio_file.write_bytes(audio)

print(f"Audio saved to {audio_file}")
