from pathlib import Path
from huggingface_hub import InferenceClient

# Load Hugging Face API secret
hf_secret: str = open(r'D:\Documents\Secrets\huggingface_secret.txt', 'r').read().strip()

# Initialize the InferenceClient with the Hugging Face API key
client = InferenceClient(api_key=hf_secret)

# Define the text to convert to speech
text = "Hello, welcome to the world of text to speech!"

# List of models to try for TTS
models = [
    None,
    "suno/bark-small",
    "microsoft/speecht5_tts",
]


# Function to try generating speech for each model
def try_tts_models(models, text):
    for model in models:
        print("\n\n")
        print(f"Trying model: {model}")

        try:
            # Generate audio from the text using the current model
            if model is None:
                audio = client.text_to_speech(text)
            else:
                audio = client.text_to_speech(text, model=model)

            # Save the audio to a file
            audio_file = Path(f"{model.replace('/', '_')}_test_sentence.flac")
            audio_file.write_bytes(audio)
            print(f"Audio successfully saved to {audio_file}")

        except Exception as e:
            # Log the error for the current model
            print(f"Error with model {model}: {str(e)}")
            continue  # Move on to the next model


# Run the function to test TTS models
try_tts_models(models, text)
