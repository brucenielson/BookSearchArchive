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
    "suno/bark",
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

# https://discuss.huggingface.co/t/undefined-error-on-inference-api-serverless-for-several-hf-text-to-speech-tasks/79230/3
# https://huggingface.co/playground?modelId=google/gemma-1.1-2b-it
# Use Spaces
# https://huggingface.co/spaces/suno/bark
# Others found here:
# https://huggingface.co/tasks/text-to-speech
# https://huggingface.co/docs/transformers/tasks/text-to-speech
# https://huggingface.co/spaces/mrfakename/E2-F5-TTS (Your own voice!)
# https://huggingface.co/spaces/bnielson/testspace
# https://www.gradio.app/guides/quickstart
# https://huggingface.co/docs/hub/spaces-sdks-gradio
# https://www.gradio.app/main/guides/the-interface-class
# https://www.gradio.app/guides/blocks-and-event-listeners
# https://www.gradio.app/guides/streaming-ai-generated-audio
# https://github.com/huggingface/parler-tts
