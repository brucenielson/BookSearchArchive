from huggingface_hub import InferenceClient
import requests
from pathlib import Path


# Load Hugging Face API secret - Put the secret in a text file and read it
hf_secret = open(r'D:\Documents\Secrets\huggingface_secret.txt', 'r').read().strip()


def get_model_details(model_id: str, token: str):
    """Fetch model details using the Hugging Face API."""
    url = f"https://huggingface.co/api/models/{model_id}"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.get(url, headers=headers)
        print(f"\n>>> Retrieving details for model: {model_id}")
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            model_info = response.json()
            return {
                "id": model_info.get("id"),
                "modelType": model_info.get("modelType"),
                "pipeline_tag": model_info.get("pipeline_tag"),
                "library_name": model_info.get("library_name"),
            }
        else:
            print(f"Failed to retrieve details for {model_id}.")
            return None
    except Exception as e:
        print(f"Error fetching model details for {model_id}: {e}")
        return None


def generate_audio(model_id: str, token: str, text: str):
    """Generate audio using InferenceClient."""
    client = InferenceClient(api_key=token)
    try:
        print(f"\n>>> Generating audio with model: {model_id}")
        audio_data = client.text_to_speech(text, model=model_id)
        if isinstance(audio_data, bytes):
            file_name = model_id.replace("/", "_")
            audio_file = Path(f"{file_name}_test_sentence.flac")
            audio_file.write_bytes(audio_data)
            print(f"Audio saved to {audio_file}")
            return True
        else:
            print(f"Unexpected response type from {model_id}.")
            return False
    except Exception as e:
        print(f"Error generating audio for {model_id}: {e}")
        return False


def try_models(models, text, token):
    """Test models and generate a summary report."""
    results = {}
    for model in models:
        print("\n" + "=" * 80)
        print(f"Processing model: {model}")
        print("=" * 80)

        model_details = get_model_details(model, token)
        if model_details:
            print("Model Details:")
            print(model_details)
        else:
            print(f"Skipping {model} due to missing details.")

        success = generate_audio(model, token, text)
        results[model] = "Success" if success else "Failed"

    print("\n" + "=" * 80)
    print("SUMMARY REPORT")
    print("=" * 80)
    for model, result in results.items():
        print(f"Model: {model}\n  Result: {result}")
    print("=" * 80)


if __name__ == "__main__":
    models = [
        "suno/bark",
        "suno/bark-small",
        "facebook/mms-tts-eng",
        "microsoft/speecht5_tts",
    ]
    text = "Hello, welcome to the world of text to speech!"
    try_models(models, text, hf_secret)


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
