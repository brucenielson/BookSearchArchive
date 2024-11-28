from pathlib import Path
from huggingface_hub import InferenceClient
import numpy as np
import sounddevice as sd
import requests
import json
import traceback


# Load Hugging Face API secret
hf_secret: str = open(r'D:\Documents\Secrets\huggingface_secret.txt', 'r').read().strip()


def advanced_model_diagnostics(model_id: str, token: str):
    """Perform comprehensive model diagnostics."""
    try:
        # Detailed model information request
        url = f"https://huggingface.co/api/models/{model_id}"
        headers = {"Authorization": f"Bearer {token}"}

        print("\n--- Detailed Model Diagnostics ---")
        model_details_response = requests.get(url, headers=headers)
        print(f"Model Details Status Code: {model_details_response.status_code}")

        if model_details_response.status_code == 200:
            model_info = model_details_response.json()
            print("Model Information:")
            print(json.dumps({
                "id": model_info.get("id"),
                "modelType": model_info.get("modelType"),
                "pipeline_tag": model_info.get("pipeline_tag"),
                "library_name": model_info.get("library_name")
            }, indent=2))

        # Inference API diagnostic
        inference_url = f"https://api-inference.huggingface.co/models/{model_id}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        print("\n--- Inference API Diagnostic ---")
        payloads = [
            {"inputs": "Test input for diagnostics"},
            {
                "inputs": "Test input for diagnostics",
                "parameters": {
                    "text_temp": 0.7,
                    "waveform_temp": 0.7
                }
            }
        ]

        for payload in payloads:
            print(f"\nTesting payload: {json.dumps(payload, indent=2)}")
            try:
                inference_response = requests.post(inference_url, headers=headers, json=payload)
                print(f"Inference Response Status Code: {inference_response.status_code}")
                print(f"Response Headers: {dict(inference_response.headers)}")
                print(f"Response Content: {inference_response.text}")
            except Exception as payload_error:
                print(f"Payload test error: {payload_error}")
    except Exception as e:
        print(f"Diagnostics Error: {e}")
        traceback.print_exc()


def text_to_speech_with_diagnostics(model_id: str, token: str, text: str):
    """Comprehensive text-to-speech generation with extensive error handling."""
    print(f"\n--- Text-to-Speech Attempt for model {model_id} ---")
    try:
        # Try InferenceClient
        client = InferenceClient(api_key=token)
        try:
            audio_response = client.text_to_speech(text, model=model_id)
            if isinstance(audio_response, bytes):
                print("Audio successfully generated via InferenceClient.")
                return audio_response
            else:
                print("Unexpected response type from InferenceClient.")
        except Exception as e:
            print(f"InferenceClient method failed for model {model_id}: {e}")
            traceback.print_exc()

        # Try Direct API Call
        url = f"https://api-inference.huggingface.co/models/{model_id}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        payload = {"inputs": text}
        try:
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                print("Audio successfully generated via Direct API.")
                return response.content
            else:
                print(f"Direct API call failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Direct API method failed for model {model_id}: {e}")
            traceback.print_exc()
    except Exception as e:
        print(f"Text-to-speech failed for model {model_id}: {e}")
        traceback.print_exc()
    return None


def try_tts_models(models, text, token):
    """Loop over models to perform TTS and diagnostics."""
    for model in models:
        print("\n\n====================================")
        model_name = model if model else "Default"
        print(f"Processing Model: {model_name}")
        try:
            if model:
                advanced_model_diagnostics(model, token)
            audio_data = text_to_speech_with_diagnostics(model, token, text)
            if audio_data:
                # Save audio file
                file_name = model.replace("/", "_") if model else "default"
                audio_file = Path(f"{file_name}_test_sentence.flac")
                audio_file.write_bytes(audio_data)
                print(f"Audio successfully saved to {audio_file}")

                # Optional playback
                try:
                    audio_array = np.frombuffer(audio_data, dtype=np.float32)
                    sd.play(audio_array, samplerate=24000)
                    sd.wait()
                except Exception as play_error:
                    print(f"Error during audio playback: {play_error}")
        except Exception as e:
            print(f"Error while processing model {model_name}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    models = [
        None,
        "suno/bark",
        "suno/bark-small",
        "microsoft/speecht5_tts",
    ]
    text = "Hello, welcome to the world of text to speech!"
    try_tts_models(models, text, hf_secret)
