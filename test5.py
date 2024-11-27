from huggingface_hub import InferenceClient
import numpy as np
import sounddevice as sd
import generator_model as gen
import requests
import json
import traceback

# Load secret
hf_secret: str = gen.get_secret(r'D:\Documents\Secrets\huggingface_secret.txt')


def advanced_model_diagnostics(model_id: str, token: str):
    """Perform comprehensive model diagnostics"""
    try:
        # Detailed model information request
        url = f"https://huggingface.co/api/models/{model_id}"
        # url = f"https://huggingface.co/spaces/suno/bark"
        headers = {"Authorization": f"Bearer {token}"}

        print("\n--- Detailed Model Diagnostics ---")

        # Model details
        model_details_response = requests.get(url, headers=headers)
        print("Model Details Response:")
        print(f"Status Code: {model_details_response.status_code}")

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

        # Try multiple payload approaches
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
                print(f"Inference API Response:")
                print(f"Status Code: {inference_response.status_code}")
                print(f"Response Headers: {dict(inference_response.headers)}")
                print(f"Response Content: {inference_response.text}")
            except Exception as payload_error:
                print(f"Payload test error: {payload_error}")

    except Exception as e:
        print(f"Diagnostics Error: {e}")
        traceback.print_exc()


def text_to_speech_with_diagnostics(model_id: str, token: str, text: str):
    """Comprehensive text-to-speech generation with extensive error handling"""
    try:
        # Create client with verbose error tracking
        print("\n--- Text-to-Speech Attempt ---")
        print(f"Attempting to generate speech for: {text}")

        # Try multiple inference approaches
        inference_methods = [
            # Method 1: Direct InferenceClient
            lambda: _try_inference_client(model_id, token, text),

            # Method 2: Direct API Call
            lambda: _try_direct_api_call(model_id, token, text)
        ]

        for method in inference_methods:
            try:
                audio_data = method()
                if audio_data:
                    return audio_data
            except Exception as method_error:
                print(f"Method failed: {method_error}")
                traceback.print_exc()

        print("All inference methods failed.")
        return None

    except Exception as main_error:
        print(f"Comprehensive Error in text-to-speech generation:")
        print(f"  - Error Type: {type(main_error)}")
        print(f"  - Error Message: {main_error}")
        traceback.print_exc()

    return None


def _try_inference_client(model_id: str, token: str, text: str):
    """Attempt text-to-speech using InferenceClient"""
    print("\nAttempting Method: InferenceClient")
    client = InferenceClient(model=model_id, token=token)

    try:
        # Try different parameters
        audio_response = client.text_to_speech(text)

        if not isinstance(audio_response, bytes):
            print("Unexpected response type")
            return None

        print("Successful audio generation via InferenceClient!")
        return audio_response

    except Exception as e:
        print(f"InferenceClient method failed: {e}")
        traceback.print_exc()
        return None


def _try_direct_api_call(model_id: str, token: str, text: str):
    """Attempt text-to-speech using direct API call"""
    print("\nAttempting Method: Direct API Call")
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    # Try multiple payload configurations
    payloads = [
        {"inputs": text},
        {
            "inputs": text,
            "parameters": {
                "text_temp": 0.7,
                "waveform_temp": 0.7
            }
        }
    ]

    for payload in payloads:
        try:
            response = requests.post(url, headers=headers, json=payload)

            print(f"Direct API Response:")
            print(f"Status Code: {response.status_code}")
            print(f"Response Headers: {dict(response.headers)}")
            print(f"Response Content Length: {len(response.content)} bytes")

            if response.status_code == 200:
                print("Successful audio generation via Direct API!")
                return response.content
            else:
                print(f"API call failed: {response.text}")

        except Exception as call_error:
            print(f"Direct API call error: {call_error}")
            traceback.print_exc()

    return None


def main():
    model_id = "suno/bark-small"

    # Perform advanced diagnostics
    advanced_model_diagnostics(model_id, hf_secret)

    # Attempt text-to-speech with full diagnostics
    audio_data = text_to_speech_with_diagnostics(
        model_id,
        hf_secret,
        "My name is Karl Popper. The philosopher of epistemology."
    )

    # Optional: Save and play audio if generated
    if audio_data:
        try:
            with open("diagnostic_output.wav", "wb") as f:
                f.write(audio_data)
            print("\nAudio saved to diagnostic_output.wav")

            # Optional audio playback
            audio_array = np.frombuffer(audio_data, dtype=np.float32)
            sd.play(audio_array, samplerate=24000)
            sd.wait()
        except Exception as save_error:
            print(f"Error saving/playing audio: {save_error}")


if __name__ == "__main__":
    main()
