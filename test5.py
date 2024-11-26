import requests
import json
import numpy as np
import sounddevice as sd
import generator_model as gen

# Load secret
hf_secret: str = gen.get_secret(r'D:\Documents\Secrets\huggingface_secret.txt')

# Model and API details
model_id = "suno/bark-small"
api_url = f"https://api-inference.huggingface.co/models/{model_id}"

# Comprehensive headers
headers = {
    "Authorization": f"Bearer {hf_secret}",
    "Content-Type": "application/json",
    "Accept": "*/*",
    "Connection": "keep-alive"
}

# Detailed payload
payload = {
    "inputs": "My name is Karl Popper. The philosopher of epistemology.",
    "parameters": {
        "text_temp": 0.7,  # Text generation temperature
        "waveform_temp": 0.7,  # Audio generation temperature
        "max_new_tokens": 256  # Limit on new tokens
    }
}


def detailed_error_handling():
    try:
        # Verbose request with full details
        print("Sending request to:", api_url)
        print("Headers:", json.dumps(headers, indent=2))
        print("Payload:", json.dumps(payload, indent=2))

        # Make the request
        response = requests.post(
            api_url,
            headers=headers,
            data=json.dumps(payload)
        )

        # Detailed response logging
        print("\nResponse Status Code:", response.status_code)
        print("Response Headers:", response.headers)

        # Handle different possible response scenarios
        if response.status_code == 200:
            # Successfully received audio
            audio_data = response.content
            audio_array = np.frombuffer(audio_data, dtype=np.float32)

            print(f"Received audio data. Length: {len(audio_array)} samples")

            # Play audio
            sd.play(audio_array, samplerate=24000)
            sd.wait()

        elif response.status_code == 503:
            print("Model is loading. This can take a few minutes for the first request.")
            print("Suggested actions:")
            print("1. Wait a few minutes and retry")
            print("2. Check Hugging Face model page for any known issues")

        else:
            # Detailed error information
            print("\nFull Error Details:")
            print("Status Code:", response.status_code)
            print("Response Content:", response.text)
            print("Response Headers:", response.headers)

    except requests.exceptions.RequestException as req_error:
        print(f"Request Error: {req_error}")
    except Exception as e:
        print(f"Unexpected Error: {e}")


# Run the detailed error handling
detailed_error_handling()
