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

# Simplified headers
headers = {
    "Authorization": f"Bearer {hf_secret}",
    "Content-Type": "application/json"
}

# Simplified payload
payload = {
    "inputs": "My name is Karl Popper. The philosopher of epistemology."
}


def inference_request():
    try:
        # Make the request
        print("Sending request to:", api_url)
        print("Payload:", json.dumps(payload, indent=2))

        response = requests.post(
            api_url,
            headers=headers,
            data=json.dumps(payload)
        )

        # Detailed logging
        print("\nResponse Status Code:", response.status_code)

        if response.status_code == 200:
            # Save audio to file for troubleshooting
            with open("output_audio.wav", "wb") as audio_file:
                audio_file.write(response.content)

            print("Audio saved to output_audio.wav")

            # Attempt to play audio
            try:
                audio_array = np.frombuffer(response.content, dtype=np.float32)
                print(f"Received audio data. Length: {len(audio_array)} samples")

                sd.play(audio_array, samplerate=24000)
                sd.wait()
            except Exception as play_error:
                print(f"Audio playback error: {play_error}")

        else:
            # Detailed error information
            print("\nFull Error Details:")
            print("Status Code:", response.status_code)
            print("Response Headers:", response.headers)
            print("Response Content:", response.text)

    except requests.exceptions.RequestException as req_error:
        print(f"Request Error: {req_error}")
    except Exception as e:
        print(f"Unexpected Error: {e}")


# Run the inference request
inference_request()