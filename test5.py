from huggingface_hub import InferenceClient
import sounddevice as sd
import numpy as np
import generator_model as gen
import requests

hf_secret: str = gen.get_secret(r'D:\Documents\Secrets\huggingface_secret.txt')  # Put your path here

# Set up the API inference endpoint
model_id = "suno/bark-small"

# Initialize the Hugging Face client
client = InferenceClient(token=hf_secret)


# Function to check model health
def check_model_health(model_id: str):
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {hf_secret}"}

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            print(f"Model {model_id} is available!")
        else:
            print(f"Model {model_id} returned status code {response.status_code}: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Error checking model health: {e}")


# Check the model health before making inference requests
check_model_health(model_id)

# Loop through voice presets
for i in range(2):
    voice_preset = f"v2/de_speaker_{i}"
    print(f"Sending request with payload: {{'text': 'My name is Karl Popper. The philosopher of epistemology.'}}")

    # Payload for inference (ensure this is correctly formatted)
    payload = {"text": "My name is Karl Popper. The philosopher of epistemology."}

    try:
        # Send inference request to Hugging Face API
        response = client.post(json=payload)  # Only send the payload as JSON

        # If the response is raw audio data (bytes)
        if isinstance(response, bytes):
            print(f"Received audio data for voice preset {voice_preset}.")
            audio_data = response  # Raw audio data (in bytes)
            # Convert bytes to numpy array and play using sounddevice
            audio_array = np.frombuffer(audio_data, dtype=np.float32)  # Convert byte data to numpy array
            sample_rate = 24000  # Adjust the sample rate if necessary
            sd.play(audio_array, samplerate=sample_rate)
            sd.wait()
        else:
            print(f"Error: Unexpected response format for voice preset {voice_preset}.")
    except Exception as e:
        print(f"Error while processing preset {voice_preset}: {str(e)}")
