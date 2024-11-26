from huggingface_hub import InferenceClient
import numpy as np
import sounddevice as sd
import generator_model as gen
import requests
import json

# Load secret
hf_secret: str = gen.get_secret(r'D:\Documents\Secrets\huggingface_secret.txt')


# Function to check model health
def check_model_health(model_id: str, token: str):
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {token}"}

    try:
        response = requests.get(url, headers=headers)
        print(f"Model Health Check:")
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {json.dumps(dict(response.headers), indent=2)}")

        if response.status_code == 200:
            print(f"Model {model_id} is available!")
            return True
        else:
            print(f"Model {model_id} returned status code {response.status_code}")
            print(f"Response Content: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Error checking model health: {e}")
        return False


# Initialize the Inference Client
def create_inference_client(model: str, token: str):
    try:
        client = InferenceClient(
            model=model,
            token=token
        )
        return client
    except Exception as e:
        print(f"Error creating InferenceClient: {e}")
        return None


def text_to_speech(client: InferenceClient, text: str):
    try:
        # Detailed input logging
        print(f"Generating speech for text: {text}")
        print(f"Using model: {client.model}")

        # Generate audio using the InferenceClient
        audio_response = client.text_to_speech(text)

        # Verify the response
        if not isinstance(audio_response, bytes):
            print("Unexpected response type")
            return None

        # Convert bytes to numpy array for playback
        audio_array = np.frombuffer(audio_response, dtype=np.float32)

        print(f"Received audio data:")
        print(f"  - Length: {len(audio_array)} samples")
        print(f"  - Data type: {audio_array.dtype}")
        print(f"  - Min value: {audio_array.min()}")
        print(f"  - Max value: {audio_array.max()}")

        # Play the audio
        try:
            sd.play(audio_array, samplerate=24000)
            sd.wait()
        except Exception as play_error:
            print(f"Audio playback error: {play_error}")

        # Save the audio file
        try:
            with open("output_speech.wav", "wb") as audio_file:
                audio_file.write(audio_response)
            print("Audio saved to output_speech.wav")
        except Exception as save_error:
            print(f"Audio saving error: {save_error}")

        return audio_response

    except Exception as e:
        print(f"Detailed Error during text-to-speech generation:")
        print(f"  - Error Type: {type(e)}")
        print(f"  - Error Message: {e}")
        return None


# Main execution
def main():
    model_id = "suno/bark-small"

    # Check model health first
    model_available = check_model_health(model_id, hf_secret)
    if not model_available:
        print("Model is not available. Exiting.")
        return

    # Create client
    client = create_inference_client(model_id, hf_secret)
    if not client:
        print("Failed to create InferenceClient. Exiting.")
        return

    # Perform text to speech
    text_to_speech(client, "My name is Karl Popper. The philosopher of epistemology.")


# Run the main function
if __name__ == "__main__":
    main()