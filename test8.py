import requests

# Space URL for API (replace 'user' and 'space-name' with the actual Space info)
SPACE_API_URL = "https://huggingface.co/spaces/parler-tts/parler-tts-expresso"

# Input text for TTS
input_text = "Hello, this is a test of the text-to-speech feature!"

# Create the payload
payload = {
    "data": [input_text]  # 'data' is often used in Gradio-based APIs
}

# Make the request
response = requests.post(SPACE_API_URL, json=payload)

# Handle the response
if response.status_code == 200:
    result = response.json()
    audio_url = result.get("data", [None])[0]  # Extract the audio file link or base64
    if audio_url:
        print("Audio URL:", audio_url)
        # Download the audio file if needed
        audio_response = requests.get(audio_url)
        with open("output_audio.wav", "wb") as audio_file:
            audio_file.write(audio_response.content)
else:
    print(f"Error: {response.status_code} - {response.text}")
