from transformers import pipeline
import scipy.io.wavfile as wavfile
import numpy as np

# Initialize the pipeline
synthesiser = pipeline("text-to-speech", model="suno/bark", device=0)

# Generate speech from text
speech = synthesiser("Hello, my dog is cooler than you!", forward_params={"do_sample": True})

# Extract the audio data
audio_data = speech["audio"]

# Reshape the audio data if it's 2D (e.g., [1, N])
if audio_data.ndim == 2 and audio_data.shape[0] == 1:
    audio_data = audio_data.squeeze(0)  # Remove the first dimension

# Validate that the audio data is now 1D
if audio_data.ndim != 1:
    raise ValueError(f"Invalid audio data shape: {audio_data.shape}. It should be a 1D array.")

# Normalize the audio data to be in the int16 range (-32768 to 32767)
# audio_data = np.int16(audio_data * 32767)  # Scale to 16-bit PCM range

# Check if the sampling rate is valid
sampling_rate = speech["sampling_rate"]
if not isinstance(sampling_rate, int):
    raise ValueError(f"Invalid sampling rate: {sampling_rate}. It must be an integer.")

# Write the audio data to a WAV file
wavfile.write("bark_out.wav", rate=sampling_rate, data=audio_data)
