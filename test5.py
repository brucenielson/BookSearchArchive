from pydub import AudioSegment
import sounddevice as sd
import numpy as np

# Path to your FLAC file
flac_path = "facebook_mms-tts-eng_test_sentence.flac"
wav_path = flac_path.replace(".flac", ".wav")

# Convert FLAC to WAV
try:
    print(f"Converting {flac_path} to WAV...")
    audio = AudioSegment.from_file(flac_path, format="flac")
    audio.export(wav_path, format="wav")
    print(f"WAV file saved to {wav_path}")
except Exception as e:
    print(f"Error converting FLAC to WAV: {e}")
    exit()

# Play the WAV file
try:
    print(f"Playing WAV file: {wav_path}")
    audio_data = np.array(audio.get_array_of_samples())
    sample_rate = audio.frame_rate
    sd.play(audio_data, samplerate=sample_rate)
    sd.wait()  # Wait until playback finishes
except Exception as e:
    print(f"Error playing WAV file: {e}")
