# https://huggingface.co/docs/transformers/main/en/model_doc/bark
from transformers import AutoProcessor, BarkModel
import torch
import sounddevice as sd

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("suno/bark-small", torch_dtype=torch.float16)
model = BarkModel.from_pretrained("suno/bark-small", torch_dtype=torch.float16).to(device)
# model.enable_cpu_offload()
# model = model.to_bettertransformer()

# Loop from v2/de_speaker_0 to v2/de_speaker_9
for i in range(0, 2):
    voice_preset = f"v2/de_speaker_{i}"
    print(f"Playing voice preset: {voice_preset}")
    inputs = processor("My name is Karl Popper. The philosopher of epistemology.",
                       voice_preset=voice_preset,
                       return_tensors="pt",
                       return_attention_mask=True)

    # Ensure inputs are moved to the correct device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    audio_array = model.generate(**inputs,
                                 ).to(device)
    audio_array = audio_array.cpu().numpy().squeeze()

    # Convert audio array to float32 for sounddevice compatibility
    audio_array = audio_array.astype("float32")

    # Get sample rate
    sample_rate = 24000
    # Play the audio
    sd.play(audio_array, samplerate=sample_rate)
    sd.wait()

# voice_preset = "v2/de_speaker_0"  # "v2/en_speaker_6"
# inputs = processor("Hello, my dog is cute", voice_preset=voice_preset)
#
# # Ensure inputs are moved to the correct device
# inputs = {key: value.to(device) for key, value in inputs.items()}
#
# audio_array = model.generate(**inputs).to(device)
# audio_array = audio_array.cpu().numpy().squeeze()
#
# # Convert audio array to float32 for sounddevice compatibility
# audio_array = audio_array.astype("float32")
#
# # Get sample rate
# sample_rate = 24000
# # Play the audio
# sd.play(audio_array, samplerate=sample_rate)
# sd.wait()

# Preset voices
# https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c
