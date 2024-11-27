import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import sounddevice as sd

device = "cuda" if torch.cuda.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

prompt = "I am Karl Popper, a philosopher of science and a critic of induction. [Pause]"
description = ("Gary has a deep voice, german accent, and arrogant tone."
               "The recording is of very high quality, with the speaker's voice "
               "sounding clear and very close up.")

input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# inputs = tokenizer(description.strip(), return_tensors="pt").to(device)
# prompt = tokenizer(preprocess(text), return_tensors="pt").to(device)


generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
audio_arr = generation.cpu().numpy().squeeze()
# sf.write("parler_tts_out.wav", audio_arr, model.config.sampling_rate)

# audio_data = audio_arr.astype("float32")
sd.play(audio_arr, samplerate=model.config.sampling_rate)
sd.wait()  # Wait until the audio finishes playing

# pip install git+https://github.com/huggingface/parler-tts.git
# https://github.com/huggingface/parler-tts/blob/main/INFERENCE.md
# https://github.com/huggingface/parler-tts
# https://github.com/huggingface/parler-tts/blob/main/INFERENCE.md
# GPUs for Spaces
# https://huggingface.co/docs/hub/en/spaces-gpus
# https://github.com/huggingface/hub-docs/blob/main/docs/hub/spaces-zerogpu.md
