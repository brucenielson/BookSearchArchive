import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, set_seed
import sounddevice as sd


device = "cuda" if torch.cuda.is_available() else "cpu"

repo_id = "parler-tts/parler-tts-mini-v1"
SEED = 42

model = ParlerTTSForConditionalGeneration.from_pretrained(repo_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(repo_id)

prompt = "I am Karl Popper, a philosopher of science and a critic of induction."
description = ("Jon has a deep voice, german accent, and arrogant tone."
               "The recording is of very high quality, with the speaker's voice "
               "sounding clear and very close up.")

input_ids = tokenizer(description.strip(), return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

set_seed(SEED)
generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
audio_arr = generation.cpu().numpy().squeeze()
sd.play(audio_arr, samplerate=model.config.sampling_rate)
sd.wait()
