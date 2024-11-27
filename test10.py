import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, AutoFeatureExtractor, set_seed
import soundfile as sf
import sounddevice as sd
from string import punctuation
from transformers.models.speecht5.number_normalizer import EnglishNumberNormalizer
import re


number_normalizer = EnglishNumberNormalizer()

def preprocess(text):
    text = number_normalizer(text).strip()
    text = text.replace("-", " ")
    if text[-1] not in punctuation:
        text = f"{text}."

    abbreviations_pattern = r'\b[A-Z][A-Z\.]+\b'

    def separate_abb(chunk):
        chunk = chunk.replace(".", "")
        print(chunk)
        return " ".join(chunk)

    abbreviations = re.findall(abbreviations_pattern, text)
    for abv in abbreviations:
        if abv in text:
            text = text.replace(abv, separate_abb(abv))
    return text


device = "cuda" if torch.cuda.is_available() else "cpu"

repo_id = "parler-tts/parler-tts-mini-v1"
feature_extractor = AutoFeatureExtractor.from_pretrained(repo_id)
SAMPLE_RATE = feature_extractor.sampling_rate
SEED = 42

model = ParlerTTSForConditionalGeneration.from_pretrained(repo_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(repo_id)

prompt = "I am Karl Popper, a philosopher of science and a critic of induction."
description = ("Gary has a deep voice, german accent, and arrogant tone."
               "The recording is of very high quality, with the speaker's voice "
               "sounding clear and very close up.")

input_ids = tokenizer(description.strip(), return_tensors="pt").input_ids.to(device)
print(prompt)
print(preprocess(prompt))
prompt_input_ids = tokenizer(preprocess(prompt), return_tensors="pt").input_ids.to(device)

# inputs = tokenizer(description.strip(), return_tensors="pt").to(device)
# prompt = tokenizer(preprocess(text), return_tensors="pt").to(device)

set_seed(SEED)
generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
audio_arr = generation.cpu().numpy().squeeze()
# sf.write("parler_tts_out.wav", audio_arr, model.config.sampling_rate)

# audio_data = audio_arr.astype("float32")
print(SAMPLE_RATE)
print(model.config.sampling_rate)
sd.play(audio_arr, samplerate=model.config.sampling_rate)
sd.wait()  # Wait until the audio finishes playing

# pip install git+https://github.com/huggingface/parler-tts.git
# https://github.com/huggingface/parler-tts/blob/main/INFERENCE.md
# https://github.com/huggingface/parler-tts
# https://github.com/huggingface/parler-tts/blob/main/INFERENCE.md
# GPUs for Spaces
# https://huggingface.co/docs/hub/en/spaces-gpus
# https://github.com/huggingface/hub-docs/blob/main/docs/hub/spaces-zerogpu.md
