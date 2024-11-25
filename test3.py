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

# Try HireSpeech
# https://huggingface.co/spaces/LeeSangHoon/HierSpeech_TTS/tree/main
# https://huggingface.co/spaces/LeeSangHoon/HierSpeech_TTS
# https://www.youtube.com/watch?v=4-2Jk8muo7c
# https://www.reddit.com/r/MachineLearning/comments/189zf6w/news_text_to_speech_is_getting_crazy_good/

# Text to Speech
# https://www.youtube.com/watch?v=47hba0If7dY
# https://huggingface.co/models?pipeline_tag=text-to-speech

# Other Databases to research
# OpenSearch
# https://haystack.deepset.ai/integrations/opensearch-document-store
# https://opensearch.org/

# Needle - Not open source!
# https://haystack.deepset.ai/integrations/needle
# https://needle.sourceforge.net/
# https://docs.needle-ai.com/

# Neo4j
# https://haystack.deepset.ai/integrations/neo4j-document-store
# https://medium.com/@trienpont/neo4j-graph-database-the-non-relational-database-that-improves-on-relations-77a3b7d9408e#:~:text=What%20is%20Neo4j?,The%20edges%20represent%20relationships.


# MongoDB - Not open source! But there is a community edition
# https://haystack.deepset.ai/integrations/mongodb
# https://www.mongodb.com/try/download/community

# FAISS
# https://haystack.deepset.ai/integrations/faiss-document-store
# https://github.com/facebookresearch/faiss#readme
# Some of the methods, like those based on binary vectors and compact quantization codes, solely use a compressed
# representation of the vectors and do not require to keep the original vectors. This generally comes at the cost of
# a less precise search but these methods can scale to billions of vectors in main memory on a single server.
# Other methods, like HNSW and NSG add an indexing structure on top of the raw vectors to make searching more efficient


# Chainlit UI Agents
# https://haystack.deepset.ai/integrations/chainlit

# Entailment Checker!!!
# https://haystack.deepset.ai/integrations/entailment-checker

# LanceDB
# https://haystack.deepset.ai/integrations/lancedb

# Llamafile!!!
# https://haystack.deepset.ai/integrations/llamafile

# Other Integrations
# https://haystack.deepset.ai/integrations

# Try out Verba!!!
# https://verba.weaviate.io/
# https://github.com/weaviate/verba

# ollama
# https://ollama.com/download

# https://docs.haystack.deepset.ai/docs/choosing-the-right-generator
