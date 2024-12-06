import soundfile as sf
with open('facebook_mms-tts-eng_test_sentence.flac', 'rb') as f:
    data, samplerate = sf.read(f)

# https://github.com/bastibe/python-soundfile/issues/450
