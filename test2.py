from pydub import AudioSegment

sound = AudioSegment.from_file(r"facebook_mms-tts-eng_test_sentence.flac")

# Gives error:
# FileNotFoundError: [WinError 2] The system cannot find the file specified
# Note that the file does exist and is in the right location. Changing to an absolute path to the file did not help.
