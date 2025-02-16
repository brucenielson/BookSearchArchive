from docling.document_converter import DocumentConverter, ConversionResult
from docling_core.types import DoclingDocument
from pathlib import Path
from kokoro import KPipeline
# import sounddevice as sd
import soundfile as sf
import numpy as np
from docling_parser import DoclingParser
from custom_haystack_components import load_valid_pages


# https://huggingface.co/hexgrad/Kokoro-82M
# https://huggingface.co/hexgrad/Kokoro-82M/discussions/64
# https://huggingface.co/hexgrad/Kokoro-82M/discussions/120
# pip install kokoro
# pip install soundfile
def load_pdf_text(file_path: str) -> str:
    """Load a PDF, caching as JSON if needed, and export its text."""
    converter = DocumentConverter()
    json_path = Path(file_path).with_suffix('.json')
    if json_path.exists():
        book = DoclingDocument.load_from_json(json_path)
    else:
        result = converter.convert(file_path)
        book = result.document
        book.save_as_json(json_path)
    return book.export_to_text()


def simple_generate_and_save_audio(text: str,
                                   output_file: str,
                                   voice: str = 'af_heart',
                                   sample_rate: int = 24000,
                                   play_audio: bool = False):
    """Generate audio from text using Kokoro, play each segment, and save combined audio to a WAV file."""
    pipeline = KPipeline(lang_code='a')
    audio_segments = []

    for i, (gs, ps, audio) in enumerate(pipeline(text, voice=voice, speed=1, split_pattern=r'\n+')):
        print(f"Segment {i}: Graphemes: {gs} | Phonemes: {ps}")
        # if play_audio:
        #     sd.play(audio, sample_rate)
        #     sd.wait()
        audio_segments.append(audio)

    combined_audio = np.concatenate(audio_segments)
    sf.write(output_file, combined_audio, sample_rate)
    print(f"Audio saved to {output_file}")


def simple_example():
    text = "Hello, world! This is a test of the Kokoro TTS system."
    simple_generate_and_save_audio(text, "output2.wav", play_audio=True)


def simple_pdf_to_audio():
    file_path = r"D:\Documents\AI\BookSearchArchive\documents\Realism and the Aim of Science -- Karl Popper -- 2017.pdf"
    text = load_pdf_text(file_path)
    print("Extracted text from PDF.")
    simple_generate_and_save_audio(text, "output.wav")


def docling_parser_pdf_to_audio(file_path: str,
                                output_file: str,
                                voice: str = 'af_heart',
                                sample_rate: int = 24000):
    converter = DocumentConverter()
    result: ConversionResult = converter.convert(file_path)
    book: DoclingDocument = result.document
    valid_pages = load_valid_pages("documents/pdf_valid_pages.csv")
    start_page = None
    end_page = None
    if book.name in valid_pages:
        start_page, end_page = valid_pages[book.name]

    parser = DoclingParser(book, {},
                           min_paragraph_size=300,
                           start_page=start_page,
                           end_page=end_page,
                           double_notes=True)
    paragraphs, meta = parser.run()
    """Generate audio from text using Kokoro, play each segment, and save combined audio to a WAV file."""
    pipeline = KPipeline(lang_code='a')
    audio_segments = []
    for i, paragraph in enumerate(paragraphs):
        print(f"Generating audio for paragraph {i+1}/{len(paragraphs)}")
        # convert paragraph which is a ByteSteam back to regular text
        text = paragraph.to_string('utf-8')

        for j, (gs, ps, audio) in enumerate(pipeline(text, voice=voice, speed=1, split_pattern=r'\n+')):
            print(f"Segment {j}: Graphemes: {gs} | Phonemes: {ps}")
            audio_segments.append(audio)

    combined_audio = np.concatenate(audio_segments)
    sf.write(output_file, combined_audio, sample_rate)
    print(f"Audio saved to {output_file}")


def test():
    import re

    string = "11.  An apple tree"  # Note: I've included the non-breaking space here

    for char in string:
        print(f"'{char}': Unicode code point = {ord(char)}")

    match = re.match(r"^\d+(?![ .]|\d|$)", string)
    if match:
        print("Match found!")
    else:
        print("Match not found.")

def main():
    file_path = r"D:\Documents\AI\BookSearchArchive\documents\A World of Propensities -- Karl Popper -- 2018.pdf"
    docling_parser_pdf_to_audio(file_path, "output.wav")
    # text = load_pdf_text(file_path)
    # simple_generate_and_save_audio(text, "output2.wav")
    # test()

if __name__ == "__main__":
    main()
