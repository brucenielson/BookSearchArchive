from docling.document_converter import DocumentConverter, ConversionResult
from docling_core.types import DoclingDocument
from pathlib import Path
from kokoro import KPipeline
import sounddevice as sd
import soundfile as sf
import numpy as np


# https://huggingface.co/hexgrad/Kokoro-82M/discussions/64
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


def generate_and_save_audio(text: str, output_file: str,
                            voice: str = 'af_heart',
                            sample_rate: int = 24000,
                            play_audio: bool = False):
    """Generate audio from text using Kokoro, play each segment, and save combined audio to a WAV file."""
    pipeline = KPipeline(lang_code='a')
    audio_segments = []

    for i, (gs, ps, audio) in enumerate(pipeline(text, voice=voice, speed=1, split_pattern=r'\n+')):
        print(f"Segment {i}: Graphemes: {gs} | Phonemes: {ps}")
        if play_audio:
            sd.play(audio, sample_rate)
            sd.wait()
        audio_segments.append(audio)

    combined_audio = np.concatenate(audio_segments)
    sf.write(output_file, combined_audio, sample_rate)
    print(f"Audio saved to {output_file}")


def main():
    file_path = r"D:\Documents\AI\BookSearchArchive\documents\Realism and the Aim of Science -- Karl Popper -- 2017.pdf"
    text = load_pdf_text(file_path)
    print("Extracted text from PDF.")
    generate_and_save_audio(text, "output.wav")


if __name__ == "__main__":
    main()
