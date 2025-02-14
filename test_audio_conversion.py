from docling.document_converter import DocumentConverter, ConversionResult
from typing import List, Optional, Dict, Any, Union, Callable, Tuple, Set
from docling_core.types import DoclingDocument
from pathlib import Path
from kokoro import KPipeline
import sounddevice as sd
import soundfile as sf
import numpy as np


def load_pdf(file_path: str) -> Tuple[DoclingDocument, Dict[str, str]]:
    converter: DocumentConverter = DocumentConverter()
    # Check if already cached as a json
    path = Path(file_path).with_suffix('.json')
    book: DoclingDocument
    if path.exists():
        book = DoclingDocument.load_from_json(path)
    else:
        result: ConversionResult = converter.convert(file_path)
        book = result.document
        # Cache the book as a json
        book.save_as_json(path)
    book_meta_data: Dict[str, str] = {
        "book_title": book.name,
        "file_path": file_path
    }
    return book, book_meta_data


def kokoro_pipeline():
    # https://huggingface.co/hexgrad/Kokoro-82M#usage
    output_file = 'output.wav'
    # 🇺🇸 'a' => American English, 🇬🇧 'b' => British English
    pipeline = KPipeline(lang_code='a')  # <= make sure lang_code matches voice

    # This text is for demonstration purposes only, unseen during training
    text = '''
    The sky above the port was the color of television, tuned to a dead channel.
    "It's not like I'm using," Case heard someone say, as he shouldered his way through the crowd around the door of the Chat. "It's like my body's developed this massive drug deficiency."
    It was a Sprawl voice and a Sprawl joke. The Chatsubo was a bar for professional expatriates; you could drink there for a week and never hear two words in Japanese.
    
    These were to have an enormous impact, not only because they were associated with Constantine, but also because, as in so many other areas, the decisions taken by Constantine (or in his name) were to have great significance for centuries to come. One of the main issues was the shape that Christian churches were to take, since there was not, apparently, a tradition of monumental church buildings when Constantine decided to help the Christian church build a series of truly spectacular structures. The main form that these churches took was that of the basilica, a multipurpose rectangular structure, based ultimately on the earlier Greek stoa, which could be found in most of the great cities of the empire. Christianity, unlike classical polytheism, needed a large interior space for the celebration of its religious services, and the basilica aptly filled that need. We naturally do not know the degree to which the emperor was involved in the design of new churches, but it is tempting to connect this with the secular basilica that Constantine completed in the Roman forum (the so-called Basilica of Maxentius) and the one he probably built in Trier, in connection with his residence in the city at a time when he was still caesar.
    
    [Kokoro](/kˈOkəɹO/) is an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient. With Apache-licensed weights, [Kokoro](/kˈOkəɹO/) can be deployed anywhere from production environments to personal projects.
    '''

    # 4️⃣ Generate, display, and save audio files in a loop.
    generator = pipeline(
        text, voice='af_heart', # <= change voice here
        speed=1, split_pattern=r'\n+'
    )
    audio_segments = []
    for i, (gs, ps, audio) in enumerate(generator):
        print(i)  # i => index
        print(gs) # gs => graphemes/text
        print(ps) # ps => phonemes
        # display(Audio(data=audio, rate=24000, autoplay=i==0))
        # sf.write(f'{i}.wav', audio, 24000) # save each audio file
        sd.play(audio, 24000)
        sd.wait()  # Wait until the current audio segment finishes playing.
        audio_segments.append(audio)

    # Concatenate all segments into one audio array.
    combined_audio = np.concatenate(audio_segments)

    # Save to a single file. Adjust sample_rate as per your model (assumed here to be 24000 Hz).
    sample_rate = 24000
    sf.write(output_file, combined_audio, sample_rate)
    print(f"Audio saved to {output_file}")


def main():
    file_path = r"D:\Documents\AI\BookSearchArchive\documents\Realism and the Aim of Science -- Karl Popper -- 2017.pdf"
    book, book_meta_data = load_pdf(file_path)
    # Convert the DoclingDocument object to Text
    text: str = book.export_to_text()
    print(text)
    # Run the kokoro pipeline
    kokoro_pipeline()


if __name__ == "__main__":
    main()
