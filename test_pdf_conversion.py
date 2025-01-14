from docling.document_converter import DocumentConverter, ConversionResult
from docling_core.types import DoclingDocument
import pathlib
import pymupdf4llm


def docling_convert_pdf(source: str):
    converter: DocumentConverter = DocumentConverter()
    print("Converting document using Docling...")
    result: ConversionResult = converter.convert(source)
    doc: DoclingDocument = result.document
    path = pathlib.Path("test_docling.json")
    doc.save_as_json(path)
    return doc


def docling_load_json(source: str):
    path = pathlib.Path(source)
    doc: DoclingDocument = DoclingDocument.load_from_json(path)
    return doc


def docling_to_markdown(doc: DoclingDocument, output_path: str):
    markdown = doc.export_to_markdown()
    path = pathlib.Path(output_path)
    path.write_text(markdown, encoding="utf-8")
    return markdown

# Loop over text in the document


def pymupdf4llm_to_markdown(source: str):
    print("Converting document using PyMuPDF4LLM...")
    markdown = pymupdf4llm.to_markdown(source)
    # Save file
    output_path = pathlib.Path("test_pymupdf4llm.md")
    output_path.write_text(markdown, encoding="utf-8")


def main():
    source = "documents/Realism and the Aim of Science -- Karl Popper -- 2017.pdf"  # document per local path or URL
    # doc = docling_convert_pdf(source)
    doc = docling_load_json("test_docling.json")
    for text in doc.texts:
        print(text.label, text.text)
        print("\n")


if __name__ == "__main__":
    main()
