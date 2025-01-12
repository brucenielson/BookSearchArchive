from docling.document_converter import DocumentConverter, ConversionResult
from docling_core.types import DoclingDocument
import pathlib
import pymupdf4llm

# Test docing
source = "documents/Realism and the Aim of Science -- Karl Popper -- 2017.pdf"  # document per local path or URL
converter: DocumentConverter = DocumentConverter()
print("Converting document using Docling...")
result: ConversionResult = converter.convert(source)
doc: DoclingDocument = result.document
# print("Conversion complete.")
# Save file
output_path = pathlib.Path("test_docling.md")
markdown = doc.export_to_markdown()
output_path.write_text(markdown, encoding="utf-8")
output_path = pathlib.Path("test_docling.json")
doc.save_as_json(output_path)
# Loop over text in the document
for text in doc.texts:
    print(text.text)
    print("\n\n")

new_doc = doc.load_from_json(output_path)
pass


# Now try with PyMuPDF4LLM
# print("Converting document using PyMuPDF4LLM...")
# markdown = pymupdf4llm.to_markdown(source)
# # Save file
# output_path = pathlib.Path("test_pymupdf4llm.md")
# output_path.write_text(markdown, encoding="utf-8")

