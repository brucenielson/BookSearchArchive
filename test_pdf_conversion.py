from docling.document_converter import DocumentConverter
import pathlib
import pymupdf4llm

# Test docing
source = "documents/Realism and the Aim of Science -- Karl Popper -- 2017.pdf"  # document per local path or URL
# converter = DocumentConverter()
# print("Converting document using Docling...")
# result = converter.convert(source)
# print("Conversion complete.")
# markdown = result.document.export_to_markdown()
# # Save file
# output_path = pathlib.Path("test_docling.md")
# output_path.write_text(markdown, encoding="utf-8")

# Now try with PyMuPDF4LLM
print("Converting document using PyMuPDF4LLM...")
markdown = pymupdf4llm.to_markdown(source)
# Save file
output_path = pathlib.Path("test_pymupdf4llm.md")
output_path.write_text(markdown, encoding="utf-8")

