from docling.document_converter import DocumentConverter

source = "https://arxiv.org/pdf/2408.09869"  # document per local path or URL
converter = DocumentConverter()
result = converter.convert(source)
print(result.document.export_to_markdown())  # output: "## Docling Technical Report[...]"
# Write out markdown file
with open("docling_technical_report.md", "w") as f:
    f.write(result.document.export_to_markdown())
    