from docling.document_converter import DocumentConverter


def test_document_converter(source):
    converter = DocumentConverter()
    result = converter.convert(source)
    print(result.document.export_to_markdown())  # output: "## Docling Technical Report[...]"
    # Write out markdown file
    with open("test_docling.md", "w") as f:
        f.write(result.document.export_to_markdown())


source = "https://arxiv.org/pdf/2408.09869"  # document per local path or URL
test_document_converter(source)
source = "documents/A World of Propensities by Karl Popper (1997).pdf"
test_document_converter(source)
