import pymupdf4llm, pymupdf
import pathlib
import fitz


doc = pymupdf.open(r"D:\Documents\AI\BookSearchArchive\documents\A World of Propensities by Karl Popper (1997).pdf")
# for page_num in range(len(doc)):
#     page = doc.load_page(page_num)
#     # md = pymupdf4llm.to_markdown(page)
#     text = page.get_text("text")
#     print(text)
#     pass

md = pymupdf4llm.to_markdown(doc)
path = pathlib.Path("test.txt")
path.write_text(md)  # , encoding="utf-8")

# https://colab.research.google.com/drive/1d3BCUI5PyV928PcJwmnx_RkvWGHGJGC9?usp=sharing

# markdown_pages = []
# for page_num in range(len(doc)):
#     page = doc.load_page(page_num)
#     page_markdown = pymupdf4llm.to_markdown(page)
#     markdown_pages.append(page_markdown)

# doc = fitz.open(r"D:\Documents\AI\BookSearchArchive\documents\A World of Propensities by Karl Popper (1997).pdf")
#
# # Initialize an empty list to store markdown content from each page
# markdown_pages = []
#
# # Process each page by creating a temporary single-page document
# for page_num in range(len(doc)):
#     # Create a new document with just the current page
#     temp_doc = fitz.open()  # Create an empty document
#     temp_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
#
#     # Convert the single-page document to markdown
#     page_markdown = pymupdf4llm.to_markdown(temp_doc)
#     markdown_pages.append(page_markdown)
#
#     # Close the temporary document
#     temp_doc.close()
#
# # Combine all pages into one markdown text
# final_markdown = "\n\n---\n\n".join(markdown_pages)
#
# # Save the combined markdown to a file
# output_path = pathlib.Path("test_markdown.txt")
# output_path.write_text(final_markdown, encoding="utf-8")
#
# print(f"Markdown output saved to {output_path.resolve()}")
#
# # Initialize an empty list to store markdown content from each page
# markdown_pages = []
#
# # Process each page by creating a temporary single-page document
# for page_num in range(len(doc)):
#     # Create a new document with just the current page
#     temp_doc = fitz.open()  # Create an empty document
#     temp_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
#
#     # Convert the single-page document to markdown
#     page_markdown = pymupdf4llm.to_markdown(temp_doc)
#     markdown_pages.append(page_markdown)
#
#     # Close the temporary document
#     temp_doc.close()
#
# # Combine all pages into one markdown text
# final_markdown = "\n\n---\n\n".join(markdown_pages)
#
# # Save the combined markdown to a file
# output_path = pathlib.Path("test_markdown.txt")
# output_path.write_text(final_markdown, encoding="utf-8")
#
# print(f"Markdown output saved to {output_path.resolve()}")


# https://pypi.org/project/Markdown/
# https://python-markdown.github.io/reference/
# https://www.digitalocean.com/community/tutorials/how-to-use-python-markdown-to-convert-markdown-text-to-html
# https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/
# https://python.plainenglish.io/why-pymupdf4llm-is-the-best-tool-for-extracting-data-from-pdfs-even-if-you-didnt-know-you-needed-7bff75313691

# PDF to Markdown - doesn't work well enough IMO
# pip install pymupdf4llm
# Needed for Haystack component
# pip install markdown-it-py mdit_plain
# Convert to HTML
# pip install markdown

# https://github.com/markdown-it/markdown-it

# https://sumansourabh.in/convert-pdf-to-markdown/
# https://github.com/VikParuchuri/marker
# https://pypi.org/project/marker-pdf/

# https://github.com/tesseract-ocr/tesseract
