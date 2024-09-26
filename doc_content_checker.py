# This file contains a simple function that can be programmed to check the content of a document.
# DocumentProcessor pipeline can take this function and will pass into it a chuck of text to check.
# The pipeline will ONLY pass in the next for paragraph 1 of each section. You then write code to determine
# if you want to include this section or skip it.
# For example, you can check if the first paragraph of a section is called "INDEX" or "CONTENTS" and then skip it.

def skip_content(first_paragraph) -> bool:
    # If the first paragraph of a section starts with the word "index" (once taken lower case) followed by newline
    # or "contents" (once taken lower case) followed by newline, then skip this section.
    paragraph: str = first_paragraph.lower()
    if (paragraph.startswith("index\n")
            or paragraph.startswith("name index\n")
            or paragraph.startswith("subject index\n")
            or paragraph.startswith("contents\n")):
        return True
    else:
        return False
