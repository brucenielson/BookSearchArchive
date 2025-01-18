from copy import deepcopy
from bs4 import BeautifulSoup, Tag
from typing import List, Dict, Tuple, Iterator, Optional, Union
# noinspection PyPackageRequirements
from haystack.dataclasses import ByteStream
from parse_utils import enhance_title
from docling.document_converter import DocumentConverter, ConversionResult
from docling_core.types import DoclingDocument
from docling_core.types.doc.document import SectionHeaderItem, ListItem, TextItem, PageItem


# def get_header_level(paragraph: Tag) -> Optional[int]:
#     """Return the level of the header (1 for h1, 2 for h2, etc.), or 0 if not a header."""
#     # Check for direct header tag
#     if paragraph.name.startswith('h') and paragraph.name[1:].isdigit():
#         return int(paragraph.name[1:])  # Extract the level from 'hX' or 'hXY'
#
#     # Check for class name equivalent to header tags
#     if hasattr(paragraph, 'attrs') and 'class' in paragraph.attrs:
#         section_headers: List[str] = ['pre-title1', 'h']
#         for cls in paragraph.attrs['class']:
#             if cls.lower() in section_headers:
#                 return 0  # Equivalent to h0 effectively
#             elif cls.lower().startswith('h') and cls[1:].isdigit():
#                 return int(cls[1:])  # Extract level from class name 'hX' or 'hXY'
#     return None
#
#
# def is_title(tag: Tag) -> bool:
#     # # A title isn't a header
#     # noinspection SpellCheckingInspection
#     keywords: List[str] = ['title', 'chtitle', 'tochead', 'title1', 'h1_label']
#     is_a_title: bool = (hasattr(tag, 'attrs') and 'class' in tag.attrs and
#                         any(cls.lower().startswith(keyword) or cls.lower().endswith(keyword)
#                             for cls in tag.attrs['class'] for keyword in keywords))
#     return is_a_title
#
#
# def is_header1_title(paragraph: Tag, h1_count: int) -> bool:
#     header_level: int = get_header_level(paragraph)
#     if header_level == 1 and h1_count == 1:
#         return True
#     return False
#
#
# def is_section_title(tag: Tag) -> bool:
#     """Check if the tag is a title, heading, or chapter number."""
#     if tag is None:
#         return False
#
#     header_lvl: int = get_header_level(tag)
#     return is_title(tag) or header_lvl is not None or is_chapter_number(tag)
#
#
# def is_chapter_number(paragraph: Tag) -> bool:
#     # List of class names to check for chapter numbers
#     # noinspection SpellCheckingInspection
#     chapter_classes = ['chno', 'ch-num']
#     # noinspection SpellCheckingInspection
#     return (hasattr(paragraph, 'attrs') and 'class' in paragraph.attrs and
#             any(cls in paragraph.attrs['class'] for cls in chapter_classes) and
#             paragraph.text.isdigit())
#
#
# def get_page_num(paragraph: Tag) -> str:
#     # Try to get a page number - return it as a string instead of an int to accommodate roman numerals
#     # Return None if none found on this paragraph
#     tags: List[Tag] = paragraph.find_all(
#         lambda x: (x.name == 'a' or x.name == 'span') and x.get('id')
#         and (x['id'].startswith('page_') or (x['id'].startswith('p') and x['id'][1:].isdigit()))
#     )
#     page_num: Optional[str] = None
#     if tags:
#         # Extract the page number from the anchor or span tag id
#         for tag in tags:
#             page_id = tag.get('id')
#             if page_id.startswith('page_'):
#                 page_num = page_id.split('_')[-1]
#             elif page_id.startswith('p') and page_id[1:].isdigit():
#                 page_num = page_id[1:]  # Extract the digits after 'p'
#
#     if not page_num:
#         # Check for a page number embedded in the paragraph's id in
#         # format 'pXXXX-' where XXXX is the page number with leading zeros
#         page_id: str = paragraph.get('id')
#         if page_id and page_id.startswith('p'):
#             page_num = page_id[1:].split('-')[0]
#             try:
#                 page_num = str(int(page_num))  # Remove leading zeros
#             except ValueError:
#                 page_num = None
#     return page_num
#
#
# def is_sup_first_content(tag, sup_tag):
#     for content in tag.contents:
#         if isinstance(content, str) and not content.strip():
#             # Skip empty or whitespace-only strings
#             continue
#         # Check if the content is exactly the <sup> tag or contains it
#         in_first_content: bool = content == sup_tag or (isinstance(content, Tag) and sup_tag in content.descendants)
#         # Is sup both in the first content and also literally the text matches (proving it's a footnote)?
#         return in_first_content and sup_tag.text.strip() == content.text.strip()
#     return False
#
#
# def recursive_yield_tags(tag: Tag, remove_footnotes: bool = False) -> Iterator[Tag]:
#     invalid_children: List[str] = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8']
#     # If the tag has no <p> tags or header tags under it and contains text, yield it
#     # Unless it is a div tag. The Haystack HTML parse doesn't always handle those right, so
#     # Dig one level deeper.
#     if not tag.name == 'div' and tag.get_text(strip=True) and not tag.find(invalid_children):
#         # Make a deep copy of the tag to avoid modifying the original
#         tag_copy: Tag = deepcopy(tag)
#         # Clean up of paragraph text
#         for br in tag_copy.find_all('br'):
#             br.insert_after(' ')
#         # Remove footnotes - but not if the sup tag is at the start of the paragraph
#         # Iterate over each <sup> tag
#         if remove_footnotes:
#             for fn in tag_copy.find_all('sup'):
#                 if not is_sup_first_content(tag_copy, fn):
#                     # Remove the <sup> tag if it's not within the first content
#                     fn.extract()
#
#         yield tag_copy
#     else:
#         # Recursively go through the children of the current tag
#         for child in tag.children:
#             if isinstance(child, Tag):
#                 # Yield the child tags that meet the criteria
#                 yield from recursive_yield_tags(child, remove_footnotes=remove_footnotes)
#
#
# def get_chapter_info(tags: List[Tag],
#                      h1_tags: List[Tag],
#                      h2_tags: List[Tag],
#                      h3_tags: List[Tag]) -> Tuple[str, int, str]:
#     if not tags:
#         return "", 0, ""
#     # Get the chapter title from the tag
#     chapter_title: str = ""
#     # Search for the chapter title within the tags that come before the first paragraph tag (that isn't
#     # stylized to look like a header tag)
#     # Use is_title to check for a specific title tag
#     # If that fails you can use get_header_level to look for either a h1 or h2 tag but ONLY if that is the sole
#     # h1 or h2 tag in the whole section.
#     # There may be more than one title (like a subtitle) and you'll want to combine them via ": " separators.
#     # Use enhance_title to clean up the title text.
#     # Once you find your first paragraph that isn't a title or header, you can assume you've got the full title.
#
#     # Create iterator using recursive_yield_tags
#     # Count h1 tags
#     # h1_tags: List[Tag] = top_tag.find_all('h1')
#     # Remove any h1 tags that have class 'ch_num'
#     h1_tags = [tag for tag in h1_tags if not is_chapter_number(tag) and not is_title(tag)]
#     h1_tag_count: int = len(h1_tags)
#     h2_tag_count: int = len(h2_tags)
#     h3_tag_count: int = len(h3_tags)
#     chapter_number: int = 0
#     tags_to_delete: List[int] = []
#     first_page_num: str = ""
#     for i, tag in enumerate(tags):
#         first_page_num = get_page_num(tag) or first_page_num
#         if is_title(tag):
#             # This tag is used in the title, so we need to delete the tag from the list of tags
#             tags_to_delete.append(i)
#             title_text = enhance_title(tag.text)
#             if chapter_title:
#                 chapter_title += ": " + title_text
#             else:
#                 chapter_title = title_text
#         elif is_chapter_number(tag):
#             tags_to_delete.append(i)
#             chapter_number = int(tag.text.strip())
#         elif chapter_title == "" and tag.name != 'p':
#             # Check for a header tag that isn't a title
#             if h1_tag_count == 1 and get_header_level(tag) == 1:
#                 # Using an H1 tag as a chapter title
#                 tags_to_delete.append(i)
#                 title_text = enhance_title(tag.text)
#                 chapter_title = title_text
#             elif h1_tag_count == 0:
#                 if h2_tag_count == 1 and get_header_level(tag) == 2:
#                     # Using an H2 tag as a chapter title
#                     tags_to_delete.append(i)
#                     title_text = enhance_title(tag.text)
#                     chapter_title = title_text
#                 elif h3_tag_count == 1 and get_header_level(tag) == 3:
#                     # I don't think this ever happens, but using a h3 tag as a chapter title
#                     tags_to_delete.append(i)
#                     title_text = enhance_title(tag.text)
#                     chapter_title = title_text
#         elif tag.name == 'p' and not is_chapter_number(tag):
#             # We allow a couple of paragraphs before the title for quotes and such
#             if chapter_title or i > 2:
#                 break
#
#     # Delete the tags that were used in the title
#     for i in sorted(tags_to_delete, reverse=True):
#         del tags[i]
#
#     return chapter_title, chapter_number, first_page_num


def is_section_header(text: Union[SectionHeaderItem, ListItem, TextItem]) -> bool:
    if text is None:
        return False
    return text.label == "section_header"


def is_page_footer(text: Union[SectionHeaderItem, ListItem, TextItem]) -> bool:
    return text.label == "page_footer"


def is_page_header(text: Union[SectionHeaderItem, ListItem, TextItem]) -> bool:
    return text.label == "page_header"


def is_footnote(text: Union[SectionHeaderItem, ListItem, TextItem]) -> bool:
    return text.label == "footnote"


def is_text_break(text: Union[SectionHeaderItem, ListItem, TextItem]) -> bool:
    return is_page_header(text) or is_section_header(text) or is_footnote(text)


def is_page_not_text(text: Union[SectionHeaderItem, ListItem, TextItem]) -> bool:
    return text.label not in ["text", "list_item", "formula"]


def is_page_text(text: Union[SectionHeaderItem, ListItem, TextItem]) -> bool:
    return not is_page_not_text(text)


def is_ends_with_punctuation(text: str) -> bool:
    return text.endswith(".") or text.endswith("?") or text.endswith("!")


def is_sentence_end(text: str) -> bool:
    has_end_punctuation: bool = is_ends_with_punctuation(text)
    # Does it end with a closing bracket, quote, etc.?
    ends_with_bracket: bool = text.endswith(")") or text.endswith("]") or text.endswith("}") or text.endswith("\"")
    return (has_end_punctuation or
            (ends_with_bracket and is_ends_with_punctuation(text[0:-1])))


class DoclingParser:
    def __init__(self, doc: DoclingDocument, meta_data: dict[str, str], min_paragraph_size: int = 300):
        self._doc: DoclingDocument = doc
        self._min_paragraph_size: int = min_paragraph_size
        self._docs_list: List[ByteStream] = []
        self._meta_list: List[Dict[str, str]] = []
        self._total_text: str = ""
        self._chapter_title: str = ""
        self._meta_data: dict[str, str] = meta_data

        # Convert the list of sources to a list of DoclingDocuments
        # converter: DocumentConverter = DocumentConverter()
        # docling_docs: List[DoclingDocument] = []
        # for source in sources:
        #     result: ConversionResult = converter.convert(source)
        #     doc: DoclingDocument = result.document
        #     docling_docs.append(doc)

        # self._docling_docs: List[DoclingDocument] = docling_docs
        # self._meta_data: dict[str, str] = meta_data
        # self._remove_footnotes: bool = remove_footnotes
        # If True, chapters and sections named 'notes' will have double the minimum paragraph size
        # This is because notes are often very short,
        # and we want to keep them together to not dominate a semantic search
        # self._double_notes: bool = double_notes

    def total_text_length(self) -> int:
        return len(self._total_text)

    @property
    def chapter_title(self):
        return self._chapter_title

    def run(self) -> Tuple[List[ByteStream], List[Dict[str, str]]]:
        def combine_paragraphs(p1_str, p2_str):
            # If the paragraph ends without final punctuation, combine it with the next paragraph
            if is_sentence_end(p1_str):
                return p1_str + "\n" + p2_str
            else:
                return p1_str + " " + p2_str

        temp_docs: List[ByteStream] = []
        temp_meta: List[Dict[str, str]] = []
        combined_paragraph: str = ""
        para_num: int = 0
        i: int
        combined_chars: int = 0
        texts: List[Union[SectionHeaderItem, ListItem, TextItem]] = list(self._doc.texts)
        section_name: str = ""
        for i, text in enumerate(texts):
            # prev_tag: Tag = tags[j - 1] if j > 0 else None
            next_text = texts[i + 1] if i < len(texts) - 1 else None

            # Check if text starts with...
            if text.text.startswith("almost as incredible as if you fired"):
                pass

            if text.text.startswith("As these examples show, falsifiability in the sense of the demarca"):
                pass

            if is_page_footer(text):
                continue

            if is_page_header(text):
                # page_header = text.text
                continue

            if is_section_header(text):
                section_name = text.text
                continue

            if is_footnote(text):
                continue

            if is_page_not_text(text):
                pass

            min_paragraph_size: int = self._min_paragraph_size

            p_str: str = str(text.text).strip()
            p_str_chars: int = len(p_str)

            # If the paragraph ends without final punctuation, combine it with the next paragraph
            if not is_sentence_end(p_str):
                # Combine this paragraph with the previous ones
                combined_paragraph = combine_paragraphs(combined_paragraph, p_str)
                combined_chars += p_str_chars
                continue
            # If next paragraph is a new section, and we're at the end of a sentence, process this paragraph
            elif is_section_header(next_text) and is_sentence_end(p_str):
                p_str = combine_paragraphs(combined_paragraph, p_str)
                p_str_chars += combined_chars
                combined_paragraph = ""
                combined_chars = 0
            # If the combined paragraph is less than the minimum size combine it with the next paragraph
            elif combined_chars + p_str_chars < min_paragraph_size:
                # If the paragraph is too short, combine it with the next one
                # Unless this is the final paragraph on the page. That is unless that final paragraph is split up
                # across pages.
                if next_text is None:
                    # If it's the last paragraph, then process this paragraph
                    combined_paragraph = combine_paragraphs(combined_paragraph, p_str)
                    combined_chars += p_str_chars
                    p_str = combined_paragraph
                elif not is_page_text(next_text) and is_sentence_end(p_str):
                    # If it's the last paragraph on the page, then break the paragraph here
                    p_str = combine_paragraphs(combined_paragraph, p_str)
                    p_str_chars += combined_chars
                    combined_paragraph = ""
                    combined_chars = 0
                else:
                    # Combine this paragraph with the previous ones
                    combined_paragraph = combine_paragraphs(combined_paragraph, p_str)
                    combined_chars += p_str_chars
                    continue
            else:
                p_str = combine_paragraphs(combined_paragraph, p_str)
                p_str_chars += combined_chars
                combined_paragraph = ""
                combined_chars = 0

            # Remove any extra dashes in the middle of a paragraph if it is breaking up a word
            p_str = p_str.replace("­", "-")
            p_str = p_str.replace("- ", "")
            para_num += 1
            self._total_text += p_str
            byte_stream: ByteStream = ByteStream(p_str.encode('utf-8'))
            paragraph_meta_data: Dict[str, str] = {}
            paragraph_meta_data.update(self._meta_data)
            paragraph_meta_data["paragraph_#"] = str(para_num)
            # paragraph_meta_data["page_header"] = page_header
            paragraph_meta_data["section_name"] = section_name

            temp_docs.append(byte_stream)
            temp_meta.append(paragraph_meta_data)

        return temp_docs, temp_meta
