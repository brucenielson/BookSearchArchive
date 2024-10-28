import re
from copy import deepcopy
from bs4 import BeautifulSoup, Tag
from typing import List, Dict, Tuple, Iterator, Optional
# noinspection PyPackageRequirements
from haystack.dataclasses import ByteStream


def is_roman_numeral(s: str) -> bool:
    roman_numeral_pattern = r'(?i)^(M{0,3})(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$'
    return bool(re.match(roman_numeral_pattern, s.strip()))


def get_header_level(paragraph: Tag) -> Optional[int]:
    """Return the level of the header (1 for h1, 2 for h2, etc.), or 0 if not a header."""
    # Check for direct header tag
    if paragraph.name.startswith('h') and paragraph.name[1:].isdigit():
        return int(paragraph.name[1:])  # Extract the level from 'hX' or 'hXY'

    # Check for class name equivalent to header tags
    if hasattr(paragraph, 'attrs') and 'class' in paragraph.attrs:
        section_headers: List[str] = ['pre-title1', 'h']
        for cls in paragraph.attrs['class']:
            if cls.lower() in section_headers:
                return 0  # Equivalent to h0 effectively
            elif cls.lower().startswith('h') and cls[1:].isdigit():
                return int(cls[1:])  # Extract level from class name 'hX' or 'hXY'
    return None


def is_title(tag: Tag) -> bool:
    # # A title isn't a header
    # noinspection SpellCheckingInspection
    keywords: List[str] = ['title', 'chtitle', 'tochead', 'title1', 'h1_label']
    is_a_title: bool = (hasattr(tag, 'attrs') and 'class' in tag.attrs and
                        any(cls.lower().startswith(keyword) or cls.lower().endswith(keyword)
                            for cls in tag.attrs['class'] for keyword in keywords))
    return is_a_title


def is_header1_title(paragraph: Tag, h1_count: int) -> bool:
    header_level: int = get_header_level(paragraph)
    if header_level == 1 and h1_count == 1:
        return True
    return False


def is_section_title(tag: Tag) -> bool:
    """Check if the tag is a title, heading, or chapter number."""
    if tag is None:
        return False

    header_lvl: int = get_header_level(tag)
    return is_title(tag) or header_lvl is not None or is_chapter_number(tag)


def is_chapter_number(paragraph: Tag) -> bool:
    # List of class names to check for chapter numbers
    # noinspection SpellCheckingInspection
    chapter_classes = ['chno', 'ch-num']
    # noinspection SpellCheckingInspection
    return (hasattr(paragraph, 'attrs') and 'class' in paragraph.attrs and
            any(cls in paragraph.attrs['class'] for cls in chapter_classes) and
            paragraph.text.isdigit())


def get_page_num(paragraph: Tag) -> str:
    # Try to get a page number - return it as a string instead of an int to accommodate roman numerals
    # Return None if none found on this paragraph
    tags: List[Tag] = paragraph.find_all(
        lambda x: (x.name == 'a' or x.name == 'span') and x.get('id')
        and (x['id'].startswith('page_') or (x['id'].startswith('p') and x['id'][1:].isdigit()))
    )
    page_num: Optional[str] = None
    if tags:
        # Extract the page number from the anchor or span tag id
        for tag in tags:
            page_id = tag.get('id')
            if page_id.startswith('page_'):
                page_num = page_id.split('_')[-1]
            elif page_id.startswith('p') and page_id[1:].isdigit():
                page_num = page_id[1:]  # Extract the digits after 'p'

    if not page_num:
        # Check for a page number embedded in the paragraph's id in
        # format 'pXXXX-' where XXXX is the page number with leading zeros
        page_id: str = paragraph.get('id')
        if page_id and page_id.startswith('p'):
            page_num = page_id[1:].split('-')[0]
            try:
                page_num = str(int(page_num))  # Remove leading zeros
            except ValueError:
                page_num = None
    return page_num


def enhance_title(text: str) -> str:
    text = text.strip()
    # If all caps but not a roman numeral and not first word before a space of a sentence roman numeral
    if text.isupper() and not is_roman_numeral(text):
        # If first word before a space is a roman numeral, leave that part as is
        first_word = text.split(' ', 1)[0]
        if is_roman_numeral(first_word) and first_word != text:
            text = first_word + text[len(first_word):].title()
        else:
            # If all caps, title case
            text = text.title()
        # Replace 'S with 's after title casing
        text = text.replace("'S", "'s")
        text = text.replace("’S", "’s")
    return text


def is_sup_first_content(tag, sup_tag):
    for content in tag.contents:
        if isinstance(content, str) and not content.strip():
            # Skip empty or whitespace-only strings
            continue
        # Check if the content is exactly the <sup> tag or contains it
        in_first_content: bool = content == sup_tag or (isinstance(content, Tag) and sup_tag in content.descendants)
        # Is sup both in the first content and also literally the text matches (proving it's a footnote)?
        return in_first_content and sup_tag.text.strip() == content.text.strip()
    return False


def recursive_yield_tags(tag: Tag, remove_footnotes: bool = False) -> Iterator[Tag]:
    invalid_children: List[str] = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8']
    # If the tag has no <p> tags or header tags under it and contains text, yield it
    # Unless it is a div tag. The Haystack HTML parse doesn't always handle those right, so
    # Dig one level deeper.
    if not tag.name == 'div' and tag.get_text(strip=True) and not tag.find(invalid_children):
        # Make a deep copy of the tag to avoid modifying the original
        tag_copy: Tag = deepcopy(tag)
        # Clean up of paragraph text
        for br in tag_copy.find_all('br'):
            br.insert_after(' ')
        # Remove footnotes - but not if the sup tag is at the start of the paragraph
        # Iterate over each <sup> tag
        if remove_footnotes:
            for fn in tag_copy.find_all('sup'):
                if not is_sup_first_content(tag_copy, fn):
                    # Remove the <sup> tag if it's not within the first content
                    fn.extract()

        yield tag_copy
    else:
        # Recursively go through the children of the current tag
        for child in tag.children:
            if isinstance(child, Tag):
                # Yield the child tags that meet the criteria
                yield from recursive_yield_tags(child, remove_footnotes=remove_footnotes)


def get_chapter_info(tags: List[Tag],
                     h1_tags: List[Tag],
                     h2_tags: List[Tag],
                     h3_tags: List[Tag]) -> Tuple[str, int, str]:
    if not tags:
        return "", 0, ""
    # Get the chapter title from the tag
    chapter_title: str = ""
    # Search for the chapter title within the tags that come before the first paragraph tag (that isn't
    # stylized to look like a header tag)
    # Use is_title to check for a specific title tag
    # If that fails you can use get_header_level to look for either a h1 or h2 tag but ONLY if that is the sole
    # h1 or h2 tag in the whole section.
    # There may be more than one title (like a subtitle) and you'll want to combine them via ": " separators.
    # Use enhance_title to clean up the title text.
    # Once you find your first paragraph that isn't a title or header, you can assume you've got the full title.

    # Create iterator using recursive_yield_tags
    # Count h1 tags
    # h1_tags: List[Tag] = top_tag.find_all('h1')
    # Remove any h1 tags that have class 'ch_num'
    h1_tags = [tag for tag in h1_tags if not is_chapter_number(tag) and not is_title(tag)]
    h1_tag_count: int = len(h1_tags)
    h2_tag_count: int = len(h2_tags)
    h3_tag_count: int = len(h3_tags)
    chapter_number: int = 0
    tags_to_delete: List[int] = []
    first_page_num: str = ""
    for i, tag in enumerate(tags):
        first_page_num = get_page_num(tag) or first_page_num
        if is_title(tag):
            # This tag is used in the title, so we need to delete the tag from the list of tags
            tags_to_delete.append(i)
            title_text = enhance_title(tag.text)
            if chapter_title:
                chapter_title += ": " + title_text
            else:
                chapter_title = title_text
        elif is_chapter_number(tag):
            tags_to_delete.append(i)
            chapter_number = int(tag.text.strip())
        elif chapter_title == "" and tag.name != 'p':
            # Check for a header tag that isn't a title
            if h1_tag_count == 1 and get_header_level(tag) == 1:
                # Using an H1 tag as a chapter title
                tags_to_delete.append(i)
                title_text = enhance_title(tag.text)
                chapter_title = title_text
            elif h1_tag_count == 0:
                if h2_tag_count == 1 and get_header_level(tag) == 2:
                    # Using an H2 tag as a chapter title
                    tags_to_delete.append(i)
                    title_text = enhance_title(tag.text)
                    chapter_title = title_text
                elif h3_tag_count == 1 and get_header_level(tag) == 3:
                    # I don't think this ever happens, but using a h3 tag as a chapter title
                    tags_to_delete.append(i)
                    title_text = enhance_title(tag.text)
                    chapter_title = title_text
        elif tag.name == 'p' and not is_chapter_number(tag):
            # We allow a couple of paragraphs before the title for quotes and such
            if chapter_title or i > 2:
                break

    # Delete the tags that were used in the title
    for i in sorted(tags_to_delete, reverse=True):
        del tags[i]

    return chapter_title, chapter_number, first_page_num


class HTMLParser:
    def __init__(self, html: str, meta_data: dict[str, str], min_paragraph_size: int = 300,
                 double_notes: bool = False, remove_footnotes: bool = True):
        self._item_html = html
        self._min_paragraph_size = min_paragraph_size
        self._docs_list: List[ByteStream] = []
        self._meta_list: List[Dict[str, str]] = []
        self._item_soup: BeautifulSoup = BeautifulSoup(self._item_html, 'html.parser')
        self._total_text: str = ""
        self._chapter_title: str = ""
        self._meta_data: dict[str, str] = meta_data
        self._remove_footnotes: bool = remove_footnotes
        # If True, chapters and sections named 'notes' will have double the minimum paragraph size
        # This is because notes are often very short,
        # and we want to keep them together to not dominate a semantic search
        self._double_notes: bool = double_notes

    def total_text_length(self) -> int:
        return len(self._total_text)

    @property
    def chapter_title(self):
        return self._chapter_title

    def run(self) -> Tuple[List[ByteStream], List[Dict[str, str]]]:
        temp_docs: List[ByteStream] = []
        temp_meta: List[Dict[str, str]] = []
        combined_paragraph: str = ""
        para_num: int = 0
        page_num: str
        headers: Dict[int, str] = {}  # Track headers by level
        j: int
        combined_chars: int = 0
        # convert item_soup to a list of tags using recursive_yield_tags
        tags: List[Tag] = list(recursive_yield_tags(self._item_soup, remove_footnotes=self._remove_footnotes))
        h1_tags: List[Tag] = self._item_soup.find_all('h1')
        h2_tags: List[Tag] = self._item_soup.find_all('h2')
        h3_tags: List[Tag] = self._item_soup.find_all('h3')
        self._chapter_title, chapter_number, page_num = get_chapter_info(tags, h1_tags, h2_tags, h3_tags)
        # Advance iter2 to be one ahead of iter1
        combine_headers: bool = False
        for j, tag in enumerate(tags):
            # prev_tag: Tag = tags[j - 1] if j > 0 else None
            next_tag: Tag = tags[j + 1] if j < len(tags) - 1 else None

            # If paragraph has a page number, update our page number
            page_num = get_page_num(tag) or page_num

            # Is it a chapter number tag?
            if is_chapter_number(tag):
                continue
            elif get_header_level(tag) is not None:  # If it's a header (that isn't a h1 being used as a title)
                header_level = get_header_level(tag)
                header_text = enhance_title(tag.text)
                # If header level is h5 or greater, treat it as a paragraph but still start a new section
                if header_level >= 6:
                    # Transform the header tag to be a paragraph tag
                    tag.name = 'p'
                else:
                    # Remove any headers that are lower than the current one (change of section)
                    headers = {level: text for level, text in headers.items() if level <= header_level}
                    # Save off header info
                    if header_text:
                        if not combine_headers or header_level not in headers:
                            headers[header_level] = header_text
                            combine_headers = True
                        else:
                            headers[header_level] = headers[header_level] + ": " + header_text
                            combine_headers = True

                    continue

            combine_headers = False
            # If we have no chapter title, check if there is a 0 level header
            if not self._chapter_title and headers and 0 in headers:
                self._chapter_title = headers[0]

            # Get top level header
            top_header_level: int = 0
            if headers:
                top_header_level = min(headers.keys())

            min_paragraph_size: int = self._min_paragraph_size

            # If headers are present, adjust the minimum paragraph size for notes
            if ((self._chapter_title and self._chapter_title.lower() == "notes")
                    or (headers and headers[top_header_level].lower() == "notes")):
                if self._double_notes:
                    min_paragraph_size = self._min_paragraph_size * 2

            p_str: str = str(tag)  # p.text.strip()
            p_str_chars: int = len(tag.text)

            # If the combined paragraph is less than the minimum size combine it with the next paragraph
            if combined_chars + p_str_chars < min_paragraph_size:
                # However, if the next pargraph is a header, we want to start a new paragraph
                # Unless the header came just after another header, in which case we want to combine them
                if is_section_title(next_tag) and not is_section_title(tag):
                    # Next paragraph is a header (and the current isn't), so break the paragraph here
                    p_str = combined_paragraph + "\n" + p_str
                    p_str_chars += combined_chars
                    combined_paragraph = ""
                    combined_chars = 0
                elif next_tag is None:
                    # If it's the last paragraph, then process this paragraph
                    combined_paragraph += "\n" + p_str
                    combined_chars += p_str_chars
                    p_str = combined_paragraph
                else:
                    # Combine this paragraph with the previous ones
                    combined_paragraph += "\n" + p_str
                    combined_chars += p_str_chars
                    continue
            else:
                p_str = combined_paragraph + "\n" + p_str
                p_str_chars += combined_chars
                combined_paragraph = ""
                combined_chars = 0
            para_num += 1
            self._total_text += p_str
            p_html: str = f"<html><head><title>Converted Epub</title></head><body>{p_str}</body></html>"
            byte_stream: ByteStream = ByteStream(p_html.encode('utf-8'))
            paragraph_meta_data: Dict[str, str] = {}
            paragraph_meta_data.update(self._meta_data)
            paragraph_meta_data["paragraph_#"] = str(para_num)
            # Page information
            if page_num:
                paragraph_meta_data["page_#"] = str(page_num)

            # Chapter information
            if self._chapter_title:
                paragraph_meta_data["chapter_title"] = self._chapter_title
            if chapter_number:
                paragraph_meta_data["chapter_#"] = str(chapter_number)

            # Include headers in the metadata
            for level, text in headers.items():
                if level == top_header_level:
                    paragraph_meta_data["section_name"] = text
                else:
                    paragraph_meta_data["subsection_name"] = paragraph_meta_data.get("subsection_name", "") + (
                        ": " + text if "subsection_name" in paragraph_meta_data else text)

            # self._print_verbose(meta_node)
            temp_docs.append(byte_stream)
            temp_meta.append(paragraph_meta_data)

        return temp_docs, temp_meta
