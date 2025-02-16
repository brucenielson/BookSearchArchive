from typing import List, Dict, Tuple, Optional, Union
import re
# noinspection PyPackageRequirements
from haystack.dataclasses import ByteStream
from docling_core.types import DoclingDocument
from docling_core.types.doc import CoordOrigin
from docling_core.types.doc.document import SectionHeaderItem, ListItem, TextItem, DocItem
import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
# Download the words corpus if needed
nltk.download('words')
nltk.download('wordnet')
nltk.download('omw-1.4')
words_list: set = set(nltk.corpus.words.words())
# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


def is_valid_word(word):
    """Check if a word is valid by comparing it directly and via stemming."""
    stem = stemmer.stem(word)
    if (word.lower() in words_list
            or word in words_list):
        return True
    elif (stem in words_list
          or stem.lower() in words_list):
        return True
    # Check all lemmatizations of the word
    options = ['n', 'v', 'a', 'r', 's']
    for option in options:
        lemma = lemmatizer.lemmatize(word, pos=option)
        if lemma in words_list:
            return True
    # Check for custom lemmatizations
    suffixes = {
        "ability": "able",  # testability -> testable
        "ibility": "ible",  # possibility -> possible
        "iness": "y",         # happiness -> happy
        "ity": "e",          # creativity -> create
        "tion": "e",         # creation -> create
        "able": "",         # testable -> test
        "ible": "",         # possible -> poss
        "ing": "",          # running -> run
        "ed": "",           # tested -> test
        "s": ""             # tests -> test
    }
    for suffix, replacement in suffixes.items():
        if word.endswith(suffix):
            if suffix != 's':
                pass
            stripped_word = word[: -len(suffix)] + replacement
            if is_valid_word(stripped_word):
                return stripped_word

    return False


def combine_hyphenated_words(p_str):
    # This regular expression looks for cases where a dash separates two parts of a word
    # The idea is to combine the two parts and check if they form a valid word.
    def replace_dash(match):
        word1, word2 = match.group(1), match.group(2)
        combined = word1.strip() + word2.strip()
        first_word = p_str.strip().split(' ')[0]

        if combined == 'Rei7':
            pass

        if combined == 'Marsden':
            pass

        # does word 1 contain a space after the hyphen?
        if word2.startswith(" ") and is_valid_word(combined):
            # When there is a space after the hyphen, it is likely that the hyphen is separating two parts
            # of a single word
            return combined
        # else check for each part individually being a word. If so, this is probably a compound word
        elif is_valid_word(word1.strip()) and is_valid_word(word2.strip()):
            return word1.strip() + '-' + word2.strip()
        # else if the combined word is a valid word, then we probably had one word broken in two
        elif is_valid_word(combined):
            return combined  # Combine the parts if they form a valid word
        # if the combined word starts with a capital letter, then it is likely a proper noun. Combine the parts.
        # TODO: I can make this work better if I be sure this is a capital that isn't the first word of a sentence.
        elif combined[0].isupper() and not word2.strip()[0].isupper() and not is_valid_word(word2.strip()):
            return combined

        # Default - assume the hyphen is separating two words
        return word1.strip() + '-' + word2.strip()

    # Replace soft hyphen characters (­) with a regular dash
    p_str = p_str.replace("­", "-")
    # p_str = p_str.replace("- ", "-")

    # Look for dashes separating word parts (no spaces involved)
    p_str = re.sub(r'(\w+)-(\s?\w+)', replace_dash, p_str)

    return p_str


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


def is_list_item(text: Union[SectionHeaderItem, ListItem, TextItem]) -> bool:
    return text.label == "list_item"


def is_text_break(text: Union[SectionHeaderItem, ListItem, TextItem]) -> bool:
    return is_page_header(text) or is_section_header(text) or is_footnote(text)


def is_page_not_text(text: Union[SectionHeaderItem, ListItem, TextItem]) -> bool:
    return text.label not in ["text", "list_item", "formula"]


def is_page_text(text: Union[SectionHeaderItem, ListItem, TextItem]) -> bool:
    return not is_page_not_text(text)


def is_ends_with_punctuation(text: str) -> bool:
    return text.endswith(".") or text.endswith("?") or text.endswith("!")


def is_near_bottom(doc_item: DocItem, same_page_items: [DocItem], threshold: float = 0.3, debug: bool = False) -> bool:
    """
    Determine if a DocItem is near the bottom of its page.

    Parameters:
    - doc_item: The DocItem object containing provenance data with 'bbox'.
    - doc: The DoclingDocument containing all DocItems.
    - threshold: Distance in points from the bottom to consider as 'near the bottom'.

    Returns:
    - True if the DocItem is within the threshold from the bottom, False otherwise.
    """
    # Check if the DocItem has provenance data with a bounding box
    if hasattr(doc_item.prov[0], 'bbox'):
        bbox = doc_item.prov[0].bbox
    else:
        return False  # No bounding box available

    # Extract the coordinate origin and bounding box coordinates
    coord_origin = bbox.coord_origin
    x0, y0, x1, y1 = bbox.l, bbox.b, bbox.r, bbox.t

    # Print out the text, page number, and x0, y0, x1, y1 for each item on the same page
    if debug:
        for item in same_page_items:
            if hasattr(item.prov[0], 'bbox'):
                bbox = item.prov[0].bbox
                print(f"Text: {item.text}, Page: {item.prov[0].page_no}, x0: {bbox.l}, y0: {bbox.b}, x1: {bbox.r}, y1: {bbox.t}")

    # Find the maximum y1 value on the page
    page_top: float = max(item.prov[0].bbox.t for item in same_page_items if hasattr(item.prov[0], 'bbox'))
    # Find the min y1 value on the page
    page_bottom: float = min(item.prov[0].bbox.b for item in same_page_items if hasattr(item.prov[0], 'bbox'))
    page_size: float = page_top - page_bottom
    # Threshold is page_bottom + (size of page * threshold amount) (i.e. % of page to be considered the 'bottom')
    bottom_threshold: float = page_bottom + (page_size * threshold)

    if coord_origin == CoordOrigin.BOTTOMLEFT:
        # In this system, y1 is the distance from the top of the paragraph to the bottom of the page
        return y1 <= bottom_threshold
    elif coord_origin == CoordOrigin.TOPLEFT:
        # In this system, y1 is the distance from the top of the paragraph to the top of the page
        return y1 >= bottom_threshold
    else:
        raise ValueError("Unknown coordinate origin.")


def is_smaller_text(doc_item: DocItem, doc: DoclingDocument, threshold: float = 0.8) -> bool:
    """
    Determine if a DocItem's text is smaller than the average text size on its page.

    Parameters:
    - doc_item: The DocItem object containing provenance data with 'bbox'.
    - doc: The DoclingDocument containing all DocItems.
    - threshold: Ratio of the average text size to consider as 'smaller text'.

    Returns:
    - True if the DocItem's text is smaller than the average text size, False otherwise.
    """
    # Check if the DocItem has provenance data with a bounding box
    if hasattr(doc_item.prov[0], 'bbox'):
        bbox = doc_item.prov[0].bbox
    else:
        return False  # No bounding box available

    # Extract the bounding box coordinates
    x0, y0, x1, y1 = bbox.l, bbox.b, bbox.r, bbox.t

    # Calculate the area of the DocItem's bounding box
    doc_item_area = (x1 - x0) * (y1 - y0)

    # Filter doc_items that are on the same page
    same_page_items = [item for item in doc.texts if item.prov[0].page_no == doc_item.prov[0].page_no]

    # Calculate the average area of bounding boxes on the page
    total_area = sum(
        (item.prov[0].bbox.r - item.prov[0].bbox.l) * (item.prov[0].bbox.t - item.prov[0].bbox.b)
        for item in same_page_items if hasattr(item.prov[0], 'bbox')
    )
    num_items = sum(1 for item in same_page_items if hasattr(item.prov[0], 'bbox'))
    average_area = total_area / num_items if num_items > 0 else 0

    # Compare the DocItem's area to the average
    return doc_item_area < average_area * threshold

def is_bottom_note(text: Union[SectionHeaderItem, ListItem, TextItem], doc: DoclingDocument) -> bool:
    debug = False
    if text.text.startswith("8Cp. my 'The Rationality of Scientific Revolutions"):
        pass
    if 'Morgenstern was then the director' in text.text:
        if hasattr(text.prov[0], 'bbox'):
            bbox = text.prov[0].bbox
        else:
            bbox = None
        pass
    if text.text.startswith("2Lord Rutherford:"):
        pass
    if text.text.startswith("12See my 'The Rationality of Scientific Revolutions'"):
        pass
    if text.text.startswith("171 do not think that Norbert Wiener"):
        pass
    if text.text.startswith("2See L.Sc.D."):
        debug = True
        pass
    if text.text.startswith("11Hume's criticism of induction"):
        debug = True
        pass
    if text.text.startswith("12 I criticized Carnap"):
        debug = True
        pass
    if text.text.startswith("19'fhis idea is taken from the last sentence"):
        debug = True
        pass
    if text.text.startswith("2The assertion made here"):
        debug = True
        pass
    if text.text.startswith("10. Summing up o f"):
        debug = True
        pass
    if text.text.startswith("15. Philosophers  and  even  scientists"):
        debug = True
        pass
    if text.text.startswith("19'. But there were other problems which led to the task of defining this phrase"):
        debug = True
        pass
    if 'valid under the condition' in text.text:
        debug = True
        pass

    if text is None or not is_page_text(text):
        return False
    # Check for · at the beginning of the line. This is often how OCR represents footnote number.
    if text.text.startswith("·") and not text.text.startswith("· "):
        return True

    # Filter doc_items that are on the same page
    same_page_items: List[DocItem] = [item for item in doc.texts if item.prov[0].page_no == text.prov[0].page_no]

    # Check if this text starts with a digit
    if bool(re.match(r"^\d", text.text)):
        # Check if this is digits NOT followed by space or period - e.g. 1Hello is always a bottom note
        if bool(re.match(r"^\d+(?![ .]|\d|$)", text.text)):
            # However, don't invoke if we're right at the top of the page (try to be sure we combine
            # top of a page with previous paragraph that might have been split by a page break)
            if is_near_bottom(text, same_page_items, threshold=0.75, debug=debug):
                return True
        # Check if we're at the bottom of the page
        if is_near_bottom(text, same_page_items, threshold=0.5, debug=debug):
            # Check if this is three digits with the third digit being a 1 followed by a space
            # This is usually where the last 1 was supposed to be an 'I'.
            if bool(re.match(r"^\d{1,2}1 ", text.text)):
                return True
            if not is_list_item(text):
                return True

    return False

    # if is_near_bottom(text, doc) and bool(re.match(r"^\d+\S.*", text.text)):
    #     return True
    # if is_list_item(text) and bool(re.match(r"^\d+[^\d\s.]", text.text)):
    #     return True
    # if text.label == 'text' and bool(re.match(r"^\d+\S.*", text.text)):
    #     return True
    # return False


def is_sentence_end(text: str) -> bool:
    has_end_punctuation: bool = is_ends_with_punctuation(text)
    # Does it end with a closing bracket, quote, etc.?
    ends_with_bracket: bool = (text.endswith(")")
                               or text.endswith("]")
                               or text.endswith("}")
                               or text.endswith("\"")
                               or text.endswith("\'"))
    return (has_end_punctuation or
            (ends_with_bracket and is_ends_with_punctuation(text[0:-1])))


def is_text_item(item: Union[SectionHeaderItem, ListItem, TextItem]) -> bool:
    return not (is_section_header(item)
                or is_page_footer(item)
                or is_page_header(item))


def get_next_text(texts: List[Union[SectionHeaderItem, ListItem, TextItem]], i: int) \
        -> Optional[Union[ListItem, TextItem]]:
    # Seek through the list of texts to find the next text item using is_text_item
    # Should return None if no more text items are found
    for j in range(i + 1, len(texts)):
        if j < len(texts) and is_text_item(texts[j]):
            return texts[j]
    return None


def remove_extra_whitespace(text: str) -> str:
    # Remove extra whitespace in the middle of the text
    return ' '.join(text.split())


def combine_paragraphs(p1_str: str, p2_str: str):
    # If the paragraph ends without final punctuation, combine it with the next paragraph
    if is_sentence_end(p1_str):
        return p1_str + "\n" + p2_str
    else:
        return p1_str + " " + p2_str


def is_roman_numeral(s: str) -> bool:
    roman_numeral_pattern = r'(?i)^(M{0,3})(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$'
    return bool(re.match(roman_numeral_pattern, s.strip()))


def get_current_page(text: Union[SectionHeaderItem, ListItem, TextItem],
                     combined_paragraph: str,
                     current_page: Optional[int]) -> Optional[int]:
    return text.prov[0].page_no if current_page is None or combined_paragraph == "" else current_page


def should_skip_element(text: Union[SectionHeaderItem, ListItem, TextItem]) -> bool:
    return any([
        is_page_footer(text),
        is_page_header(text),
        is_roman_numeral(text.text)
    ])


def clean_text(p_str: str) -> str:
    p_str = str(p_str).strip()  # Convert text to a string and remove leading/trailing whitespace
    p_str = p_str.encode('utf-8').decode('utf-8')
    p_str = re.sub(r'\s+', ' ', p_str).strip()  # Replace multiple whitespace with single space
    p_str = re.sub(r"([.!?]) '", r"\1'", p_str)  # Remove the space between punctuation (.!?) and '
    p_str = re.sub(r'([.!?]) "', r'\1"', p_str)  # Remove the space between punctuation (.!?) and "
    p_str = re.sub(r'\s+\)', ')', p_str)  # Remove whitespace before a closing parenthesis
    p_str = re.sub(r'\s+]', ']', p_str)  # Remove whitespace before a closing square bracket
    p_str = re.sub(r'\s+}', '}', p_str)  # Remove whitespace before a closing curly brace
    p_str = re.sub(r'\s+,', ',', p_str)  # Remove whitespace before a comma
    p_str = re.sub(r'\(\s+', '(', p_str)  # Remove whitespace after an opening parenthesis
    p_str = re.sub(r'\[\s+', '[', p_str)  # Remove whitespace after an opening square bracket
    p_str = re.sub(r'\{\s+', '{', p_str)  # Remove whitespace after an opening curly brace
    p_str = re.sub(r'(?<=\s)\.([a-zA-Z])', r'\1',
                   p_str)  # Remove a period that follows a whitespace and comes before a letter
    p_str = re.sub(r'\s+\.', '.', p_str)  # Remove any whitespace before a period
    # Remove footnote numbers at end of a sentence. Check for a digit at the end and drop it
    # until there are no more digits or the sentence is now a valid end of a sentence.
    while p_str and p_str[-1].isdigit() and not is_sentence_end(p_str):
        p_str = p_str[:-1].strip()
    return p_str


class DoclingParser:
    def __init__(self, doc: DoclingDocument,
                 meta_data: dict[str, str],
                 min_paragraph_size: int = 300,
                 start_page: Optional[int] = None,
                 end_page: Optional[int] = None,
                 double_notes: bool = False):
        self._doc: DoclingDocument = doc
        self._min_paragraph_size: int = min_paragraph_size
        self._docs_list: List[ByteStream] = []
        self._meta_list: List[Dict[str, str]] = []
        self._meta_data: dict[str, str] = meta_data
        self._start_page: Optional[int] = start_page
        self._end_page: Optional[int] = end_page
        self._double_notes: bool = double_notes

    def run(self) -> Tuple[List[ByteStream], List[Dict[str, str]]]:
        temp_docs: List[ByteStream] = []
        temp_meta: List[Dict[str, str]] = []
        combined_paragraph: str = ""
        i: int
        combined_chars: int = 0
        para_num: int = 0
        section_name: str = ""
        page_no: Optional[int] = None
        first_note: bool = False

        texts = self._get_processed_texts()

        for i, text in enumerate(texts):
            next_text = get_next_text(texts, i)
            page_no = get_current_page(text, combined_paragraph, page_no)

            # Check if the current page is within the valid range
            if self._start_page is not None and page_no is not None and page_no < self._start_page:
                page_no = None
                continue
            if self._end_page is not None and page_no is not None and page_no > self._end_page:
                if self._double_notes and not first_note:
                    self._min_paragraph_size *= 2
                    first_note = True
                continue

            # Update section header if the element is a section header
            if is_section_header(text):
                section_name = text.text
                continue

            if should_skip_element(text):
                continue

            p_str = clean_text(text.text)
            p_str_chars = len(p_str)

            # If the paragraph does not end with final punctuation, accumulate it
            if not is_sentence_end(p_str):
                combined_paragraph = combine_paragraphs(combined_paragraph, p_str)
                combined_chars += p_str_chars
                continue

            # p_str ends with a sentence end; decide whether to process or accumulate it
            total_chars = combined_chars + p_str_chars
            if is_section_header(next_text):
                # Immediately process if the next text is a section header
                p_str = combine_paragraphs(combined_paragraph, p_str)
                combined_paragraph, combined_chars = "", 0
            elif total_chars < self._min_paragraph_size:
                # Not enough characters accumulated yet; decide based on next_text
                if next_text is None or (not is_page_text(next_text) and is_sentence_end(p_str)):
                    # End of document or next text item is not a text item and current paragraph ends with punctuation
                    # Process the paragraph and reset the accumulator even though this is a short paragraph
                    p_str = combine_paragraphs(combined_paragraph, p_str)
                    combined_paragraph, combined_chars = "", 0
                else:
                    # Combine with next paragraph
                    combined_paragraph = combine_paragraphs(combined_paragraph, p_str)
                    combined_chars = total_chars
                    continue
            else:
                # Sufficient characters: process the paragraph and reset the accumulator
                p_str = combine_paragraphs(combined_paragraph, p_str)
                combined_paragraph, combined_chars = "", 0

            p_str = combine_hyphenated_words(p_str)
            if p_str:  # Only add non-empty content
                para_num += 1
                self._add_paragraph(p_str, para_num, section_name, page_no, temp_docs, temp_meta)
                page_no = None

        return temp_docs, temp_meta

    def _get_processed_texts(self) -> List:
        # Before we begin, we need to find all footnotes and move them to the end of the texts list
        # This is because footnotes are often interspersed with the text, and we want to process them all at once
        # Split texts into regular content and notes (footnotes + bottom notes)
        regular = [t for t in self._doc.texts if not (is_footnote(t) or is_bottom_note(t, self._doc))]
        notes = [t for t in self._doc.texts if is_footnote(t) or is_bottom_note(t, self._doc)]
        return regular + notes

    def _add_paragraph(self, text: str, para_num: int, section: str,
                       page: Optional[int], docs: List[ByteStream], meta: List[Dict]):
        docs.append(ByteStream(text.encode('utf-8')))
        meta.append({
            **self._meta_data,
            # "paragraph_#": str(para_num),
            "section_name": section,
            "page_#": str(page)
        })
