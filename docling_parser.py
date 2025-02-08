from copy import deepcopy
from bs4 import BeautifulSoup, Tag
from typing import List, Dict, Tuple, Iterator, Optional, Union
import re
# noinspection PyPackageRequirements
from haystack.dataclasses import ByteStream
from parse_utils import enhance_title
from docling.document_converter import DocumentConverter, ConversionResult
from docling_core.types import DoclingDocument
from docling_core.types.doc.document import SectionHeaderItem, ListItem, TextItem, PageItem
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


def clean_text(p_str):
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


def is_text_break(text: Union[SectionHeaderItem, ListItem, TextItem]) -> bool:
    return is_page_header(text) or is_section_header(text) or is_footnote(text)


def is_page_not_text(text: Union[SectionHeaderItem, ListItem, TextItem]) -> bool:
    return text.label not in ["text", "list_item", "formula"]


def is_page_text(text: Union[SectionHeaderItem, ListItem, TextItem]) -> bool:
    return not is_page_not_text(text)


def is_ends_with_punctuation(text: str) -> bool:
    return text.endswith(".") or text.endswith("?") or text.endswith("!")


def is_bottom_note(text: str) -> bool:
    # Check for · at the beginning of the line. This is often how OCR represents footnote number.
    if text.startswith("·") and not text.startswith("· "):
        return True
    return bool(re.match(r"^\d+[^\s].*", text))


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
                or is_page_header(item)
                or is_footnote(item)
                or is_bottom_note(item.text))


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


def combine_paragraphs(p1_str, p2_str):
    # If the paragraph ends without final punctuation, combine it with the next paragraph
    if is_sentence_end(p1_str):
        return p1_str + "\n" + p2_str
    else:
        return p1_str + " " + p2_str


def is_roman_numeral(s: str) -> bool:
    roman_numeral_pattern = r'(?i)^(M{0,3})(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$'
    return bool(re.match(roman_numeral_pattern, s.strip()))


class DoclingParser:
    def __init__(self, doc: DoclingDocument,
                 meta_data: dict[str, str],
                 min_paragraph_size: int = 300,
                 start_page: Optional[int] = None,
                 end_page: Optional[int] = None):
        self._doc: DoclingDocument = doc
        self._min_paragraph_size: int = min_paragraph_size
        self._docs_list: List[ByteStream] = []
        self._meta_list: List[Dict[str, str]] = []
        self._meta_data: dict[str, str] = meta_data
        self._start_page: Optional[int] = start_page
        self._end_page: Optional[int] = end_page

    def run(self) -> Tuple[List[ByteStream], List[Dict[str, str]]]:
        temp_docs: List[ByteStream] = []
        temp_meta: List[Dict[str, str]] = []
        combined_paragraph: str = ""
        para_num: int = 0
        i: int
        combined_chars: int = 0
        section_name: str = ""
        page_no: Optional[int] = None

        # Before we begin, we need to find all footnotes and move them to the end of the texts list
        # This is because footnotes are often interspersed with the text, and we want to process them all at once
        # Do this as a single list comprehension
        texts = list(self._doc.texts)
        footnotes = [text for text in texts if is_footnote(text)]
        bottom_notes = [text for text in texts if is_bottom_note(text.text)]
        filtered_texts = [text for text in texts if not is_footnote(text) and not is_bottom_note(text.text)]

        for i, text in enumerate(texts):
            # Deal with page number
            if page_no is None or combined_paragraph == "":
                page_no = text.prov[0].page_no

            # Check if paragraph is in valid range
            if self._start_page is not None and page_no is not None and page_no < self._start_page:
                page_no = None
                continue
            if self._end_page is not None and page_no is not None and page_no > self._end_page:
                break

            text.text = text.text.encode('utf-8').decode('utf-8')
            # prev_tag: Tag = tags[j - 1] if j > 0 else None
            next_text = get_next_text(texts, i)

            # Update the section header
            if is_section_header(text):
                section_name = text.text
                continue

            # Skip conditions
            if is_page_footer(text): continue
            if is_page_header(text): continue
            if is_footnote(text): continue
            if is_page_not_text(text): pass
            if is_bottom_note(text.text): continue
            if is_roman_numeral(text.text): continue

            p_str = str(text.text).strip()
            p_str = re.sub(r'\s+', ' ', p_str).strip()
            p_str_chars = len(p_str)

            p_str = re.sub(r"([.!?]) '", r"\1'", p_str)
            p_str = re.sub(r'([.!?]) "', r'\1"', p_str)
            p_str = re.sub(r'\s+\)', ')', p_str)
            p_str = re.sub(r'\s+\]', ']', p_str)
            p_str = re.sub(r'\s+\}', '}', p_str)
            p_str = re.sub(r'\s+,', ',', p_str)
            p_str = re.sub(r'\(\s+', '(', p_str)
            p_str = re.sub(r'\[\s+', '[', p_str)
            p_str = re.sub(r'\{\s+', '{', p_str)
            p_str = re.sub(r'(?<=\s)\.([a-zA-Z])', r'\1', p_str)
            p_str = re.sub(r'\s+\.', '.', p_str)

            # Remove footnote numbers at end of a sentence. Check for a digit at the end and drop it
            # until there are no more digits or the sentence is now a valid end of a sentence.
            while p_str and p_str[-1].isdigit() and not is_sentence_end(p_str):
                p_str = p_str[:-1].strip()

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
            elif combined_chars + p_str_chars < self._min_paragraph_size:
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
                    combined_paragraph = combine_paragraphs(combined_paragraph, p_str)
                    combined_chars += p_str_chars
                    continue
            else:
                p_str = combine_paragraphs(combined_paragraph, p_str)
                p_str_chars += combined_chars
                combined_paragraph = ""
                combined_chars = 0

            p_str = clean_text(p_str)
            if p_str:  # Only create entry if we have content
                para_num += 1
                byte_stream = ByteStream(p_str.encode('utf-8'))
                meta_data = {}
                meta_data.update(self._meta_data)
                meta_data["paragraph_#"] = str(para_num)
                meta_data["section_name"] = section_name
                meta_data["page_#"] = str(page_no)
                page_no = None
                temp_docs.append(byte_stream)
                temp_meta.append(meta_data)

        return temp_docs, temp_meta
