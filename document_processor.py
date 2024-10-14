# Pytorch imports
import torch
# EPUB imports
from bs4 import BeautifulSoup, Tag
from ebooklib import epub, ITEM_DOCUMENT
# Haystack imports
# noinspection PyPackageRequirements
from haystack import Pipeline, Document
# noinspection PyPackageRequirements
from haystack.components.routers import ConditionalRouter
# noinspection PyPackageRequirements
from haystack.dataclasses import ByteStream
# noinspection PyPackageRequirements
from haystack.components.preprocessors import DocumentCleaner
# noinspection PyPackageRequirements
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
# noinspection PyPackageRequirements
from haystack.components.converters import HTMLToDocument
# noinspection PyPackageRequirements
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
# noinspection PyPackageRequirements
from haystack.utils import ComponentDevice, Device
# noinspection PyPackageRequirements
from haystack.document_stores.types import DuplicatePolicy
# noinspection PyPackageRequirements
from haystack.utils.auth import Secret
# Other imports
from typing import List, Optional, Dict, Any, Tuple, Iterator
from pathlib import Path
from typing_extensions import Set
from generator_model import get_secret
from doc_content_checker import skip_content
from custom_haystack_components import CustomDocumentSplitter, RemoveIllegalDocs, FinalDocCounter, DuplicateChecker
import csv
import re
from copy import deepcopy
from itertools import tee


# Helper functions for processing EPUB or HTML content
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
    keywords: List[str] = ['title', 'chtitle', 'tochead']
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


def get_page_number(paragraph: Tag) -> str:
    # Try to get a page number - return it as a string instead of an int to accommodate roman numerals
    # Return None if none found on this paragraph
    page_anchors: List[Tag] = paragraph.find_all('a', id=lambda x: x and x.startswith('page_'))
    page_number: Optional[str] = None
    if page_anchors:
        # Extract the page number from the anchor tag id
        for anchor in page_anchors:
            page_id = anchor.get('id')
            page_number = page_id.split('_')[-1]
    return page_number


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


def is_roman_numeral(s: str) -> bool:
    roman_numeral_pattern = r'(?i)^(M{0,3})(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$'
    return bool(re.match(roman_numeral_pattern, s.strip()))


def recursive_yield_tags(tag: Tag) -> Iterator[Tag]:
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
        yield tag_copy
    else:
        # Recursively go through the children of the current tag
        for child in tag.children:
            if isinstance(child, Tag):
                # Yield the child tags that meet the criteria
                yield from recursive_yield_tags(child)


def get_chapter_info(top_tag: BeautifulSoup) -> Tuple[str, int]:
    # Get the chapter title from the tag
    chapter_title: str = ""
    # Search for the chapter title within the tags that come before the first paragraph tag (that isn't
    # stylized to look like a header tag)
    # Use is_title to check for a specific title tag
    # If that fails you can use get_header_level to look for either an h1 or h2 tag but ONLY if that is the sole
    # h1 or h2 tag in the whole section.
    # There may be more than one title (like a subtible) and you'll want to combine them via ": " separators.
    # Use enhance_title to clean up the title text.
    # Once you find your first paragraph that isn't a title or header, you can assume you've got the full title.

    # Create iterator using recursive_yield_tags
    tags_iter: Iterator[Tag] = recursive_yield_tags(top_tag)
    # Count h1 tags
    h1_tags: List[Tag] = top_tag.find_all('h1')
    # Remove any h1 tags that have class 'ch_num'
    h1_tags = [tag for tag in h1_tags if not is_chapter_number(tag) and not is_title(tag)]
    h1_tag_count: int = len(h1_tags)
    h2_tag_count: int = len(top_tag.find_all('h2'))
    chapter_number: int = 0
    for i, tag in enumerate(tags_iter):
        if is_title(tag):
            title_text = enhance_title(tag.text)
            if chapter_title:
                chapter_title += ": " + title_text
            else:
                chapter_title = title_text
        elif is_chapter_number(tag):
            chapter_number = int(tag.text.strip())
            continue
        elif get_header_level(tag) == 1 and h1_tag_count == 1 and not chapter_title:
            title_text = enhance_title(tag.text)
            chapter_title = title_text
        elif tag.name == 'p' and not is_chapter_number(tag):
            # We allow a couple of paragraphs before the title for quotes and such
            if i > 2:
                break

    return chapter_title, chapter_number


class DocumentProcessor:
    def __init__(self,
                 file_or_folder_path: str,
                 table_name: str = 'haystack_pgvector_docs',
                 recreate_table: bool = False,
                 postgres_user_name: str = 'postgres',
                 postgres_password: str = None,
                 postgres_host: str = 'localhost',
                 postgres_port: int = 5432,
                 postgres_db_name: str = 'postgres',
                 skip_content_func: Optional[callable] = None,
                 min_section_size: int = 1000,
                 min_paragraph_size: int = 500,
                 embedder_model_name: Optional[str] = None,
                 verbose: bool = False
                 ) -> None:

        # Instance variables
        self._table_name: str = table_name
        self._recreate_table: bool = recreate_table
        self._min_section_size = min_section_size
        self._embedder_model_name: Optional[str] = embedder_model_name
        self._sentence_embedder: Optional[SentenceTransformersDocumentEmbedder] = None
        self._min_paragraph_size: int = min_paragraph_size
        self._skip_content: Optional[callable] = skip_content_func
        self._verbose: bool = verbose

        # File paths
        self._file_or_folder_path: str = file_or_folder_path  # New instance variable

        # Determine if the path is a file or directory
        if Path(self._file_or_folder_path).is_file():
            self._is_directory = False
        elif Path(self._file_or_folder_path).is_dir():
            self._is_directory = True
        else:
            raise ValueError("The provided path must be a valid file or directory.")

        # GPU or CPU
        self._has_cuda: bool = torch.cuda.is_available()
        self._torch_device: torch.device = torch.device("cuda" if self._has_cuda else "cpu")
        self._component_device: ComponentDevice = ComponentDevice(Device.gpu() if self._has_cuda else Device.cpu())

        # Passwords and connection strings
        if postgres_password is None:
            raise ValueError("Postgres password must be provided")
        # PG_CONN_STR="postgresql://USER:PASSWORD@HOST:PORT/DB_NAME
        self._postgres_connection_str: str = (f"postgresql://{postgres_user_name}:{postgres_password}@"
                                              f"{postgres_host}:{postgres_port}/{postgres_db_name}")

        # Sections to skip
        self._sections_to_skip: Dict[str, Set[str]] = self._load_sections_to_skip()
        for book_title, sections in self._sections_to_skip.items():
            self._print_verbose(f"Skipping sections for book '{book_title}': {sections}")
        self._print_verbose("Initializing document store")

        self._document_store: Optional[PgvectorDocumentStore] = None
        self._doc_convert_pipeline: Optional[Pipeline] = None
        self._initialize_document_store()

    @property
    def verbose(self) -> bool:
        return self._verbose

    @verbose.setter
    def verbose(self, value: bool) -> None:
        self._verbose = value

    @property
    def context_length(self) -> Optional[int]:
        """
        Get the context length of the sentence embedder model.

        Returns:
            Optional[int]: The maximum context length of the sentence embedder model, if available.
        """
        self._setup_embedder()
        if self._sentence_embedder is not None and self._sentence_embedder.embedding_backend is not None:
            return self._sentence_embedder.embedding_backend.model.get_max_seq_length()
        else:
            return None

    @property
    def embed_dims(self) -> Optional[int]:
        """
        Get the embedding dimensions of the sentence embedder model.

        Returns:
            Optional[int]: The embedding dimensions of the sentence embedder model, if available.
        """
        self._setup_embedder()
        if self._sentence_embedder is not None and self._sentence_embedder.embedding_backend is not None:
            return self._sentence_embedder.embedding_backend.model.get_sentence_embedding_dimension()
        else:
            return None

    def _load_sections_to_skip(self) -> Dict[str, Set[str]]:
        sections_to_skip: Dict[str, Set[str]] = {}
        if self._is_directory:
            csv_path = Path(self._file_or_folder_path) / "sections_to_skip.csv"
        else:
            # Get the directory of the file and then look for the csv file in that directory
            csv_path = Path(self._file_or_folder_path).parent / "sections_to_skip.csv"

        if csv_path.exists():
            with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
                reader: csv.DictReader[str] = csv.DictReader(csvfile)
                row: dict[str, str]
                for row in reader:
                    book_title: str = row['Book Title'].strip()
                    section_title: str = row['Section Title'].strip()
                    if book_title and section_title:
                        if book_title not in sections_to_skip:
                            sections_to_skip[book_title] = set()
                        sections_to_skip[book_title].add(section_title)

            # Count total sections to skip across all books
            skip_count: int = sum(len(sections) for _, sections in sections_to_skip.items())
            self._print_verbose(f"Loaded {skip_count} sections to skip.")
        else:
            self._print_verbose("No sections_to_skip.csv file found. Processing all sections.")

        return sections_to_skip

    def _print_verbose(self, *args, **kwargs) -> None:
        if self._verbose:
            print(*args, **kwargs)

    def _setup_embedder(self) -> None:
        if self._sentence_embedder is None:
            if self._embedder_model_name is not None:
                self._sentence_embedder = SentenceTransformersDocumentEmbedder(model=self._embedder_model_name,
                                                                               device=self._component_device,
                                                                               trust_remote_code=True)
            else:
                self._sentence_embedder = SentenceTransformersDocumentEmbedder(device=self._component_device)

            if hasattr(self._sentence_embedder, 'warm_up'):
                self._sentence_embedder.warm_up()

    def draw_pipeline(self) -> None:
        """
        Draw and save visual representations of the document conversion pipelines.
        """
        if self._doc_convert_pipeline is not None:
            self._doc_convert_pipeline.draw(Path("Document Conversion Pipeline.png"))

    def _load_files(self) -> Iterator[Tuple[ByteStream, Dict[str, str]]]:
        if self._is_directory:
            for file_path in Path(self._file_or_folder_path).glob('*.epub'):
                docs, meta = self._load_epub(str(file_path))
                yield docs, meta
        else:
            docs, meta = self._load_epub(self._file_or_folder_path)
            yield docs, meta

    def _load_epub(self, file_path: str) -> Tuple[List[ByteStream], List[Dict[str, str]]]:
        docs_list: List[ByteStream] = []
        meta_list: List[Dict[str, str]] = []
        included_sections: List[str] = []
        book: epub.EpubBook = epub.read_epub(file_path)
        self._print_verbose()
        self._print_verbose(f"Book Title: {book.title}")
        section_num: int = 1
        book_meta_data: Dict[str, str] = {
            "book_title": book.title,
            "file_path": file_path
        }
        i: int
        for i, item in enumerate(book.get_items_of_type(ITEM_DOCUMENT)):
            if i == 9 and book.title == "The Poverty of Historicism":
                pass
            item_meta_data: Dict[str, str] = {
                "item_num": str(section_num),
                "item_id": item.id,
            }
            item_html: str = item.get_body_content().decode('utf-8')
            # print(section_html)
            item_soup: BeautifulSoup = BeautifulSoup(item_html, 'html.parser')
            h1_tag_count: int = len(item_soup.find_all('h1'))
            chapter_title: str = ""
            new_chapter_title: str
            chapter_number: int = 0
            new_chapter_title, chapter_number = get_chapter_info(item_soup)
            # print()
            # print("HTML:")
            # print(section_soup)
            # print()
            temp_docs: List[ByteStream] = []
            temp_meta: List[Dict[str, str]] = []
            total_text: str = ""
            combined_paragraph: str = ""
            para_num: int = 0
            page_number: str = ""
            headers: Dict[int, str] = {}  # Track headers by level
            j: int
            combined_chars: int = 0
            # Setup iterators
            tags = recursive_yield_tags(item_soup)
            iter1, iter2 = tee(tags)
            # Advance iter2 to be one ahead of iter1
            next(iter2, None)
            for j, tag in enumerate(iter1):

                if item.id == 'notes1' and para_num == 0:
                    pass

                try:
                    next_tag = next(iter2)
                except StopIteration:
                    next_tag = None  # This handles the final tag case

                # If paragraph has a page number, update our page number
                page_number = get_page_number(tag) or page_number

                # Check for title information
                if is_title(tag) or is_header1_title(tag, h1_tag_count):
                    continue
                # Is it a chapter number tag?
                elif is_chapter_number(tag):
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
                        headers = {level: text for level, text in headers.items() if level < header_level}
                        # Save off header info
                        if header_text:
                            headers[header_level] = header_text
                        continue

                # Set metadata
                # Pick current title
                # chapter_title = (chapter_title or item.title)
                chapter_title = new_chapter_title or item.title

                if chapter_title != new_chapter_title:
                    pass
                elif chapter_title == new_chapter_title and chapter_title != "":
                    pass

                # If we have no chapter title, check if there is a 0 level header
                if not chapter_title and headers and 0 in headers:
                    chapter_title = headers[0]

                p_str: str = str(tag)  # p.text.strip()
                p_str_chars: int = len(tag.text)
                min_paragraph_size: int = self._min_paragraph_size

                # Get top level header
                top_header_level: int = 0
                if headers:
                    top_header_level = min(headers.keys())

                # If headers are present, adjust the minimum paragraph size for notes
                if ((chapter_title and chapter_title.lower() == "notes") or
                        (headers and headers[top_header_level].lower() == "notes")):
                    # If we're in the notes section, we want to combine paragraphs into larger sections
                    # This is because the notes are often very short, and we want to keep them together
                    # And also so that they don't dominate a semantic search
                    # We could just drop notes, but often they contain useful information
                    min_paragraph_size = self._min_paragraph_size * 2

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
                total_text += p_str
                p_html: str = f"<html><head><title>Converted Epub</title></head><body>{p_str}</body></html>"
                byte_stream: ByteStream = ByteStream(p_html.encode('utf-8'))
                paragraph_meta_data: Dict[str, str] = {}
                # Combine meta_node with book_meta_data and section_meta_data
                paragraph_meta_data.update(book_meta_data)
                paragraph_meta_data.update(item_meta_data)
                paragraph_meta_data["paragraph_num"] = str(para_num)

                # Page information
                if page_number:
                    paragraph_meta_data["page_number"] = str(page_number)

                # Chapter information
                if chapter_title:
                    paragraph_meta_data["chapter_title"] = chapter_title
                if chapter_number:
                    paragraph_meta_data["chapter_number"] = str(chapter_number)

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

            if (len(total_text) > self._min_section_size
                    and item.id not in self._sections_to_skip.get(book.title, set())):
                self._print_verbose(f"Book: {book.title}; Section {section_num}. Section Title: {chapter_title}. "
                                    f"Length: {len(total_text)}")
                docs_list.extend(temp_docs)
                meta_list.extend(temp_meta)
                included_sections.append(book.title + ", " + item.id)
                section_num += 1
            else:
                self._print_verbose(f"Book: {book.title}; Title: {chapter_title}. Length: {len(total_text)}. Skipped.")

        self._print_verbose(f"Sections included:")
        for item in included_sections:
            self._print_verbose(item)
        self._print_verbose()
        return docs_list, meta_list

    def _doc_converter_pipeline(self) -> None:
        self._setup_embedder()
        # Create the custom splitter
        custom_splitter: CustomDocumentSplitter = CustomDocumentSplitter(self._sentence_embedder,
                                                                         verbose=self._verbose,
                                                                         skip_content_func=self._skip_content)

        routes = [
            {
                "condition": "{{documents|length > 0}}",
                "output": "{{documents}}",
                "output_name": "has_documents",
                "output_type": List[Document],
            },
            {
                "condition": "{{documents|length <= 0}}",
                "output": "{{0}}",
                "output_name": "no_documents",
                "output_type": int,
            },
        ]
        router = ConditionalRouter(routes=routes)

        # Create the document conversion pipeline
        doc_convert_pipe: Pipeline = Pipeline()
        doc_convert_pipe.add_component("converter", HTMLToDocument())
        doc_convert_pipe.add_component("remove_illegal_docs", instance=RemoveIllegalDocs())
        doc_convert_pipe.add_component("cleaner", DocumentCleaner())
        doc_convert_pipe.add_component("splitter", custom_splitter)
        doc_convert_pipe.add_component("duplicate_checker", DuplicateChecker(document_store=self._document_store))
        doc_convert_pipe.add_component("embedder", self._sentence_embedder)
        doc_convert_pipe.add_component("router", router)
        doc_convert_pipe.add_component("writer",
                                       DocumentWriter(document_store=self._document_store,
                                                      policy=DuplicatePolicy.OVERWRITE))
        doc_convert_pipe.add_component("final_counter", FinalDocCounter())

        doc_convert_pipe.connect("converter", "remove_illegal_docs")
        doc_convert_pipe.connect("remove_illegal_docs", "cleaner")
        doc_convert_pipe.connect("cleaner", "splitter")
        doc_convert_pipe.connect("splitter", "duplicate_checker")
        doc_convert_pipe.connect("duplicate_checker", "router")
        doc_convert_pipe.connect("router.has_documents", "embedder")
        doc_convert_pipe.connect("embedder", "writer")
        doc_convert_pipe.connect("writer.documents_written", "final_counter.documents_written")
        doc_convert_pipe.connect("router.no_documents", "final_counter.no_documents")

        self._doc_convert_pipeline = doc_convert_pipe

    def _initialize_document_store(self) -> None:
        def init_doc_store(force_recreate: bool = False) -> PgvectorDocumentStore:
            connection_token: Secret = Secret.from_token(self._postgres_connection_str)
            doc_store: PgvectorDocumentStore = PgvectorDocumentStore(
                connection_string=connection_token,
                table_name=self._table_name,
                embedding_dimension=self.embed_dims,
                vector_function="cosine_similarity",
                recreate_table=self._recreate_table or force_recreate,
                search_strategy="hnsw",
                hnsw_recreate_index_if_exists=True,
                hnsw_index_name=self._table_name + "_hnsw_index",
                keyword_index_name=self._table_name + "_keyword_index",
            )

            return doc_store

        document_store: PgvectorDocumentStore
        document_store = init_doc_store()
        self._document_store = document_store

        doc_count: int = document_store.count_documents()
        self._print_verbose("Document Count: " + str(doc_count))
        self._print_verbose("Loading document file")

        # Iterate over the document and metadata pairs as they are loaded
        total_written: int = 0
        self._doc_converter_pipeline()

        source: List[ByteStream]
        meta: List[Dict[str, str]]

        for source, meta in self._load_files():
            title: str = (meta[0].get('book_title', '') + " " +
                          meta[0].get('chapter_title', '') + " " +
                          meta[0].get('section_name', '')).strip()
            self._print_verbose(f"Processing document: {title} ")

            # If needed, you can batch process here instead of processing one by one
            # Pass the source and meta to the document conversion pipeline
            results: Dict[str, Any] = self._doc_convert_pipeline.run({"converter": {"sources": source, "meta": meta}})
            written = results["final_counter"]["documents_written"]
            total_written += written
            self._print_verbose(f"Wrote {written} documents for {meta[0]['book_title']}.")

        self._print_verbose(f"Finished writing documents to document store. Final document count: {total_written}")


def main() -> None:
    # epub_file_path: str = "documents/Karl Popper - All Life is Problem Solving-Taylor and Francis.epub"
    # epub_file_path: str = "documents/Karl Popper - The Myth of the Framework-Taylor and Francis.epub"
    # epub_file_path: str = "documents/Karl Popper - Conjectures and Refutations-Taylor and Francis (2018).epub"
    # epub_file_path: str = "documents/Karl Popper - The Open Society and Its Enemies-Princeton University Press (2013).epub"  # noqa: E501
    # epub_file_path: str = "documents/Karl Popper - The World of Parmenides-Taylor & Francis (2012).epub"
    # epub_file_path: str = "documents/Karl Popper - The Poverty of Historicism-Routledge (2002).epub"
    epub_file_path: str = "documents"
    postgres_password = get_secret(r'D:\Documents\Secrets\postgres_password.txt')
    # noinspection SpellCheckingInspection
    processor: DocumentProcessor = DocumentProcessor(
        table_name="book_archive",
        recreate_table=False,
        embedder_model_name="BAAI/llm-embedder",
        file_or_folder_path=epub_file_path,
        postgres_user_name='postgres',
        postgres_password=postgres_password,
        postgres_host='localhost',
        postgres_port=5432,
        postgres_db_name='postgres',
        skip_content_func=skip_content,
        min_section_size=3000,
        min_paragraph_size=300,
        verbose=True
    )

    # Draw images of the pipelines
    if processor.verbose:
        processor.draw_pipeline()
        print("Sentence Embedder Dims: " + str(processor.embed_dims))
        print("Sentence Embedder Context Length: " + str(processor.context_length))


if __name__ == "__main__":
    main()

# TODO: Get code to work with upgraded Haystack version
# TODO: There has got to be a way to get actual page numbers from an EPUB file
# TODO: Add PDF support (and maybe other document types)

# Possible new approach:
# iter1, iter2 = tee(section_soup.descendants)
# next(iter2, None)
#
# for j, (p, next_p) in enumerate(zip(iter1, iter2)):
#     if isinstance(p, Tag) and p.get_text(strip=True):
#         if p.find(valid_tags):
#             # This tag contains a paragraph, so skip it
#             continue
#     elif not isinstance(p, Tag):
#         continue
