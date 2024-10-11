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


# Helper functions for processing EPUB or HTML content
def get_header_level(paragraph: Tag) -> int:
    """Return the level of the header (1 for h1, 2 for h2, etc.), or 0 if not a header."""
    # Check for direct header tag
    if paragraph.name.startswith('h') and paragraph.name[1:].isdigit():
        return int(paragraph.name[1:])  # Extract the level from 'hX' or 'hXY'

    # Check for class name equivalent to header tags
    if hasattr(paragraph, 'attrs') and 'class' in paragraph.attrs:
        for cls in paragraph.attrs['class']:
            if cls.lower() == 'pre-title1':
                return 1  # Equivalent to h1
            if cls.lower().startswith('h') and cls[1:].isdigit():
                return int(cls[1:])  # Extract level from class name 'hX' or 'hXY'
    return 0


def is_title(paragraph: Tag, h1_count: int) -> bool:
    header_level: int = get_header_level(paragraph)
    if header_level == 1 and h1_count == 1:
        return True

    # noinspection SpellCheckingInspection
    keywords: List[str] = ['title', 'chtitle', 'tochead']

    return (hasattr(paragraph, 'attrs') and 'class' in paragraph.attrs and
            any(cls.lower().startswith(keyword) or cls.lower().endswith(keyword)
                for cls in paragraph.attrs['class'] for keyword in keywords))


def is_title_or_heading(paragraph: Tag, h1_count: int) -> bool:
    """Check if the paragraph is a title, heading, or chapter number."""
    if paragraph is None:
        return False

    header_lvl: int = get_header_level(paragraph)
    return is_title(paragraph, h1_count) or header_lvl > 0 or is_chapter_number(paragraph)


def is_chapter_number(paragraph: Tag) -> bool:
    # List of class names to check for chapter numbers
    # noinspection SpellCheckingInspection
    chapter_classes = ['chno', 'ch-num']
    # noinspection SpellCheckingInspection
    return (hasattr(paragraph, 'attrs') and 'class' in paragraph.attrs and
            any(cls in paragraph.attrs['class'] for cls in chapter_classes) and
            paragraph.text.isdigit())


class DocumentProcessor:
    """
    A class that implements a Retrieval-Augmented Generation (RAG) system using Haystack and Pgvector.

    This class provides functionality to set up and use a RAG system for question answering
    tasks on a given corpus of text, currently from an EPUB file. It handles document
    indexing, embedding, retrieval, and generation of responses using a language _model.

    The system uses a Postgres database with the Pgvector extension for efficient
    similarity search of embedded documents.

    Public Methods:
        draw_pipelines(): Visualize the RAG and document conversion pipelines.
        generate_response(query: str): Generate a response to a given query.

    Properties:
        sentence_context_length: Get the context length of the sentence embedder.
        sentence_embed_dims: Get the embedding dimensions of the sentence embedder.

    The class handles initialization of the document store, embedding models,
    and language models internally. It also manages the creation and execution
    of the document processing and RAG pipelines.
    """
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
        """
        Initialize the HaystackPgvector instance.

        Args:
            table_name (str): Name of the table in the Pgvector database.
            recreate_table (bool): Whether to recreate the database table.
            file_or_folder_path (Optional[str]): Path to the EPUB file to be processed.
            postgres_user_name (str): Username for Postgres database.
            postgres_password (str): Password for Postgres database.
            postgres_host (str): Host address for Postgres database.
            postgres_port (int): Port number for Postgres database.
            postgres_db_name (str): Name of the Postgres database.
            embedder_model_name (Optional[str]): Name of the embedding _model to use.
            min_section_size (int): Minimum size of a section to be considered for indexing.
        """

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
        # Find all h1 tags in the current section
        h1_tag_count: int = 0
        i: int
        for i, section in enumerate(book.get_items_of_type(ITEM_DOCUMENT)):
            section_html: str = section.get_body_content().decode('utf-8')
            # print(section_html)
            section_soup: BeautifulSoup = BeautifulSoup(section_html, 'html.parser')
            h1_tag_count = len(section_soup.find_all('h1'))
            # print()
            # print("HTML:")
            # print(section_soup)
            # print()
            paragraphs: List[Tag] = section_soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            temp_docs: List[ByteStream] = []
            temp_meta: List[Dict[str, str]] = []
            total_text: str = ""
            combined_paragraph: str = ""
            title_tag_info: str = ""
            chapter_title: str = ""
            chapter_number: int = 0
            para_num: int = 0
            page_number: str = ""
            headers: Dict[int, str] = {}  # Track headers by level
            j: int
            combined_chars: int = 0
            for j, p in enumerate(paragraphs):
                updated: bool = False
                next_p: Optional[Tag] = None
                if j < len(paragraphs) - 1:
                    next_p = paragraphs[j + 1]
                # prev_p: Optional[Tag] = None
                # if j > 0:
                #     prev_p = paragraphs[j - 1]

                # Try to get a page number
                page_anchors: List[Tag] = p.find_all('a', id=lambda x: x and x.startswith('page_'))
                if page_anchors:
                    # Extract the page number from the anchor tag id
                    for anchor in page_anchors:
                        page_id = anchor.get('id')
                        page_number = page_id.split('_')[-1]

                # Check for title information
                if is_title(p, h1_tag_count):
                    for br in p.find_all('br'):
                        br.insert_after(' ')

                    if title_tag_info == "":
                        title_tag_info = p.text.strip().title()
                    elif title_tag_info != p.text.strip().title():
                        title_tag_info += ": " + p.text.strip().title()
                        chapter_title = ""

                    # Replace 'S with 's after title casing
                    title_tag_info = title_tag_info.replace("'S", "'s")
                    title_tag_info = title_tag_info.replace("’S", "’s")

                    if chapter_title and chapter_title != title_tag_info:
                        chapter_title += ": " + title_tag_info

                    updated = True
                # Is it a chapter number tag?
                elif is_chapter_number(p):
                    chapter_number = int(p.text.strip())
                    updated = True
                elif get_header_level(p) > 0:  # If it's a header
                    header_level = get_header_level(p)
                    if header_level == 1:
                        for br in p.find_all('br'):
                            br.insert_after(' ')

                    header_text = p.text.strip()

                    if len(header_text) > 5:
                        header_text = header_text.strip().title()

                    # Remove any headers that are lower than the current one
                    headers = {level: text for level, text in headers.items() if level < header_level}

                    if header_text:
                        headers[header_level] = header_text
                    updated = True

                # Set metadata
                # Pick current title
                chapter_title = (chapter_title or section.title or title_tag_info or
                                 (headers.get(1, '') if h1_tag_count == 1 else ''))
                # # Merge header 1 and title tag
                # if headers.get(1, '') and headers.get(1, '') == chapter_title:
                #     # Delete header 1 as it is being used as chapter title
                #     del headers[1]

                # If we used the paragraph to fill in metadata, we don't want to include it in the text
                if updated:
                    continue

                p_str: str = str(p)  # p.text.strip()
                p_str_chars: int = len(p.text)
                min_paragraph_size: int = self._min_paragraph_size

                # If headers are present, adjust the minimum paragraph size for notes
                if headers and headers.get(1, '').lower() == "notes":
                    # If we're in the notes section, we want to combine paragraphs into larger sections
                    # This is because the notes are often very short, and we want to keep them together
                    # And also so that they don't dominate a semantic search
                    # We could just drop notes, but often they contain useful information
                    min_paragraph_size = self._min_paragraph_size * 2

                # If the combined paragraph is less than the minimum size combine it with the next paragraph
                if combined_chars + p_str_chars < min_paragraph_size:
                    # However, if the next pargraph is a header, we want to start a new paragraph
                    # Unless the header came just after another header, in which case we want to combine them
                    if is_title_or_heading(next_p, h1_tag_count) and not is_title_or_heading(p, h1_tag_count):
                        # Next paragraph is a header (and the current isn't), so break the paragraph here
                        p_str = combined_paragraph + "\n" + p_str
                        p_str_chars += combined_chars
                        combined_paragraph = ""
                        combined_chars = 0
                    elif j == len(paragraphs) - 1:
                        # If it's the last paragraph, then b
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
                meta_node: Dict[str, str] = {
                    "section_num": str(section_num),
                    "paragraph_num": str(para_num),
                    "book_title": book.title,
                    "section_id": section.id,
                    "file_path": file_path
                }

                # Page information
                if page_number:
                    meta_node["page_number"] = str(page_number)

                # Chapter information
                if chapter_title:
                    meta_node["chapter_title"] = chapter_title
                if chapter_number:
                    meta_node["chapter_number"] = str(chapter_number)

                # Include headers in the metadata
                # Get top level header
                top_header_level: int = 0
                if headers:
                    top_header_level = min(headers.keys())
                for level, text in headers.items():
                    if level == top_header_level:
                        meta_node["section_name"] = text
                    elif level == top_header_level - 1:
                        meta_node["subsection_name"] = text
                    else:
                        meta_node[f"header_{level}"] = text

                # self._print_verbose(meta_node)
                temp_docs.append(byte_stream)
                temp_meta.append(meta_node)

            if (len(total_text) > self._min_section_size
                    and section.id not in self._sections_to_skip.get(book.title, set())):
                self._print_verbose(f"Book: {book.title}; Section {section_num}. Section Title: {chapter_title}. "
                                    f"Length: {len(total_text)}")
                docs_list.extend(temp_docs)
                meta_list.extend(temp_meta)
                included_sections.append(book.title + ", " + section.id)
                section_num += 1
            else:
                self._print_verbose(f"Book: {book.title}; Title: {chapter_title}. Length: {len(total_text)}. Skipped.")

        self._print_verbose(f"Sections included:")
        for section in included_sections:
            self._print_verbose(section)
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
            self._print_verbose(f"Processing document: {meta[0]['book_title']}")

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
