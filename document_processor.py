# Hugging Face and Pytorch imports
import csv

import torch
from sentence_transformers import SentenceTransformer
# EPUB imports
from bs4 import BeautifulSoup
from ebooklib import epub, ITEM_DOCUMENT
# Haystack imports
# noinspection PyPackageRequirements
from haystack import Pipeline, Document, component
# noinspection PyPackageRequirements
from haystack.dataclasses import ByteStream
# noinspection PyPackageRequirements
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
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
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from typing_extensions import Set
from generator_model import get_secret
from doc_content_checker import skip_content


@component
class RemoveIllegalDocs:
    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
        documents = [Document(content=doc.content, meta=doc.meta) for doc in documents if doc.content is not None]
        documents = list({doc.id: doc for doc in documents}.values())
        return {"documents": documents}


@component
class CustomDocumentSplitter:
    def __init__(self, embedder: SentenceTransformersDocumentEmbedder,
                 verbose=True,
                 skip_content_func: Optional[callable] = None):
        self._embedder: SentenceTransformersDocumentEmbedder = embedder
        self._verbose: bool = verbose
        self._skip_content_func: Optional[callable] = skip_content_func
        self._model: SentenceTransformer = embedder.embedding_backend.model
        self._tokenizer = self._model.tokenizer
        self._max_seq_length = self._model.get_max_seq_length()

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> dict:
        processed_docs = []
        last_section_num = None  # Track the last section number
        sections_to_skip = set()  # Sections to skip

        # Delete "first_paragraph_section.txt"
        file_name: str = "first_paragraph_per_section.txt"
        if self._verbose:
            with open(file_name, "w", encoding="utf-8") as file:
                file.write("")

        for doc in documents:
            # Extract section_num and paragraph_num from the metadata
            section_num = int(doc.meta.get("section_num"))
            paragraph_num = int(doc.meta.get("paragraph_num"))

            # If this is a section to skip, go to the next document
            if (doc.meta.get("book_title"), section_num) in sections_to_skip:
                continue

            # If verbose is True, print the content when section_num changes and paragraph_num == 1
            if section_num != last_section_num and paragraph_num == 1:
                if self._verbose:
                    with open(file_name, "a", encoding="utf-8") as file:
                        file.write(f"Section: {section_num}\n")
                        file.write(f"Book Title: {doc.meta.get('book_title')}\n")
                        file.write(f"Section Title: {doc.meta.get('section_title')}\n")
                        file.write(f"Content:\n{doc.content}\n\n")
                # For the first paragraph, check for possible section skipping
                if self._skip_content_func is not None and self._skip_content_func(doc.content):
                    if self._verbose:
                        # Skip this section
                        print(f"Skipping section {doc.meta.get('book_title')} / {doc.meta.get('section_title')} "
                              f"due to content check")
                    sections_to_skip.add((doc.meta.get("book_title"), section_num))
                    continue

            # Update the last_section_num
            last_section_num = section_num

            # Process and extend documents
            processed_docs.extend(self.process_document(doc))

        if self._verbose:
            print(f"Processed {len(documents)} documents into {len(processed_docs)} documents")
        return {"documents": processed_docs}

    def process_document(self, document: Document) -> List[Document]:
        token_count = self.count_tokens(document.content)
        if token_count <= self._max_seq_length:
            # Document fits within max sequence length, no need to split
            return [document]

        # Document exceeds max sequence length, find optimal split_length
        split_docs = self.find_optimal_split(document)
        return split_docs

    def find_optimal_split(self, document: Document) -> List[Document]:
        split_length = 10  # Start with 10 sentences
        while split_length > 0:
            splitter = DocumentSplitter(
                split_by="sentence",
                split_length=split_length,
                split_overlap=min(1, split_length - 1),
                split_threshold=min(3, split_length)
            )
            split_docs = splitter.run(documents=[document])["documents"]

            # Check if all split documents fit within max_seq_length
            if all(self.count_tokens(doc.content) <= self._max_seq_length for doc in split_docs):
                return split_docs

            # If not, reduce split_length and try again
            split_length -= 1

        # If we get here, even single sentences exceed max_seq_length
        # So just let the splitter truncate the document
        # But give warning that document was truncated
        if self._verbose:
            print(f"Document was truncated to fit within max sequence length of {self._max_seq_length}: "
                  f"Actual length: {self.count_tokens(document.content)}")
            print(f"Problem Document: {document.content}")
        return [document]

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text, verbose=False))


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
                 table_name: str = 'haystack_pgvector_docs',
                 recreate_table: bool = False,
                 file_or_folder_path: Optional[str] = None,
                 postgres_user_name: str = 'postgres',
                 postgres_password: str = None,
                 postgres_host: str = 'localhost',
                 postgres_port: int = 5432,
                 postgres_db_name: str = 'postgres',
                 skip_content_func: Optional[callable] = None,
                 min_section_size: int = 1000,
                 min_paragraph_size: int = 1000,
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
        self._file_or_folder_path: Optional[str] = file_or_folder_path  # New instance variable

        # Determine if the path is a file or directory
        if self._file_or_folder_path is not None:
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

    def _load_files(self) -> Tuple[List[ByteStream], List[Dict[str, str]]]:
        all_docs: List[ByteStream] = []
        all_meta: List[Dict[str, str]] = []

        if self._is_directory:
            for file_path in Path(self._file_or_folder_path).glob('*.epub'):
                docs, meta = self._load_epub(str(file_path))
                all_docs.extend(docs)
                all_meta.extend(meta)
        else:
            all_docs, all_meta = self._load_epub(self._file_or_folder_path)

        return all_docs, all_meta

    def _load_epub(self, file_path: str) -> Tuple[List[ByteStream], List[Dict[str, str]]]:
        docs: List[ByteStream] = []
        meta: List[Dict[str, str]] = []
        included_sections: List[str] = []
        book: epub.EpubBook = epub.read_epub(file_path)
        self._print_verbose()
        self._print_verbose(f"Book Title: {book.title}")
        section_num: int = 1
        i: int
        for i, section in enumerate(book.get_items_of_type(ITEM_DOCUMENT)):
            section_html: str = section.get_body_content().decode('utf-8')
            section_soup: BeautifulSoup = BeautifulSoup(section_html, 'html.parser')
            headings: List[str] = [heading.get_text().strip() for heading in section_soup.find_all('h1')]
            section_id: str = section.id
            section_title: str = ' '.join(headings)
            if section_title == "":
                section_title = section_id
            else:
                section_title = section_title + " - " + section_id
            paragraphs: List[Any] = section_soup.find_all('p')
            temp_docs: List[ByteStream] = []
            temp_meta: List[Dict[str, str]] = []
            total_text: str = ""
            combined_paragraph: str = ""
            para_num: int = 0
            j: int
            for j, p in enumerate(paragraphs):
                p_str: str = str(p)
                if len(combined_paragraph) + len(p_str) < self._min_paragraph_size:
                    combined_paragraph += "\n" + p_str
                    # If it's the last paragraph, process it
                    if j == len(paragraphs) - 1:
                        p_str = combined_paragraph
                    else:
                        continue
                else:
                    p_str = combined_paragraph + "\n" + p_str
                    combined_paragraph = ""
                para_num += 1
                total_text += p_str
                p_html: str = f"<html><head><title>Converted Epub</title></head><body>{p_str}</body></html>"
                byte_stream: ByteStream = ByteStream(p_html.encode('utf-8'))
                meta_node: Dict[str, str] = {
                    "section_num": str(section_num),
                    "paragraph_num": str(para_num),
                    "book_title": book.title,
                    "section_id": section_id,
                    "section_title": section_title,
                    "file_path": file_path
                }
                temp_docs.append(byte_stream)
                temp_meta.append(meta_node)

            if (len(total_text) > self._min_section_size
                    and section_id not in self._sections_to_skip.get(book.title, set())):
                self._print_verbose(f"Book: {book.title}; Section {section_num}. Section Title: {section_title}. "
                                    f"Length: {len(total_text)}")
                docs.extend(temp_docs)
                meta.extend(temp_meta)
                included_sections.append(book.title + ", " + section_id)
                section_num += 1
            else:
                self._print_verbose(f"Book: {book.title}; Title: {section_title}. Length: {len(total_text)}. Skipped.")

        self._print_verbose(f"Sections included:")
        for section in included_sections:
            self._print_verbose(section)
        self._print_verbose()
        return docs, meta

    def _doc_converter_pipeline(self) -> None:
        self._setup_embedder()
        # Create the custom splitter
        custom_splitter: CustomDocumentSplitter = CustomDocumentSplitter(self._sentence_embedder,
                                                                         verbose=self._verbose,
                                                                         skip_content_func=self._skip_content)
        # Create the document conversion pipeline
        doc_convert_pipe: Pipeline = Pipeline()
        doc_convert_pipe.add_component("converter", HTMLToDocument())
        doc_convert_pipe.add_component("remove_illegal_docs", instance=RemoveIllegalDocs())
        doc_convert_pipe.add_component("cleaner", DocumentCleaner())
        doc_convert_pipe.add_component("splitter", custom_splitter)
        doc_convert_pipe.add_component("embedder", self._sentence_embedder)
        doc_convert_pipe.add_component("writer",
                                       DocumentWriter(document_store=self._document_store,
                                                      policy=DuplicatePolicy.OVERWRITE))

        doc_convert_pipe.connect("converter", "remove_illegal_docs")
        doc_convert_pipe.connect("remove_illegal_docs", "cleaner")
        doc_convert_pipe.connect("cleaner", "splitter")
        doc_convert_pipe.connect("splitter", "embedder")
        doc_convert_pipe.connect("embedder", "writer")

        self._doc_convert_pipeline = doc_convert_pipe

    def _initialize_document_store(self) -> None:
        def create_doc_store(force_recreate: bool = False) -> PgvectorDocumentStore:
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
        document_store = create_doc_store()
        self._document_store = document_store

        self._print_verbose("Document Count: " + str(document_store.count_documents()))

        if document_store.count_documents() == 0 and self._file_or_folder_path is not None:
            sources: List[ByteStream]
            meta: List[Dict[str, str]]
            self._print_verbose("Loading document file")
            sources, meta = self._load_files()
            self._print_verbose("Writing documents to document store")
            self._doc_converter_pipeline()
            results: Dict[str, Any] = self._doc_convert_pipeline.run({"converter": {"sources": sources, "meta": meta}})
            self._print_verbose(f"\n\nNumber of documents: {results['writer']['documents_written']}")


def main() -> None:
    epub_file_path: str = "documents/Karl Popper - The Myth of the Framework-Taylor and Francis.epub"
    postgres_password = get_secret(r'D:\Documents\Secrets\postgres_password.txt')
    processor: DocumentProcessor = DocumentProcessor(
        table_name="popper_archive",
        recreate_table=True,
        embedder_model_name="BAAI/llm-embedder",
        file_or_folder_path=epub_file_path,
        postgres_user_name='postgres',
        postgres_password=postgres_password,
        postgres_host='localhost',
        postgres_port=5432,
        postgres_db_name='postgres',
        skip_content_func=skip_content,
        min_section_size=3000,
        min_paragraph_size=1000,
        verbose=True
    )

    # Draw images of the pipelines
    if processor.verbose:
        processor.draw_pipeline()
        print("Sentence Embedder Dims: " + str(processor.embed_dims))
        print("Sentence Embedder Context Length: " + str(processor.context_length))


if __name__ == "__main__":
    main()

# TODO: Rewrite this to load one document into the store at a time so I don't hold everything in memory.
# TODO: There should be a 'true' section number based on finding a number then a return line character in paragraph 1
# TODO: Add hybrid search
