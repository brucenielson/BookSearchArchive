# Hugging Face and Pytorch imports
import torch
from sentence_transformers import SentenceTransformer
# EPUB imports
from bs4 import BeautifulSoup
from ebooklib import epub, ITEM_DOCUMENT
# Haystack imports
from haystack import Pipeline, Document, component
from haystack.dataclasses import ByteStream
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.converters import HTMLToDocument
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from haystack.utils import ComponentDevice, Device
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils.auth import Secret
# Other imports
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from generator_model import get_secret


@component
class RemoveIllegalDocs:
    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
        documents = [Document(content=doc.content, meta=doc.meta) for doc in documents if doc.content is not None]
        documents = list({doc.id: doc for doc in documents}.values())
        return {"documents": documents}


@component
class CustomDocumentSplitter:
    def __init__(self, embedder: SentenceTransformersDocumentEmbedder):
        self.embedder: SentenceTransformersDocumentEmbedder = embedder
        self.model: SentenceTransformer = embedder.embedding_backend.model
        self.tokenizer = self.model.tokenizer
        self.max_seq_length = self.model.get_max_seq_length()

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> dict:
        processed_docs = []
        for doc in documents:
            processed_docs.extend(self.process_document(doc))

        print(f"Processed {len(documents)} documents into {len(processed_docs)} documents")
        return {"documents": processed_docs}

    def process_document(self, document: Document) -> List[Document]:
        token_count = self.count_tokens(document.content)

        if token_count <= self.max_seq_length:
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
            if all(self.count_tokens(doc.content) <= self.max_seq_length for doc in split_docs):
                return split_docs

            # If not, reduce split_length and try again
            split_length -= 1

        # If we get here, even single sentences exceed max_seq_length
        # So just let the splitter truncate the document
        # But give warning that document was truncated
        print(f"Document was truncated to fit within max sequence length of {self.max_seq_length}: "
              f"Actual length: {self.count_tokens(document.content)}")
        print(f"Problem Document: {document.content}")
        return [document]

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))


class DocumentProcessor:
    """
    A class that implements a Retrieval-Augmented Generation (RAG) system using Haystack and Pgvector.

    This class provides functionality to set up and use a RAG system for question answering
    tasks on a given corpus of text, currently from an EPUB file. It handles document
    indexing, embedding, retrieval, and generation of responses using a language model.

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
                 book_file_path: Optional[str] = None,
                 postgres_user_name: str = 'postgres',
                 postgres_password: str = None,
                 postgres_host: str = 'localhost',
                 postgres_port: int = 5432,
                 postgres_db_name: str = 'postgres',
                 min_section_size: int = 1000,
                 embedder_model_name: Optional[str] = None,
                 ) -> None:
        """
        Initialize the HaystackPgvector instance.

        Args:
            table_name (str): Name of the table in the Pgvector database.
            recreate_table (bool): Whether to recreate the database table.
            book_file_path (Optional[str]): Path to the EPUB file to be processed.
            postgres_user_name (str): Username for Postgres database.
            postgres_password (str): Password for Postgres database.
            postgres_host (str): Host address for Postgres database.
            postgres_port (int): Port number for Postgres database.
            postgres_db_name (str): Name of the Postgres database.
            embedder_model_name (Optional[str]): Name of the embedding model to use.
            min_section_size (int): Minimum size of a section to be considered for indexing.
        """

        # Instance variables
        self._book_file_path: Optional[str] = book_file_path
        self._table_name: str = table_name
        self._recreate_table: bool = recreate_table
        self._min_section_size = min_section_size
        self._embedder_model_name: Optional[str] = embedder_model_name
        self._sentence_embedder: Optional[SentenceTransformersDocumentEmbedder] = None

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

        print("Initializing document store")
        self._document_store: Optional[PgvectorDocumentStore] = None
        self._doc_convert_pipeline: Optional[Pipeline] = None
        self._initialize_document_store()

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

    def _load_epub(self) -> Tuple[List[ByteStream], List[Dict[str, str]]]:
        docs: List[ByteStream] = []
        meta: List[Dict[str, str]] = []
        book: epub.EpubBook = epub.read_epub(self._book_file_path)
        section_num: int = 1
        for i, section in enumerate(book.get_items_of_type(ITEM_DOCUMENT)):
            section_html: str = section.get_body_content().decode('utf-8')
            section_soup: BeautifulSoup = BeautifulSoup(section_html, 'html.parser')
            headings = [heading.get_text().strip() for heading in section_soup.find_all('h1')]
            title = ' '.join(headings)
            paragraphs: List[Any] = section_soup.find_all('p')
            temp_docs: List[ByteStream] = []
            temp_meta: List[Dict[str, str]] = []
            total_text: str = ""
            for p in paragraphs:
                p_str: str = str(p)
                # Concatenate paragraphs to form a single document string
                total_text += p_str
                p_html: str = f"<html><head><title>Converted Epub</title></head><body>{p_str}</body></html>"
                byte_stream: ByteStream = ByteStream(p_html.encode('utf-8'))
                meta_node: Dict[str, str] = {"section_num": section_num, "title": title}
                temp_docs.append(byte_stream)
                temp_meta.append(meta_node)

            # If the total text length is greater than the minimum section size, add the section to the list
            if len(total_text) > self._min_section_size:
                docs.extend(temp_docs)
                meta.extend(temp_meta)
                section_num += 1
        return docs, meta

    def _doc_converter_pipeline(self) -> None:
        self._setup_embedder()
        # Create the custom splitter
        custom_splitter: CustomDocumentSplitter = CustomDocumentSplitter(self._sentence_embedder)
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
        connection_token: Secret = Secret.from_token(self._postgres_connection_str)
        document_store: PgvectorDocumentStore = PgvectorDocumentStore(
            connection_string=connection_token,
            table_name=self._table_name,
            embedding_dimension=self.embed_dims,
            vector_function="cosine_similarity",
            recreate_table=self._recreate_table,
            search_strategy="hnsw",
            hnsw_recreate_index_if_exists=True,
            hnsw_index_name=self._table_name + "_haystack_hnsw_index",
            keyword_index_name=self._table_name + "_haystack_keyword_index",
        )

        self._document_store = document_store

        print("Document Count: " + str(document_store.count_documents()))

        if document_store.count_documents() == 0 and self._book_file_path is not None:
            sources: List[ByteStream]
            meta: List[Dict[str, str]]
            print("Loading document file")
            sources, meta = self._load_epub()
            print("Writing documents to document store")
            self._doc_converter_pipeline()
            results: Dict[str, Any] = self._doc_convert_pipeline.run({"converter": {"sources": sources, "meta": meta}})
            print(f"\n\nNumber of documents: {results['writer']['documents_written']}")


def main() -> None:
    epub_file_path: str = "Federalist Papers.epub"
    postgres_password = get_secret(r'D:\Documents\Secrets\postgres_password.txt')
    processor: DocumentProcessor = DocumentProcessor(
        table_name="federalist_papers",
        recreate_table=False,
        embedder_model_name="Alibaba-NLP/gte-large-en-v1.5",
        book_file_path=epub_file_path,
        postgres_user_name='postgres',
        postgres_password=postgres_password,
        postgres_host='localhost',
        postgres_port=5432,
        postgres_db_name='bookarchive',
        min_section_size=1000,
    )

    # Draw images of the pipelines
    processor.draw_pipeline()
    print("Sentence Embedder Dims: " + str(processor.embed_dims))
    print("Sentence Embedder Context Length: " + str(processor.context_length))


if __name__ == "__main__":
    main()
