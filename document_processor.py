# Pytorch imports
import torch
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
from typing import List, Optional, Dict, Any
from pathlib import Path
from generator_model import get_secret
from doc_content_checker import skip_content
from custom_haystack_components import (CustomDocumentSplitter, RemoveIllegalDocs, FinalDocCounter, DuplicateChecker,
                                        EPubLoader, HTMLParserComponent)


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

    def _load_files(self) -> str:
        if self._is_directory:
            for file_path in Path(self._file_or_folder_path).glob('*.epub'):
                yield str(file_path)
        else:
            path: Path = Path(self._file_or_folder_path)
            yield str(path)

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
        router = ConditionalRouter(routes=routes, unsafe=True)  # unsafe must be set to True to allow Document outputs

        # Create the document conversion pipeline
        doc_convert_pipe: Pipeline = Pipeline()
        doc_convert_pipe.add_component("epub_loader", EPubLoader(verbose=self._verbose))
        doc_convert_pipe.add_component("html_parser",
                                       HTMLParserComponent(min_paragraph_size=self._min_paragraph_size,
                                                           min_section_size=self._min_section_size,
                                                           verbose=self._verbose))
        doc_convert_pipe.add_component("html_converter", HTMLToDocument())
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

        doc_convert_pipe.connect("epub_loader.html_pages", "html_parser.html_pages")
        doc_convert_pipe.connect("epub_loader.meta", "html_parser.meta")
        doc_convert_pipe.connect("html_parser.sources", "html_converter.sources")
        doc_convert_pipe.connect("html_parser.meta", "html_converter.meta")
        doc_convert_pipe.connect("html_converter", "remove_illegal_docs")
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
        # self._doc_convert_pipeline.run({"epub_loader": {"file_or_folder_path": self._file_or_folder_path}})

        for file_path in self._load_files():
            self._print_verbose(f"Processing file: {file_path} ")
            results: Dict[str, Any] = self._doc_convert_pipeline.run({"epub_loader": {"file_path": file_path}})
            written = results["final_counter"]["documents_written"]
            total_written += written
            self._print_verbose(f"Wrote {written} documents for {file_path}.")

        self._print_verbose(f"Finished writing documents to document store. Final document count: {total_written}")


def main() -> None:
    # epub_file_path: str = "documents/Karl Popper - All Life is Problem Solving-Taylor and Francis.epub"
    # epub_file_path: str = "documents/Karl Popper - The Myth of the Framework-Taylor and Francis.epub"
    # epub_file_path: str = "documents/Karl Popper - Conjectures and Refutations-Taylor and Francis (2018).epub"
    # epub_file_path: str = "documents/Karl Popper - The Open Society and Its Enemies-Princeton University Press (2013).epub"  # noqa: E501
    # epub_file_path: str = "documents/Karl Popper - The World of Parmenides-Taylor & Francis (2012).epub"
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

# TODO: Add PDF support (and maybe other document types)
