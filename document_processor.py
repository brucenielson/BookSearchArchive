# Pytorch imports
import torch
# Haystack imports
# noinspection PyPackageRequirements
from haystack import Pipeline, Document
# noinspection PyPackageRequirements
from haystack.components.routers import ConditionalRouter
# noinspection PyPackageRequirements
from haystack.components.preprocessors import DocumentCleaner
# noinspection PyPackageRequirements
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
# noinspection PyPackageRequirements
from haystack.components.converters import HTMLToDocument, MarkdownToDocument, PyPDFToDocument, TextFileToDocument
# noinspection PyPackageRequirements
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
# noinspection PyPackageRequirements
from haystack.utils import ComponentDevice, Device
# noinspection PyPackageRequirements
from haystack.document_stores.types import DuplicatePolicy
# noinspection PyPackageRequirements
from haystack.utils.auth import Secret
from neo4j_haystack import Neo4jDocumentStore
# Other imports
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from enum import Enum
from generator_model import get_secret
from doc_content_checker import skip_content
from custom_haystack_components import (CustomDocumentSplitter, RemoveIllegalDocs, FinalDocCounter, DuplicateChecker,
                                        EPubLoader, HTMLParserComponent, print_debug_results, EpubVsPdfSplitter,
                                        EPubPdfMerger, PyMuPdf4LLM, PDFReader, PyMuPDFReader,
                                        PdfLoader, DoclingParserComponent)


# Create an enum for PDF reading strategy: PyPDFToDocument, PDFReader, PyMuPdf4LLM, PyMuPDFReader
class PDFReadingStrategy(Enum):
    PyPDFToDocument = 1
    PDFReader = 2
    PyMuPdf4LLM = 3
    PyMuPDFReader = 4
    Docling = 5


class DocumentStoreType(Enum):
    Pgvector = 1
    Neo4j = 2


class DocumentProcessor:
    def __init__(self,
                 file_folder_path_or_list: Union[str, List[str]],
                 table_name: str = 'haystack_pgvector_docs',
                 recreate_table: bool = False,
                 db_user_name: str = 'postgres',
                 db_password: str = None,
                 db_name: str = 'postgres',
                 postgres_host: str = 'localhost',
                 postgres_port: int = 5432,
                 neo4j_url: str = 'bolt://localhost:7687',
                 skip_content_func: Optional[callable] = None,
                 min_section_size: int = 1000,
                 min_paragraph_size: int = 500,
                 embedder_model_name: Optional[str] = None,
                 include_outputs_from: Optional[set[str]] = None,
                 verbose: bool = False,
                 write_to_file: bool = False,
                 pdf_reading_strategy: PDFReadingStrategy = PDFReadingStrategy.Docling,
                 document_store_type: DocumentStoreType = DocumentStoreType.Pgvector,
                 create_audio: bool = False,
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
        self._write_to_file: bool = write_to_file
        self._include_outputs_from: Optional[set[str]] = include_outputs_from
        self._pdf_reading_strategy: PDFReadingStrategy = pdf_reading_strategy
        self._document_store_type: DocumentStoreType = document_store_type
        self._neo4j_url: str = neo4j_url
        self._db_user_name: str = db_user_name
        self._db_password: str = db_password
        self._db_name: str = db_name
        self._create_audio: bool = create_audio

        # File paths
        self._file_folder_path_or_list: Union[str, List[str]] = file_folder_path_or_list  # New instance variable

        # GPU or CPU
        self._has_cuda: bool = torch.cuda.is_available()
        self._torch_device: torch.device = torch.device("cuda" if self._has_cuda else "cpu")
        self._component_device: ComponentDevice = ComponentDevice(Device.gpu() if self._has_cuda else Device.cpu())

        # Passwords and connection strings
        if db_password is None:
            raise ValueError("Postgres password must be provided")
        # PG_CONN_STR="postgresql://USER:PASSWORD@HOST:PORT/DB_NAME
        self._postgres_connection_str: str = (f"postgresql://{db_user_name}:{db_password}@"
                                              f"{postgres_host}:{postgres_port}/{db_name}")

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

        """
        Run the document processing pipeline.

        Args:
            file_folder_path_or_list (Union[str, List[str]], optional): The file or folder path to process.
                If not provided, the previously set path will be used.
        """
        if file_folder_path_or_list is not None:
            self._file_folder_path_or_list = file_folder_path_or_list
        if self._doc_convert_pipeline is None:
            self._init_doc_converter_pipeline()
        self._process_documents(file_folder_path_or_list=self._file_folder_path_or_list)

    def draw_pipeline(self) -> None:
        """
        Draw and save visual representations of the document conversion pipelines.
        """
        if self._doc_convert_pipeline is not None:
            self._doc_convert_pipeline.draw(Path("Document Conversion Pipeline.png"))

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

    def _create_file_list(self) -> str:
        # Handle the case where the input is a list of file paths
        if isinstance(self._file_folder_path_or_list, list):
            yield from self._file_folder_path_or_list

        # Now cast to be a string
        if not isinstance(self._file_folder_path_or_list, str):
            self._file_folder_path_or_list = str(self._file_folder_path_or_list[0])
        if Path(self._file_folder_path_or_list).is_dir():
            for file_path in Path(self._file_folder_path_or_list).glob('*'):
                if file_path.suffix.lower() in {'.epub', '.pdf'}:
                    yield str(file_path)
        elif Path(self._file_folder_path_or_list).is_file():
            path: Path = Path(self._file_folder_path_or_list)
            if path.suffix.lower() in {'.epub', '.pdf'}:
                yield str(path)
            else:
                raise ValueError("The provided file must be an .epub or .pdf")
        else:
            raise ValueError("The provided path must be a valid file, directory, or list of files.")

    def _init_doc_converter_pipeline(self) -> None:
        self._setup_embedder()
        # Create the custom splitter
        custom_splitter: CustomDocumentSplitter = CustomDocumentSplitter(self._sentence_embedder,
                                                                         verbose=self._write_to_file,
                                                                         skip_content_func=self._skip_content)

        embedding_routes: List[Dict] = [
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
        # Embedding router is used to bypass the embedder if all documents were marked as duplicates
        # unsafe must be set to True to allow Document outputs
        embedding_router = ConditionalRouter(routes=embedding_routes, unsafe=True)

        # Create the document conversion pipeline
        doc_convert_pipe: Pipeline = Pipeline()
        doc_convert_pipe.add_component("epub_vs_pdf_splitter", EpubVsPdfSplitter())
        if self._pdf_reading_strategy == PDFReadingStrategy.PyPDFToDocument:
            doc_convert_pipe.add_component("pdf_loader", PyPDFToDocument())
        elif self._pdf_reading_strategy == PDFReadingStrategy.PDFReader:
            doc_convert_pipe.add_component("pdf_loader", PDFReader())
        elif self._pdf_reading_strategy == PDFReadingStrategy.PyMuPdf4LLM:
            doc_convert_pipe.add_component("pdf_loader", PyMuPdf4LLM())
            doc_convert_pipe.add_component("markdown_converter", MarkdownToDocument())
        elif self._pdf_reading_strategy == PDFReadingStrategy.Docling:
            doc_convert_pipe.add_component("pdf_loader", PdfLoader(verbose=self._verbose))
            doc_convert_pipe.add_component("docling_parser",
                                           DoclingParserComponent(min_paragraph_size=self._min_paragraph_size,
                                                                  min_section_size=self._min_section_size,
                                                                  verbose=self._verbose))
            doc_convert_pipe.add_component("text_converter", TextFileToDocument())
        elif self._pdf_reading_strategy == PDFReadingStrategy.PyMuPDFReader:
            doc_convert_pipe.add_component("pdf_loader", PyMuPDFReader())

        doc_convert_pipe.add_component("epub_loader", EPubLoader(verbose=self._verbose))
        doc_convert_pipe.add_component("html_parser",
                                       HTMLParserComponent(min_paragraph_size=self._min_paragraph_size,
                                                           min_section_size=self._min_section_size,
                                                           verbose=self._verbose))
        doc_convert_pipe.add_component("html_converter", HTMLToDocument())
        doc_convert_pipe.add_component("epub_pdf_merger", EPubPdfMerger())
        doc_convert_pipe.add_component("remove_illegal_docs", instance=RemoveIllegalDocs())
        doc_convert_pipe.add_component("cleaner", DocumentCleaner())
        doc_convert_pipe.add_component("splitter", custom_splitter)
        doc_convert_pipe.add_component("duplicate_checker", DuplicateChecker(document_store=self._document_store))
        doc_convert_pipe.add_component("embedder", self._sentence_embedder)
        doc_convert_pipe.add_component("router", embedding_router)
        doc_convert_pipe.add_component("writer",
                                       DocumentWriter(document_store=self._document_store,
                                                      policy=DuplicatePolicy.OVERWRITE))
        doc_convert_pipe.add_component("final_counter", FinalDocCounter())

        # Connect the components in the pipeline
        # Start at epub_vs_pdf_splitter which routes to either the epub or pdf pipeline
        doc_convert_pipe.connect("epub_vs_pdf_splitter.epub_paths", "epub_loader.file_paths")
        doc_convert_pipe.connect("epub_vs_pdf_splitter.pdf_paths", "pdf_loader.sources")
        # EPUB pipeline
        doc_convert_pipe.connect("epub_loader.html_pages", "html_parser.html_pages")
        doc_convert_pipe.connect("epub_loader.meta", "html_parser.meta")
        doc_convert_pipe.connect("html_parser.sources", "html_converter.sources")
        doc_convert_pipe.connect("html_parser.meta", "html_converter.meta")
        doc_convert_pipe.connect("html_converter.documents", "epub_pdf_merger.epub_docs")
        # PDF pipeline
        if self._pdf_reading_strategy == PDFReadingStrategy.PyPDFToDocument:
            doc_convert_pipe.connect("pdf_loader.documents", "epub_pdf_merger.pdf_docs")
        elif self._pdf_reading_strategy == PDFReadingStrategy.PDFReader:
            doc_convert_pipe.connect("pdf_loader.documents", "epub_pdf_merger.pdf_docs")
        elif self._pdf_reading_strategy == PDFReadingStrategy.PyMuPdf4LLM:
            doc_convert_pipe.connect("pdf_loader.sources", "markdown_converter.sources")
            doc_convert_pipe.connect("pdf_loader.meta", "markdown_converter.meta")
            doc_convert_pipe.connect("markdown_converter.documents", "epub_pdf_merger.pdf_docs")
        elif self._pdf_reading_strategy == PDFReadingStrategy.Docling:
            doc_convert_pipe.connect("pdf_loader.docling_docs", "docling_parser.sources")
            doc_convert_pipe.connect("pdf_loader.meta", "docling_parser.meta")
            doc_convert_pipe.connect("docling_parser.sources", "text_converter.sources")
            doc_convert_pipe.connect("docling_parser.meta", "text_converter.meta")
            doc_convert_pipe.connect("text_converter.documents", "epub_pdf_merger.pdf_docs")
        elif self._pdf_reading_strategy == PDFReadingStrategy.PyMuPDFReader:
            doc_convert_pipe.connect("pdf_loader.documents", "epub_pdf_merger.pdf_docs")

        # Remaining pipeline to final counter
        doc_convert_pipe.connect("epub_pdf_merger.documents", "remove_illegal_docs")
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
        def init_doc_store(force_recreate: bool = False) -> Union[PgvectorDocumentStore, Neo4jDocumentStore]:
            if self._document_store_type == DocumentStoreType.Pgvector:
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
            elif self._document_store_type == DocumentStoreType.Neo4j:
                # https://haystack.deepset.ai/integrations/neo4j-document-store
                doc_store: Neo4jDocumentStore = Neo4jDocumentStore(
                    url=self._neo4j_url,
                    username=self._db_user_name,
                    password=self._db_password,
                    database=self._db_name,
                    embedding_dim=self.embed_dims,
                    embedding_field="embedding",
                    index="document-embeddings",  # The name of the Vector Index in Neo4j
                    node_label="Document",  # Providing a label to Neo4j nodes which store Documents
                    recreate_index=self._recreate_table or force_recreate,
                )
                return doc_store

        document_store: Union[PgvectorDocumentStore, Neo4jDocumentStore]
        document_store = init_doc_store()
        self._document_store = document_store

    def _process_documents(self, file_folder_path_or_list: Union[str, List[str]] = None) -> None:
        doc_count: int = self._document_store.count_documents()
        self._print_verbose("Document Count: " + str(doc_count))
        self._print_verbose("Loading document file")

        # Iterate over the document and metadata pairs as they are loaded
        total_written: int = 0

        for file_path in self._create_file_list():
            file_path_list: List[str] = [file_path]
            self._print_verbose(f"Processing file: {file_path} ")
            results: Dict[str, Any] = self._doc_convert_pipeline.run(
                {"epub_vs_pdf_splitter": {"file_paths": file_path_list}},
                include_outputs_from=self._include_outputs_from)
            print_debug_results(results, include_outputs_from=self._include_outputs_from, verbose=self._verbose)
            written = results["final_counter"]["documents_written"]
            total_written += written
            self._print_verbose(f"Wrote {written} documents for {file_path}.")

        self._print_verbose(f"Finished writing documents to document store. Final document count: {total_written}")


def main() -> None:
    file_path: str = "documents"
    doc_store_type: DocumentStoreType = DocumentStoreType.Pgvector
    password: str = ""
    user_name: str = ""
    db_name: str = ""
    if doc_store_type == DocumentStoreType.Pgvector:
        password = get_secret(r'D:\Documents\Secrets\postgres_password.txt')
        user_name = "postgres"
        db_name = "postgres"
    elif doc_store_type == DocumentStoreType.Neo4j:
        password = get_secret(r'D:\Documents\Secrets\neo4j_password.txt')
        user_name = "neo4j"
        db_name = "neo4j"
    include_outputs_from: Optional[set[str]] = None  # {"final_counter"}
    # noinspection SpellCheckingInspection
    processor: DocumentProcessor = DocumentProcessor(
        table_name="book_archive",
        recreate_table=False,
        embedder_model_name="BAAI/llm-embedder",
        file_folder_path_or_list=file_path,
        db_user_name=user_name,
        db_password=password,
        postgres_host='localhost',
        postgres_port=5432,
        db_name=db_name,
        skip_content_func=skip_content,
        min_section_size=3000,
        min_paragraph_size=300,
        include_outputs_from=include_outputs_from,
        verbose=True,
        pdf_reading_strategy=PDFReadingStrategy.Docling,
        write_to_file=True,
        document_store_type=doc_store_type,
    )

    # Process documents in the specified folder
    processor.run(file_path)

    # Draw images of the pipelines
    if processor.verbose:
        processor.draw_pipeline()
        print("Sentence Embedder Dims: " + str(processor.embed_dims))
        print("Sentence Embedder Context Length: " + str(processor.context_length))


if __name__ == "__main__":
    main()

# TODO: Add other document types

# TODO: Problems with section headers with PDF loader
"""
Score: 0.9775
Page #: 188
Book Title: Realism and the Aim of Science -- Karl Popper -- 2017
Section Name: (1)  The preceding section was, as noted above, first published in 1 957. It contains, among other things, a refutation of the view, held
Quote: By induction I mean an argument which, given some empirical (singular or particular) premises, leads to a universal conclusion, a universal theory, either with logical certainty, or with 'probability' (in the sense in which this term is used in the calculus of probability).
The argument against induction that I wish to restate here is very simple: Many theories, such as Newton's, which have been thought to be the result of induction, actually are inconsistent with their alleged (partial) inductive premises, as shown above.



Score: 0.9992
Page #: 188
Book Title: Realism and the Aim of Science -- Karl Popper -- 2017
Section Name: (1)  The preceding section was, as noted above, first published in 1 957. It contains, among other things, a refutation of the view, held
Quote: By induction I mean an argument which, given some empirical (singular or particular) premises, leads to a universal conclusion, a universal theory, either with logical certainty, or with 'probability' (in the sense in which this term is used in the calculus of probability).
The argument against induction that I wish to restate here is very simple: Many theories, such as Newton's, which have been thought to be the result of induction, actually are inconsistent with their alleged (partial) inductive premises, as shown above.

"""

# TODO: This passage is too long I think. Why isn't it getting split up? This is the EPUB loader.
#  Semantic search won't work on end of this passage.
"""
Score: 0.0007
Page #: 267
Book Title: The Open Society and Its Enemies (New One-Volume Edition)
Paragraph #: 53
Section Name: 12: Hegel And The New Tribalism
Subsection Name: III
Quote: These are a few episodes in the career of the man whose ‘windbaggery’ has given rise to modern nationalism as well as to modern Idealist philosophy, erected upon the perversion of Kant’s teaching. (I follow Schopenhauer in distinguishing between Fichte’s ‘windbaggery’ and Hegel’s ‘charlatanry’, although I must admit that to insist on this distinction is perhaps a little pedantic.) The whole story is interesting mainly because of the light it throws upon the ‘history of philosophy’ and upon ‘history’ in general. I mean not only the perhaps more humorous than scandalous fact that such clowns are taken seriously, and that they are made the objects of a kind of worship, of solemn although often boring studies (and of examination papers to match). I mean not only the appalling fact that the windbag Fichte and the charlatan Hegel are treated on a level with men like Democritus, Pascal, Descartes, Spinoza, Locke, Hume, Kant, J. S. Mill, and Bertrand Russell, and that their moral teaching is taken seriously and perhaps even considered superior to that of these other men. But I mean that many of these eulogist historians of philosophy, unable to discriminate between thought and fancy, not to mention good and bad, dare to pronounce that their history is our judge, or that their history of philosophy is an implicit criticism of the different ‘systems of thought’. For it is clear, I think, that their adulation can only be an implicit criticism of their histories of philosophy, and of that pomposity and conspiracy of noise by which the business of philosophy is glorified. It seems to be a law of what these people are pleased to call ‘human nature’ that bumptiousness grows in direct proportion to deficiency of thought and inversely to the amount of service rendered to human welfare.
"""
