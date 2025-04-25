# Pytorch imports
import torch
# Haystack imports
# noinspection PyPackageRequirements
from haystack import Pipeline
# noinspection PyPackageRequirements
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
# noinspection PyPackageRequirements
from haystack.dataclasses import StreamingChunk
from haystack_integrations.components.retrievers.pgvector import PgvectorEmbeddingRetriever, PgvectorKeywordRetriever
# noinspection PyPackageRequirements
from haystack import Document
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
# noinspection PyPackageRequirements
from haystack.utils import ComponentDevice, Device
# noinspection PyPackageRequirements
from haystack.utils.auth import Secret
# Neo4j imports
from neo4j_haystack import Neo4jDocumentStore
# Other imports
from typing import Optional, Dict, Any, Union, List, Tuple
from pathlib import Path
import generator_model as gen
from enum import Enum
from document_processor import DocumentStoreType
from custom_haystack_components import (DocumentCollector, RetrieverWrapper, print_documents,
                                        QueryComponent, print_debug_results, Reranker
                                        )


class SearchMode(Enum):
    LEXICAL = 1
    SEMANTIC = 2
    HYBRID = 3


class DocRetrievalPipeline:
    # The amount of text streamed since last newline.
    _streamed_text_length: int = 0

    def __init__(self,
                 table_name: str = 'haystack_pgvector_docs',
                 db_user_name: str = 'postgres',
                 db_password: str = None,
                 postgres_host: str = 'localhost',
                 postgres_port: int = 5432,
                 db_name: str = 'postgres',
                 embedder_model_name: Optional[str] = None,
                 verbose: bool = False,
                 llm_top_k: int = 5,
                 retriever_top_k_docs: int = None,
                 search_mode: SearchMode = SearchMode.HYBRID,
                 use_reranker: bool = True,
                 include_outputs_from: Optional[set[str]] = None,
                 ) -> None:

        # streaming_callback function to print to screen
        def streaming_callback(chunk: StreamingChunk) -> None:
            # Print the content of the chunks but wrap the text after 80 characters
            if self._allow_streaming_callback:
                DocRetrievalPipeline._streamed_text_length += len(chunk.content)
                if DocRetrievalPipeline._streamed_text_length < 80 or chunk.content in ['.', ',', ';', ':', '!', '?', ' ', '\n']:
                    print(chunk.content, end='')
                    if chunk.content == '\n':
                        DocRetrievalPipeline._streamed_text_length = 0
                else:
                    print()
                    print(chunk.content.strip(), end='')
                    DocRetrievalPipeline._streamed_text_length = len(chunk.content)

        # Instance variables
        self._table_name: str = table_name
        self._sentence_embedder: Optional[SentenceTransformersDocumentEmbedder] = None
        self._embedder_model_name: Optional[str] = embedder_model_name
        self._verbose: bool = verbose
        self._llm_top_k: int = llm_top_k
        self._retriever_top_k: int = max(retriever_top_k_docs or float('-inf'), llm_top_k)
        self._include_outputs_from: Optional[set[str]] = include_outputs_from
        self._search_mode: SearchMode = search_mode
        self._allow_streaming_callback: bool = False
        self._use_reranker: bool = use_reranker
        self._db_user_name: str = db_user_name
        self._db_password: str = db_password
        self._db_name: str = db_name

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
        self._document_store: Optional[Union[PgvectorDocumentStore, Neo4jDocumentStore]] = None
        self._initialize_document_store()

        if self._use_reranker:
            self._ranker: Reranker = Reranker(component_device=self._component_device)

        # Declare rag pipeline
        self._pipeline: Optional[Pipeline] = None
        # Create the RAG pipeline
        self._create_rag_pipeline()

    @property
    def llm_top_k(self) -> int:
        return self._llm_top_k

    @llm_top_k.setter
    def llm_top_k(self, value: int) -> None:
        self._llm_top_k = value
        self._retriever_top_k = max(self._retriever_top_k or float('-inf'), self._llm_top_k)
        self._create_rag_pipeline()

    @property
    def retriever_top_k(self) -> int:
        return self._retriever_top_k

    @retriever_top_k.setter
    def retriever_top_k(self, value: int) -> None:
        self._retriever_top_k = max(self._llm_top_k or float('-inf'), value)
        self._create_rag_pipeline()

    @property
    def sentence_context_length(self) -> Optional[int]:
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
    def sentence_embed_dims(self) -> Optional[int]:
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

    @property
    def verbose(self) -> bool:
        return self._verbose

    @verbose.setter
    def verbose(self, value: bool) -> None:
        self._verbose = value

    def _print_verbose(self, *args, **kwargs) -> None:
        if self._verbose:
            print(*args, **kwargs)

    def _setup_embedder(self) -> None:
        if self._sentence_embedder is None:
            if self._embedder_model_name is not None:
                self._sentence_embedder = SentenceTransformersTextEmbedder(model=self._embedder_model_name,
                                                                           device=self._component_device,
                                                                           trust_remote_code=True)
            else:
                self._sentence_embedder = SentenceTransformersTextEmbedder(device=self._component_device)

            if hasattr(self._sentence_embedder, 'warm_up'):
                self._sentence_embedder.warm_up()

    def draw_pipeline(self) -> None:
        """
        Draw and save visual representations of the RAG and document conversion pipelines.
        """
        if self._pipeline is not None:
            self._pipeline.draw(Path("Doc Retrieval Pipeline.png"))

    def _initialize_document_store(self) -> None:
        def init_doc_store() -> PgvectorDocumentStore:
            connection_token: Secret = Secret.from_token(self._postgres_connection_str)
            doc_store: PgvectorDocumentStore = PgvectorDocumentStore(
                connection_string=connection_token,
                table_name=self._table_name,
                embedding_dimension=self.sentence_embed_dims,
                vector_function="cosine_similarity",
                recreate_table=False,
                search_strategy="hnsw",
                hnsw_recreate_index_if_exists=True,
                hnsw_index_name=self._table_name + "_hnsw_index",
                keyword_index_name=self._table_name + "_keyword_index",
            )
            return doc_store

        self._document_store = init_doc_store()
        self._print_verbose("Document Count: " + str(self._document_store.count_documents()))

    def generate_response(self, query: str, min_score: float = 0.0) -> Tuple[List[Document], List[Document]]:
        """
        Generate a response to a given query using the RAG pipeline.

        Args:
            query (str): The input query to process.
            min_score (float): The minimum score for documents to be included in the response.
        """
        # Prepare inputs for the pipeline
        inputs: Dict[str, Any] = {
            "query_input": {"query": query,
                            "llm_top_k": self._llm_top_k,
                            "retriever_top_k": self._retriever_top_k},
        }

        # Run the pipeline
        results: Dict[str, Any] = self._pipeline.run(inputs, include_outputs_from=self._include_outputs_from)
        ranked_documents: List[Document] = []
        if self._use_reranker:
            ranked_documents: List[Document] = results["reranker"]["top_documents"]
            all_documents: List[Document] = results["reranker"]["all_documents"]
        else:
            all_documents: List[Document] = results["doc_query_collector"]["documents"]
        print_debug_results(results, self._include_outputs_from, verbose=self._verbose)

        if self._use_reranker:
            # Filter documents based on the minimum score
            ranked_documents = [doc for doc in ranked_documents if doc.score >= min_score]
            return ranked_documents, all_documents
        else:
            # Filter documents based on the minimum score
            all_documents = [doc for doc in all_documents if doc.score >= min_score]
            return all_documents, all_documents

    def _create_rag_pipeline(self) -> None:
        rag_pipeline: Pipeline = Pipeline()
        self._setup_embedder()

        # Pass query to the query input component
        rag_pipeline.add_component("query_input", QueryComponent())
        # Connect the query input to the query component if this is a semantic or hybrid search
        if self._search_mode == SearchMode.SEMANTIC or self._search_mode == SearchMode.HYBRID:
            rag_pipeline.add_component("query_embedder", self._sentence_embedder)
            rag_pipeline.connect("query_input.query", "query_embedder.text")

        # Add the document query collector component with an inline callback function to specify when completed
        # This is an extra way to be sure the LLM doesn't prematurely start calling the streaming callback
        doc_collector: DocumentCollector = DocumentCollector(do_stream=False,
                                                             callback_func=None)
        rag_pipeline.add_component("doc_query_collector", doc_collector)

        # Add the retriever component(s) depending on search mode
        if self._search_mode == SearchMode.LEXICAL or self._search_mode == SearchMode.HYBRID:
            lex_retriever: RetrieverWrapper = RetrieverWrapper(
                PgvectorKeywordRetriever(document_store=self._document_store))
            rag_pipeline.add_component("lex_retriever", lex_retriever)
            rag_pipeline.connect("query_input.query", "lex_retriever.query")
            rag_pipeline.connect("query_input.retriever_top_k", "lex_retriever.top_k")
            rag_pipeline.connect("lex_retriever.documents", "doc_query_collector.lexical_documents")

        if self._search_mode == SearchMode.SEMANTIC or self._search_mode == SearchMode.HYBRID:
            semantic_retriever: RetrieverWrapper
            semantic_retriever = RetrieverWrapper(
                PgvectorEmbeddingRetriever(document_store=self._document_store))
            rag_pipeline.add_component("semantic_retriever", semantic_retriever)
            rag_pipeline.connect("query_embedder.embedding", "semantic_retriever.query")
            rag_pipeline.connect("query_input.retriever_top_k", "semantic_retriever.top_k")
            rag_pipeline.connect("semantic_retriever.documents", "doc_query_collector.semantic_documents")

        if self._use_reranker:
            # Reranker
            rag_pipeline.add_component("reranker", self._ranker)
            rag_pipeline.connect("doc_query_collector.documents", "reranker.documents")
            rag_pipeline.connect("query_input.query", "reranker.query")
            rag_pipeline.connect("query_input.llm_top_k", "reranker.top_k")

        # Set the pipeline instance
        self._pipeline = rag_pipeline


def main() -> None:
    password: str = gen.get_secret(r'D:\Documents\Secrets\postgres_password.txt')
    user_name: str = "postgres"
    db_name: str = "postgres"

    include_outputs_from: Optional[set[str]] = None  # {"prompt_builder", "reranker_streamer"}
    rag_processor: DocRetrievalPipeline = DocRetrievalPipeline(table_name="popper_archive",
                                                               db_user_name=user_name,
                                                               db_password=password,
                                                               postgres_host='localhost',
                                                               postgres_port=5432,
                                                               db_name=db_name,
                                                               verbose=True,
                                                               llm_top_k=5,
                                                               retriever_top_k_docs=5,
                                                               include_outputs_from=include_outputs_from,
                                                               search_mode=SearchMode.HYBRID,
                                                               use_reranker=True,
                                                               embedder_model_name="BAAI/llm-embedder")

    if rag_processor.verbose:
        # Draw images of the pipelines
        rag_processor.draw_pipeline()
        print("Sentence Embedder Dims: " + str(rag_processor.sentence_embed_dims))
        print("Sentence Embedder Context Length: " + str(rag_processor.sentence_context_length))

    query: str = "What is your stance on coercion?"
    # "Should we strive to make our theories as severely testable as possible?"
    # "Should you ad hoc save your theory?"
    # "How are refutation, falsification, and testability related?"
    print()
    print()
    print("Query: " + query)
    print()
    # Pause for user to hit enter
    input("Press Enter to continue...")
    rag_processor.generate_response(query)


if __name__ == "__main__":
    main()
