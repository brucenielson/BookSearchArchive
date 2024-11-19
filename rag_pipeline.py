# Pytorch imports
import torch
# Haystack imports
# noinspection PyPackageRequirements
from haystack import Pipeline
# noinspection PyPackageRequirements
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
# noinspection PyPackageRequirements
from haystack.components.builders import PromptBuilder
# noinspection PyPackageRequirements
from haystack.components.generators import HuggingFaceLocalGenerator
# noinspection PyPackageRequirements
from haystack.components.rankers import TransformersSimilarityRanker
# noinspection PyPackageRequirements
from haystack.dataclasses import StreamingChunk
from haystack_integrations.components.generators.google_ai import GoogleAIGeminiGenerator
from haystack_integrations.components.retrievers.pgvector import PgvectorEmbeddingRetriever, PgvectorKeywordRetriever
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
# noinspection PyPackageRequirements
from haystack.utils import ComponentDevice, Device
# noinspection PyPackageRequirements
from haystack.utils.auth import Secret
# Other imports
from typing import Optional, Dict, Any, Union
from pathlib import Path
import generator_model as gen
from enum import Enum
import textwrap
from custom_haystack_components import (MergeResults, DocumentQueryCollector, RetrieverWrapper, print_documents,
                                        QueryComponent, print_debug_results, DocumentStreamer)


class SearchMode(Enum):
    LEXICAL = 1
    SEMANTIC = 2
    HYBRID = 3


class RagPipeline:
    # The amount of text streamed since last newline.
    _streamed_text_length: int = 0

    def __init__(self,
                 table_name: str = 'haystack_pgvector_docs',
                 postgres_user_name: str = 'postgres',
                 postgres_password: str = None,
                 postgres_host: str = 'localhost',
                 postgres_port: int = 5432,
                 postgres_db_name: str = 'postgres',
                 generator_model: Union[gen.GeneratorModel, HuggingFaceLocalGenerator, GoogleAIGeminiGenerator] = None,
                 embedder_model_name: Optional[str] = None,
                 use_streaming: bool = False,
                 verbose: bool = False,
                 llm_top_k: int = 5,
                 retriever_top_k_docs: int = None,
                 search_mode: SearchMode = SearchMode.HYBRID,
                 use_reranker: bool = False,
                 include_outputs_from: Optional[set[str]] = None
                 ) -> None:

        # streaming_callback function to print to screen
        def streaming_callback(chunk: StreamingChunk) -> None:
            # Print the content of the chunks but wrap the text after 80 characters
            if self._allow_streaming_callback:
                RagPipeline._streamed_text_length += len(chunk.content)
                if RagPipeline._streamed_text_length < 80 or chunk.content in ['.', ',', ';', ':', '!', '?', ' ', '\n']:
                    print(chunk.content, end='')
                    if chunk.content == '\n':
                        RagPipeline._streamed_text_length = 0
                else:
                    print()
                    print(chunk.content.strip(), end='')
                    RagPipeline._streamed_text_length = len(chunk.content)

        # Instance variables
        self._table_name: str = table_name
        self._sentence_embedder: Optional[SentenceTransformersDocumentEmbedder] = None
        self._embedder_model_name: Optional[str] = embedder_model_name
        self._use_streaming: bool = use_streaming
        self._verbose: bool = verbose
        self._llm_top_k: int = llm_top_k
        self._retriever_top_k: int = max(retriever_top_k_docs or float('-inf'), llm_top_k)
        self._include_outputs_from: Optional[set[str]] = include_outputs_from
        self._search_mode: SearchMode = search_mode
        self._allow_streaming_callback: bool = False
        self._use_reranker: bool = use_reranker

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
        self._initialize_document_store()

        if self._use_reranker:
            # Warmup Reranker model
            # https://docs.haystack.deepset.ai/docs/transformerssimilarityranker
            ranker = TransformersSimilarityRanker(device=self._component_device, top_k=self._llm_top_k,
                                                  score_threshold=0.20)
            ranker.warm_up()
            self._ranker = ranker
            # TODO: Fix this with annotations

        if generator_model is None:
            raise ValueError("Generator model must be provided")
        self._generator_model: Optional[Union[gen.GeneratorModel, HuggingFaceLocalGenerator, GoogleAIGeminiGenerator]]
        self._generator_model = generator_model
        self._generator_model.verbose = self._verbose
        # Handle callbacks for streaming if applicable
        if self._can_stream() and self._generator_model.streaming_callback is None:
            self._generator_model.streaming_callback = streaming_callback

        # Default prompt template
        # noinspection SpellCheckingInspection
        self._prompt_template: str = (  # noqa: E101
            "<start_of_turn>user\n"
            "Using the information contained in the context where possible, "
            "give a comprehensive answer to the question. Pay more attention to passages with "
            "higher relevance scores.\n\n"
            "Context:\n"
            "{% for i in range([documents|count, llm_top_k|int] | min) | reverse %}"
                "Relevance Score {{ '%.2f' | format(documents[i].score) }}\n"
                    "{% if documents[i] is defined %}"
                        "{{ documents[i].content }}\n"
                    "{% endif %}\n"
            "{% endfor %}End of Context\n\n"
            "Question: {{query}}<end_of_turn>\n\n"
            "<start_of_turn>model\n"
        )

        # self._print_verbose("Prompt Template:")
        # self._print_verbose(self._prompt_template)

        # Declare rag pipeline
        self._rag_pipeline: Optional[Pipeline] = None
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
        if self._generator_model is not None:
            self._generator_model.verbose = value

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

    def _setup_generator(self) -> None:
        # If the generator model has a warm_up method, call it
        if hasattr(self._generator_model, 'warm_up'):
            self._generator_model.warm_up()

    def draw_pipeline(self) -> None:
        """
        Draw and save visual representations of the RAG and document conversion pipelines.
        """
        self._setup_generator()
        if self._rag_pipeline is not None:
            self._rag_pipeline.draw(Path("RAG Pipeline.png"))

    def _initialize_document_store(self) -> None:
        connection_token: Secret = Secret.from_token(self._postgres_connection_str)
        document_store: PgvectorDocumentStore = PgvectorDocumentStore(
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

        self._document_store = document_store
        self._print_verbose("Document Count: " + str(document_store.count_documents()))

    def _can_stream(self) -> bool:
        return (self._use_streaming
                and self._generator_model is not None
                and isinstance(self._generator_model, gen.GeneratorModel)
                and hasattr(self._generator_model, 'streaming_callback'))

    def generate_response(self, query: str) -> None:
        """
        Generate a response to a given query using the RAG pipeline.

        Args:
            query (str): The input query to process.
        """
        print()
        print("Generating Response...")

        # Prepare inputs for the pipeline
        inputs: Dict[str, Any] = {
            "query_input": {"query": query, "llm_top_k": self._llm_top_k},
        }

        # Run the pipeline
        if self._can_stream():
            # Document streaming and LLM streaming will be handled inside the components
            results: Dict[str, Any] = self._rag_pipeline.run(inputs, include_outputs_from=self._include_outputs_from)
            print()
            print_debug_results(results, self._include_outputs_from, verbose=self._verbose)
        else:
            results: Dict[str, Any] = self._rag_pipeline.run(inputs, include_outputs_from=self._include_outputs_from)
            print()
            print_debug_results(results, self._include_outputs_from, verbose=self._verbose)

            merged_results = results["merger"]["merged_results"]

            # Print retrieved documents
            print()
            self._print_verbose("Retrieved Documents:")
            print_documents(merged_results["documents"])

            # Print generated response
            # noinspection SpellCheckingInspection
            print("\nLLM's Response:")
            if merged_results["replies"]:
                answer: str = merged_results["replies"][0]
                print(textwrap.fill(answer, width=80))
            else:
                print("No response was generated.")

    def _create_rag_pipeline(self) -> None:
        def doc_collector_completed() -> None:
            self._allow_streaming_callback = True

        rag_pipeline: Pipeline = Pipeline()
        self._setup_embedder()
        self._setup_generator()

        # Pass query to the query input component
        rag_pipeline.add_component("query_input", QueryComponent())
        # Connect the query input to the query component if this is a semantic or hybrid search
        if self._search_mode == SearchMode.SEMANTIC or self._search_mode == SearchMode.HYBRID:
            rag_pipeline.add_component("query_embedder", self._sentence_embedder)
            rag_pipeline.connect("query_input.query", "query_embedder.text")

        # Add the document query collector component with an inline callback function to specify when completed
        # This is an extra way to be sure the LLM doesn't prematurely start calling the streaming callback
        doc_collector: DocumentQueryCollector = DocumentQueryCollector(do_stream=self._can_stream(),
                                                                       callback_func=lambda: doc_collector_completed())
        rag_pipeline.add_component("doc_query_collector", doc_collector)
        rag_pipeline.connect("query_input.query", "doc_query_collector.query")
        rag_pipeline.connect("query_input.llm_top_k", "doc_query_collector.llm_top_k")

        # Add the retriever component(s) depending on search mode
        if self._search_mode == SearchMode.LEXICAL or self._search_mode == SearchMode.HYBRID:
            lex_retriever: RetrieverWrapper = RetrieverWrapper(
                PgvectorKeywordRetriever(document_store=self._document_store, top_k=self._retriever_top_k))
            rag_pipeline.add_component("lex_retriever", lex_retriever)
            rag_pipeline.connect("query_input.query", "lex_retriever.query")
            rag_pipeline.connect("lex_retriever.documents", "doc_query_collector.lexical_documents")

        if self._search_mode == SearchMode.SEMANTIC or self._search_mode == SearchMode.HYBRID:
            semantic_retriever: RetrieverWrapper = RetrieverWrapper(
                PgvectorEmbeddingRetriever(document_store=self._document_store, top_k=self._retriever_top_k))
            rag_pipeline.add_component("semantic_retriever", semantic_retriever)
            rag_pipeline.connect("query_embedder.embedding", "semantic_retriever.query")
            rag_pipeline.connect("semantic_retriever.documents", "doc_query_collector.semantic_documents")

        if self._use_reranker:
            # Reranker
            rag_pipeline.add_component("reranker", self._ranker)
            rag_pipeline.connect("doc_query_collector.documents", "reranker.documents")
            rag_pipeline.connect("doc_query_collector.query", "reranker.query")
            rag_pipeline.connect("doc_query_collector.llm_top_k", "reranker.top_k")
            # Stream the reranked documents
            rag_pipeline.add_component("reranker_streamer", DocumentStreamer(do_stream=self._can_stream()))
            rag_pipeline.connect("reranker.documents", "reranker_streamer.documents")

        # Add the prompt builder component
        prompt_builder: PromptBuilder = PromptBuilder(template=self._prompt_template)
        rag_pipeline.add_component("prompt_builder", prompt_builder)
        rag_pipeline.connect("doc_query_collector.query", "prompt_builder.query")
        rag_pipeline.connect("doc_query_collector.llm_top_k", "prompt_builder.llm_top_k")
        if self._use_reranker:
            # Connect the reranker documents to the prompt builder
            rag_pipeline.connect("reranker_streamer.documents", "prompt_builder.documents")
        else:
            # Connect the doc collector documents to the prompt builder
            rag_pipeline.connect("doc_query_collector.documents", "prompt_builder.documents")

        # Add the LLM component
        if isinstance(self._generator_model, gen.GeneratorModel):
            rag_pipeline.add_component("llm", self._generator_model.generator_component)
        else:
            rag_pipeline.add_component("llm", self._generator_model)

        if not self._can_stream():
            # Add the final merger of documents and llm response only when streaming is disabled
            rag_pipeline.add_component("merger", MergeResults())
            rag_pipeline.connect("doc_query_collector.documents", "merger.documents")
            rag_pipeline.connect("llm.replies", "merger.replies")

        # Connect prompt builder to the llm
        rag_pipeline.connect("prompt_builder", "llm")

        # Set the pipeline instance
        self._rag_pipeline = rag_pipeline


def main() -> None:
    postgres_password = gen.get_secret(r'D:\Documents\Secrets\postgres_password.txt')
    hf_secret: str = gen.get_secret(r'D:\Documents\Secrets\huggingface_secret.txt')  # Put your path here
    google_secret: str = gen.get_secret(r'D:\Documents\Secrets\gemini_secret.txt')  # Put your path here # noqa: F841
    # model: gen.GeneratorModel = gen.HuggingFaceLocalModel(password=hf_secret, model_name="google/gemma-1.1-2b-it")
    # model: gen.GeneratorModel = gen.GoogleGeminiModel(password=google_secret)
    model: gen.GeneratorModel = gen.HuggingFaceAPIModel(password=hf_secret, model_name="HuggingFaceH4/zephyr-7b-alpha")  # noqa: E501
    # Possible outputs to include in the debug results: "lex_retriever", "semantic_retriever", "prompt_builder",
    # "joiner", "llm", "prompt_builder", "doc_query_collector"
    include_outputs_from: Optional[set[str]] = None
    rag_processor: RagPipeline = RagPipeline(table_name="popper_archive",
                                             generator_model=model,
                                             postgres_user_name='postgres',
                                             postgres_password=postgres_password,
                                             postgres_host='localhost',
                                             postgres_port=5432,
                                             postgres_db_name='postgres',
                                             use_streaming=True,
                                             verbose=True,
                                             llm_top_k=5,
                                             retriever_top_k_docs=50,
                                             include_outputs_from=include_outputs_from,
                                             search_mode=SearchMode.HYBRID,
                                             use_reranker=True,
                                             embedder_model_name="BAAI/llm-embedder")

    if rag_processor.verbose:
        # Draw images of the pipelines
        rag_processor.draw_pipeline()
        print("Generator Embedder Dims: " + str(model.embedding_dimensions))
        print("Generator Context Length: " + str(model.context_length))
        print("Sentence Embedder Dims: " + str(rag_processor.sentence_embed_dims))
        print("Sentence Embedder Context Length: " + str(rag_processor.sentence_context_length))

    query: str = "Should we strive to make our theories as severely testable as possible?"
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

# TODO: Add a reranker component to rerank the documents before passing them to the LLM
# TODO: Add a way to chat with the model
# TODO: Add graph rag pipeline
