# noinspection PyPackageRequirements
from haystack import Document, component
from typing import List, Optional, Dict, Any, Union, Callable
from collections import defaultdict
import itertools
from math import inf
import textwrap
from haystack_integrations.components.retrievers.pgvector import PgvectorEmbeddingRetriever, PgvectorKeywordRetriever
# noinspection PyPackageRequirements
from haystack.components.preprocessors import DocumentSplitter
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
# noinspection PyPackageRequirements
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from sentence_transformers import SentenceTransformer


def print_documents(documents: List[Document]) -> None:
    for i, doc in enumerate(documents, 1):
        print()
        print(f"Document {i}:")
        print(f"Score: {doc.score}")
        if hasattr(doc, 'meta') and doc.meta:
            if 'book_title' in doc.meta:
                print(textwrap.fill(f"Book Title: {doc.meta['book_title']}", width=80))
            if 'section_title' in doc.meta:
                print(textwrap.fill(f"Section Title: {doc.meta['section_title']}", width=80))
            if 'section_id' in doc.meta:
                print(textwrap.fill(f"Section ID: {doc.meta['section_id']}", width=80))
            if 'section_num' in doc.meta:
                print(textwrap.fill(f"Section #: {doc.meta['section_num']}", width=80))
            if 'paragraph_num' in doc.meta:
                print(textwrap.fill(f"Paragraph #: {doc.meta['paragraph_num']}", width=80))
        # Use text wrap to wrap the content at 80 characters
        print(textwrap.fill(f"Content: {doc.content}", width=80))
        print("-" * 50)


@component
class DocumentQueryCollector:
    def __init__(self, do_stream: bool = False, callback_func: Callable = None) -> None:
        self._do_stream: bool = do_stream
        self._callback_func: Callable = callback_func
    """
    A simple component that takes a List of Documents from the DocumentJoiner
    as well as the query and llm_top_k from the QueryComponent and returns them in a dictionary
    so that we can connect it to other components.

    This component should be unnecessary, but a bug in DocumentJoiner requires it to avoid
    strange results on streaming happening from the LLM component prior to receiving the documents.
    """
    @component.output_types(documents=List[Document], query=str, llm_top_k=int)
    def run(self, query: str,
            llm_top_k: int = 5,
            semantic_documents: Optional[List[Document]] = None,
            lexical_documents: Optional[List[Document]] = None
            ) -> Dict[str, Any]:
        documents: List[Document] = []
        # Check for semantic documents vs lexical documents and, if both exist, merge them
        if semantic_documents is not None and lexical_documents is not None:
            # Combine semantic and lexical documents. But only include each document once and take highest scores first.
            output: List[Document] = []
            document_lists: List[list] = [semantic_documents, lexical_documents]
            docs_per_id: defaultdict = defaultdict(list)
            doc: Document
            for doc in itertools.chain.from_iterable(document_lists):
                docs_per_id[doc.id].append(doc)
            docs: list
            for docs in docs_per_id.values():
                # Take the document with the best score
                doc_with_best_score = max(docs, key=lambda a_doc: a_doc.score if a_doc.score else -inf)
                # Give a slight boost to the score for each duplicate - Add .1 to the score for each duplicate
                # but adjust the 0.1 boost by score of the duplicate
                if len(docs) > 1:
                    for doc in docs:
                        if doc != doc_with_best_score:
                            doc_with_best_score.score += min(max(doc.score, 0.0), 0.1)
                output.append(doc_with_best_score)
            output.sort(key=lambda a_doc: a_doc.score if a_doc.score else -inf, reverse=True)
            documents = output
        elif semantic_documents is not None:
            documents = semantic_documents
        elif lexical_documents is not None:
            documents = lexical_documents
        if self._do_stream:
            print()
            print("Retrieved Documents:")
            print_documents(documents)
        if self._callback_func is not None:
            self._callback_func()
        return {"documents": documents, "query": query, "llm_top_k": llm_top_k}


@component
class QueryComponent:
    """
    A simple component that takes a query and llm_top_k and returns it in a dictionary so that we can connect it to
    other components.
    """
    @component.output_types(query=str, llm_top_k=int)
    def run(self, query: str, llm_top_k: int) -> Dict[str, Any]:
        return {"query": query, "llm_top_k": llm_top_k}


@component
class MergeResults:
    @component.output_types(merged_results=Dict[str, Any])
    def run(self, documents: List[Document],
            replies: List[Union[str, Dict[str, str]]]) -> Dict[str, Dict[str, Any]]:
        return {
            "merged_results": {
                "documents": documents,
                "replies": replies
            }
        }


@component
class RetrieverWrapper:
    def __init__(self, retriever: Union[PgvectorEmbeddingRetriever, PgvectorKeywordRetriever],
                 do_stream: bool = False) -> None:
        self._retriever: Union[PgvectorEmbeddingRetriever, PgvectorKeywordRetriever] = retriever
        self._do_stream: bool = do_stream
        # Alternatively, you can set the input types:
        # component.set_input_types(self, query_embedding=List[float], query=Optional[str])

    @component.output_types(documents=List[Document])
    def run(self, query: Union[List[float], str]) -> Dict[str, Any]:
        documents: List[Document] = []
        if isinstance(query, list):
            documents = self._retriever.run(query_embedding=query)['documents']
        elif isinstance(query, str):
            documents = self._retriever.run(query=query)['documents']
        if self._do_stream:
            print()
            if isinstance(self._retriever, PgvectorEmbeddingRetriever):
                print("Semantic Retriever Results:")
            elif isinstance(self._retriever, PgvectorKeywordRetriever):
                print("Lexical Retriever Results:")
            print_documents(documents)
        # Return a dictionary with documents
        return {"documents": documents}


@component
class FinalDocCounter:
    # A component that connects to both 'router' and 'writer' components and determines
    # how many, if any, documents were written to the document store. This avoids having to check if 'writer' exists
    # at the last node of the pipeline. I can always just be assured this component will exist.
    @component.output_types(documents_written=int)
    def run(self, documents_written: int = 0, no_documents: int = 0) -> Dict[str, int]:
        return {"documents_written": documents_written + no_documents}


@component
class RemoveIllegalDocs:
    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
        documents = [Document(content=doc.content, meta=doc.meta) for doc in documents if doc.content is not None]
        documents = list({doc.id: doc for doc in documents}.values())
        return {"documents": documents}


@component
class DuplicateChecker:
    def __init__(self, document_store: PgvectorDocumentStore):
        self.document_store = document_store

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
        unique_documents = []
        for doc in documents:
            if not self._is_duplicate(doc):
                unique_documents.append(doc)
        return {"documents": unique_documents}

    def _is_duplicate(self, document: Document) -> bool:
        # Use a simpler filter that checks for exact content match
        filters = {
            "field": "content",
            "operator": "==",
            "value": document.content
        }
        results = self.document_store.filter_documents(filters=filters)
        return len(results) > 0


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
            print(f"Split {len(documents)} documents into {len(processed_docs)} documents")
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