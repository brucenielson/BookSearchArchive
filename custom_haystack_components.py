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
import re


def print_documents(documents: List[Document]) -> None:
    for i, doc in enumerate(documents, 1):
        print(f"\nDocument {i}:")
        print(f"Score: {doc.score}")

        # Dynamically iterate over all keys in doc.meta, excluding 'file_path'
        if hasattr(doc, 'meta') and doc.meta:
            for key, value in doc.meta.items():
                if key == 'file_path':  # Skip 'file_path'
                    continue
                if key == 'section_headings':
                    pass
                # Print the key-value pair, wrapped at 80 characters
                print(textwrap.fill(f"{key.replace('_', ' ').title()}: {value}", width=80))

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


def analyze_content(doc: Document, paragraph_num: int, title_line_max: int = 100) -> Dict[str, Optional[str]]:
    result: Dict[str, Optional[Union[str, int]]] = {"chapter_number": None, "chapter_title": None,
                                                    "cleaned_content": None}

    # Split the content into lines
    meta: Dict[str, str] = doc.meta
    content: str = doc.content
    section_id: str = meta.get("section_id", "").lower()
    lines: List[str] = content.split("\n", 3)  # Only split into first two lines
    first_line: str = lines[0].strip() if len(lines) > 0 else ""
    second_line: str = lines[1].strip() if len(lines) > 1 else ""

    # Check section title for the chapter number pattern if not already found
    match = re.search(r'(?<!-)(?:chapter|ch)\D*(\d+)', section_id.lower())
    if match and result["chapter_number"] is None:
        result["chapter_number"] = int(match.group(1))  # Capture the chapter number

    # Paragraph 1 is special - it may contain chapter number and title or DOI lines to remove
    if paragraph_num == 1:
        # Remove lines that start with "DOI:" on paragraph 1 - this is an unneeded line
        if first_line.lower().startswith("doi:"):
            content = content.replace(first_line, "", 1).strip()
            result["cleaned_content"] = content
            lines = content.split("\n", 3)  # Only split into first two lines
            first_line = lines[0].strip() if len(lines) > 0 else ""
            second_line = lines[1].strip() if len(lines) > 1 else ""

        # Only analyze if the first line is under title_line_max characters
        if len(first_line) < title_line_max:
            # Check if the first line is a chapter number (an integer) - we prefer this over the section_id
            if first_line.isdigit():
                result["chapter_number"] = int(first_line)
                # If first line is a lone chapter number, the second line is likely the chapter title
                if len(second_line) < title_line_max and result["chapter_title"] is None:
                    result["chapter_title"] = second_line.title()
            # Check if the first line is short enough to be a title
            elif len(first_line) < title_line_max and first_line.isupper():
                result["chapter_title"] = first_line.title()

    else:  # This is any other paragraph other than the first
        # Check if the first line is a subsection title
        # Patter is an integer followed by a period and then a title
        if len(first_line) < title_line_max:
            match = re.match(r'(\d+)\.\s*(.*)', first_line)
            if match:
                result["subsection_num"] = int(match.group(1))
                result["subsection_title"] = match.group(2).title()

    return result


@component
class CustomDocumentSplitter:
    def __init__(self,
                 embedder: SentenceTransformersDocumentEmbedder,
                 verbose: bool = True,
                 skip_content_func: Optional[callable] = None,
                 verbose_file_name: str = "first_paragraph_per_section.txt") -> None:
        self._embedder: SentenceTransformersDocumentEmbedder = embedder
        self._verbose: bool = verbose
        self._skip_content_func: Optional[callable] = skip_content_func
        self._model: SentenceTransformer = embedder.embedding_backend.model
        self._tokenizer = self._model.tokenizer
        self._max_seq_length: int = self._model.get_max_seq_length()
        # Delete "first_paragraph_section.txt"
        self._file_name: str = verbose_file_name
        if self._verbose:
            with open(self._file_name, "w", encoding="utf-8") as file:
                file.write("")

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
        processed_docs: List[Document] = []
        last_section_num: Optional[int] = None  # Track the last section number
        sections_to_skip: set = set()  # Sections to skip

        current_chapter_number: Optional[int] = None  # Store chapter number for the section
        current_chapter_title: Optional[str] = None  # Store chapter title for the section

        for doc in documents:
            # Extract section_num and paragraph_num from the metadata
            section_num: int = int(doc.meta.get("section_num"))
            paragraph_num: int = int(doc.meta.get("paragraph_num"))
            book_title: str = doc.meta.get("book_title")

            # If this is a section to skip, go to the next document
            if (book_title, section_num) in sections_to_skip:
                continue

            # If verbose is True, print the content when section_num changes and paragraph_num == 1
            # Otherwise, just save chapter info off
            if section_num != last_section_num and paragraph_num == 1:
                # Analyze the first two lines using the helper function
                analysis_results: Dict[str, Optional[str]] = analyze_content(doc, paragraph_num)

                # Update metadata with chapter number and chapter title if available
                current_chapter_number = analysis_results["chapter_number"]
                current_chapter_title = analysis_results["chapter_title"]

                if current_chapter_number is not None:
                    doc.meta["chapter_number"] = current_chapter_number

                if current_chapter_title is not None:
                    doc.meta["chapter_title"] = current_chapter_title

                # Update document content with cleaned content
                if analysis_results["cleaned_content"] is not None:
                    doc.content = analysis_results["cleaned_content"]

                if self._verbose:
                    with open(self._file_name, "a", encoding="utf-8") as file:
                        # Loop through all metadata attributes
                        for key, value in doc.meta.items():
                            if key != 'file_path':  # Skip the 'file_path' attribute
                                file.write(f"{key.replace('_', ' ').title()}: {value}\n")

                        # Write content at the end
                        file.write(f"Content:\n{doc.content}\n\n")

                # For the first paragraph, check for possible section skipping
                if self._skip_content_func is not None and self._skip_content_func(doc.content):
                    if self._verbose:
                        # Skip this section
                        print(f"Skipping section {doc.meta.get('book_title')} / {doc.meta.get('section_title')} "
                              f"due to content check")
                    sections_to_skip.add((doc.meta.get("book_title"), section_num))
                    continue

            elif section_num == last_section_num:
                # Apply stored chapter number and chapter title to all paragraphs in the same section
                if current_chapter_number is not None:
                    doc.meta["chapter_number"] = current_chapter_number

                if current_chapter_title is not None:
                    doc.meta["chapter_title"] = current_chapter_title

                if paragraph_num > 1:
                    analysis_results: Dict[str, Optional[str]] = analyze_content(doc, paragraph_num)
                    if analysis_results.get("subsection_num") is not None:
                        doc.meta["subsection_num"] = analysis_results["subsection_num"]
                    if analysis_results.get("subsection_title") is not None:
                        doc.meta["subsection_title"] = analysis_results["subsection_title"]

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
