import csv
from ebooklib import ITEM_DOCUMENT, epub
# noinspection PyPackageRequirements
from haystack import Document, component
from typing import List, Optional, Dict, Any, Union, Callable, Tuple, Set
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
from html_parser import HTMLParser
# noinspection PyPackageRequirements
from haystack.dataclasses import ByteStream
from pathlib import Path
from pypdf import PdfReader, DocumentInformation


def print_debug_results(results: Dict[str, Any],
                        include_outputs_from: Optional[set[str]] = None,
                        verbose: bool = True) -> None:
    level: int = 1
    if verbose and include_outputs_from is not None:
        # Exclude excess outputs
        results_filtered = {k: v for k, v in results.items() if k in include_outputs_from}
        if results_filtered:
            print()
            print("Debug Results:")
            # Call the recursive function to print the results hierarchically
            _print_hierarchy(results_filtered, level)


def _print_hierarchy(data: Dict[str, Any], level: int) -> None:
    for key, value in data.items():
        # Print the key with the corresponding level
        if level == 1:
            print()
        print(f"Level {level}: {key}")

        # Check if the value is a dictionary
        if isinstance(value, dict):
            _print_hierarchy(value, level + 1)
        # Check if the value is a list
        elif isinstance(value, list):
            for index, item in enumerate(value):
                print(f"Level {level + 1}: Item {index + 1}")  # Indicating it's an item in a list
                if isinstance(item, dict):
                    _print_hierarchy(item, level + 2)
                else:
                    print(item)  # Print the item directly
        else:
            # If the value is neither a dict nor a list, print it directly
            print(value)


@component
class PDFReader:
    def __init__(self, min_page_size: int = 1000):
        self._min_page_size = min_page_size

    @component.output_types(documents=List[Document])
    def run(self, sources: List[str]) -> Dict[str, List[Document]]:
        # result = PyPDFToDocument().run(sources)
        documents: List[Document] = []
        for source in sources:
            pdf_reader = PdfReader(source)
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if len(page_text) < self._min_page_size:
                    continue
                meta_properties: List[str] = ["author", "title", "subject"]
                meta: Dict[str, Any] = PDFReader._create_meta_data(pdf_reader.metadata, meta_properties)
                meta["page_#"] = page_num + 1
                if not meta.get("title"):
                    # Use file name for title if none found in metadata
                    source_title: str = Path(source).stem
                    meta["title"] = source_title

                documents.append(Document(content=page_text, meta=meta))

        return {"documents": documents}

    @staticmethod
    def _create_meta_data(pdf_meta_data: DocumentInformation, meta_data_titles: List[str]) -> Dict[str, str]:
        meta_data: Dict[str, str] = {}
        for title in meta_data_titles:
            value: str = getattr(pdf_meta_data, title, "")
            if hasattr(pdf_meta_data, title) and value is not None and value != "":
                meta_data[title] = getattr(pdf_meta_data, title, "")
        return meta_data


@component
class EPubPdfMerger:
    @component.output_types(documents=List[Document])
    def run(self, epub_docs: List[Document], pdf_docs: List[Document]) -> Dict[str, List[Document]]:
        documents: List[Document] = []
        for doc in epub_docs:
            documents.append(doc)
        for doc in pdf_docs:
            documents.append(doc)
        return {"documents": documents}


@component
class EpubVsPdfSplitter:
    @component.output_types(epub_paths=List[str], pdf_paths=List[str])
    def run(self, file_paths: List[str]) -> Dict[str, List[str]]:
        epub_paths: List[str] = []
        pdf_paths: List[str] = []
        for file_path in file_paths:
            if file_path.lower().endswith('.epub'):
                epub_paths.append(file_path)
            elif file_path.lower().endswith('.pdf'):
                pdf_paths.append(file_path)
            else:
                raise ValueError(f"File type not supported: {file_path}")
        return {"epub_paths": epub_paths, "pdf_paths": pdf_paths}


@component
class EPubLoader:
    def __init__(self, verbose: bool = False, skip_file: str = "sections_to_skip.csv") -> None:
        self._verbose: bool = verbose
        self._is_directory: bool = False
        self._file_paths: List[str] = []
        self._skip_file: str = skip_file
        self._sections_to_skip: Dict[str, Set[str]] = {}

    @component.output_types(html_pages=List[str], meta=List[Dict[str, str]])
    def run(self, file_paths: Union[List[str], List[Path], str]) -> Dict[str, Any]:
        # Handle not documents passed in
        if len(file_paths) == 0:
            return {"html_pages": [], "meta": []}
        # Handle passing in a string with a path instead of a list of paths
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        # Handle passing in a list of Path objects instead of a list of strings
        if isinstance(file_paths, list) and isinstance(file_paths[0], Path):
            file_paths = [str(file_path) for file_path in file_paths]
        # Verify that every single file path ends with .epub
        if not all(file_path.lower().endswith('.epub') for file_path in file_paths):
            raise ValueError("EpubLoader only accepts .epub files.")
        self._file_paths = file_paths
        self._sections_to_skip = self._load_sections_to_skip()
        # Load the EPUB file
        html_pages: List[str]
        meta: List[Dict[str, str]]
        html_pages, meta = self._load_file()
        return {"html_pages": html_pages, "meta": meta}

    def _load_file(self) -> Tuple[List[str], List[Dict[str, str]]]:
        sources: List[str] = []
        meta: List[Dict[str, str]] = []
        for file_path in self._file_paths:
            sources_temp: List[str]
            meta_temp: List[Dict[str, str]]
            sources_temp, meta_temp = self._load_epub(file_path)
            sources.extend(sources_temp)
            meta.extend(meta_temp)
        return sources, meta

    def _load_epub(self, file_path: str) -> Tuple[List[str], List[Dict[str, str]]]:
        book: epub.EpubBook = epub.read_epub(file_path)
        self._print_verbose()
        self._print_verbose(f"Loaded Book: {book.title}")
        book_meta_data: Dict[str, str] = {
            "book_title": book.title,
            "file_path": file_path
        }
        i: int
        item: epub.EpubHtml
        html_pages: List[str] = []
        meta_data: List[Dict[str, str]] = []
        for i, item in enumerate(book.get_items_of_type(ITEM_DOCUMENT)):
            if item.id not in self._sections_to_skip.get(book.title, set()):
                item_meta_data: Dict[str, str] = {
                    "item_id": item.id
                }
                book_meta_data.update(item_meta_data)
                item_html: str = item.get_body_content().decode('utf-8')
                html_pages.append(item_html)
                meta_data.append(book_meta_data.copy())
            else:
                self._print_verbose(f"Book: {book.title}; Section Id: {item.id}. User Skipped.")

        return html_pages, meta_data

    def _print_verbose(self, *args, **kwargs) -> None:
        if self._verbose:
            print(*args, **kwargs)

    def _load_sections_to_skip(self) -> Dict[str, Set[str]]:
        sections_to_skip: Dict[str, Set[str]] = {}
        if self._is_directory:
            csv_path = Path(self._file_paths[0]) / self._skip_file
        else:
            # Get the directory of the file and then look for the csv file in that directory
            csv_path = Path(self._file_paths[0]).parent / self._skip_file

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


@component
class HTMLParserComponent:
    def __init__(self, min_paragraph_size: int = 300, min_section_size: int = 1000, verbose: bool = False) -> None:
        self._min_section_size: int = min_section_size
        self._min_paragraph_size: int = min_paragraph_size
        self._verbose: bool = verbose
        self._sections_to_skip: Dict[str, Set[str]] = {}

    @component.output_types(sources=List[ByteStream], meta=List[Dict[str, str]])
    def run(self, html_pages: List[str], meta: List[Dict[str, str]]) -> Dict[str, Any]:
        docs_list: List[ByteStream] = []
        meta_list: List[Dict[str, str]] = []
        included_sections: List[str] = []
        missing_chapter_titles: List[str] = []
        section_num: int = 1

        for i, html_page in enumerate(html_pages):
            page_meta_data: Dict[str, str] = meta[i]
            parser: HTMLParser
            item_id: str = page_meta_data.get("item_id", "").lower()
            if item_id.startswith('notes'):
                parser = HTMLParser(html_page, page_meta_data, min_paragraph_size=self._min_paragraph_size * 2,
                                    double_notes=False)  # If we're already doubling size, don't have parser do it too.
            else:
                parser = HTMLParser(html_page, page_meta_data, min_paragraph_size=self._min_paragraph_size,
                                    double_notes=True)

            temp_docs: List[ByteStream]
            temp_meta: List[Dict[str, str]]
            temp_docs, temp_meta = parser.run()
            item_id: str = page_meta_data.get("item_id", "")
            book_title: str = page_meta_data.get("book_title", "")
            if (parser.total_text_length() > self._min_section_size
                    and item_id not in self._sections_to_skip.get(book_title, set())):
                self._print_verbose(f"Book: {book_title}; Section {section_num}. "
                                    f"Chapter Title: {parser.chapter_title}. "
                                    f"Length: {parser.total_text_length()}")
                # Add section number to metadata
                [meta.update({"item_#": str(section_num)}) for meta in temp_meta]
                docs_list.extend(temp_docs)
                meta_list.extend(temp_meta)
                included_sections.append(book_title + ", " + item_id)
                section_num += 1
                if parser.chapter_title is None or parser.chapter_title == "":
                    missing_chapter_titles.append(book_title + ", " + item_id)
            else:
                self._print_verbose(f"Book: {book_title}; Chapter Title: {parser.chapter_title}. "
                                    f"Length: {parser.total_text_length()}. Skipped.")

        if len(docs_list) > 0:
            self._print_verbose(f"Sections included:")
            for item in included_sections:
                self._print_verbose(item)
            if missing_chapter_titles:
                self._print_verbose()
                self._print_verbose(f"Sections missing chapter titles:")
                for item in missing_chapter_titles:
                    self._print_verbose(item)
            self._print_verbose()
        return {"sources": docs_list, "meta": meta_list}

    def _print_verbose(self, *args, **kwargs) -> None:
        if self._verbose:
            print(*args, **kwargs)


def print_documents(documents: List[Document]) -> None:
    ignore_keys: set = {'file_path', 'source_id'}
    for i, doc in enumerate(documents, 1):
        print(f"\nDocument {i}:")
        print(f"Score: {doc.score}")

        # Dynamically iterate over all keys in doc.meta, excluding 'file_path'
        if hasattr(doc, 'meta') and doc.meta:
            for key, value in doc.meta.items():
                if key.lower() in ignore_keys or key.startswith('_') or key.startswith('split'):
                    continue
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
    # Temporarily disable this function
    return result

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
                 verbose_file_name: str = "documents.txt") -> None:
        self._embedder: SentenceTransformersDocumentEmbedder = embedder
        self._verbose: bool = verbose
        self._skip_content_func: Optional[callable] = skip_content_func
        self._model: SentenceTransformer = embedder.embedding_backend.model
        self._tokenizer = self._model.tokenizer
        self._max_seq_length: int = self._model.get_max_seq_length()
        # Delete verbose txt file
        self._pre_file_name: str = "pre_" + verbose_file_name
        self._post_file_name: str = "post_" + verbose_file_name
        if self._verbose:
            with open(self._pre_file_name, "w", encoding="utf-8") as file:
                file.write("")
            with open(self._post_file_name, "w", encoding="utf-8") as file:
                file.write("")

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
        processed_docs: List[Document] = []
        last_item_num: Optional[int] = None  # Track the last section number
        sections_to_skip: set = set()  # Sections to skip

        current_chapter_number: Optional[int] = None  # Store chapter number for the section
        current_chapter_title: Optional[str] = None  # Store chapter title for the section

        for doc in documents:
            # Extract item_num and paragraph_num from the metadata
            item_num: Optional[int] = int(doc.meta.get("item_#")) if doc.meta.get("item_#") is not None else None
            paragraph_num: Optional[int] = int(doc.meta.get("paragraph_#")) \
                if doc.meta.get("paragraph_#") is not None else None
            book_title: str = doc.meta.get("book_title")

            # If this is a section to skip, go to the next document
            if (book_title, item_num) in sections_to_skip:
                continue

            # If verbose is True, print the content when item_num changes and paragraph_num == 1
            # Otherwise, just save chapter info off
            if True or item_num != last_item_num and paragraph_num == 1:
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

                self.write_verbose_file(doc, file_name=self._pre_file_name)

                # For the first paragraph, check for possible section skipping
                if self._skip_content_func is not None and self._skip_content_func(doc.content):
                    if self._verbose:
                        # Skip this section
                        print(f"Skipping section {doc.meta.get('book_title')} / {doc.meta.get('section_title')} "
                              f"due to content check")
                    sections_to_skip.add((doc.meta.get("book_title"), item_num))
                    continue

            elif item_num == last_item_num:
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

            # Update the last_item_num
            last_item_num = item_num

            # Process and extend documents
            processed_docs.extend(self.process_document(doc))

        if self._verbose:
            print(f"Split {len(documents)} documents into {len(processed_docs)} documents")
        self.write_verbose_file(processed_docs, file_name=self._post_file_name)
        return {"documents": processed_docs}

    def write_verbose_file(self, documents: Union[Document, List[Document]], file_name: str = "documents.txt") -> None:
        if self._verbose:
            if isinstance(documents, Document):
                documents = [documents]
            for doc in documents:
                with open(file_name, "a", encoding="utf-8") as file:
                    # Loop through all metadata attributes
                    key: str
                    value: str
                    for key, value in doc.meta.items():
                        if isinstance(key, str):
                            key = key.strip()
                        if isinstance(value, str):
                            value = value.strip()
                        if key not in ['file_path', '_split_overlap', 'source_id', 'split_id', 'split_idx_start',
                                       'page_number']:
                            file.write(f"{key.replace('_', ' ').title()}: {value}\n")

                    # Write content at the end
                    file.write(f"Content:\n{doc.content}\n\n")

    def process_document(self, document: Document) -> List[Document]:
        token_count = self.count_tokens(document.content)
        if token_count <= self._max_seq_length:
            # Document fits within max sequence length, no need to split
            return [document]

        # Document exceeds max sequence length, find optimal split_length
        split_docs = self.find_optimal_split(document)
        return split_docs

    def find_optimal_split(self, document: Document) -> List[Document]:
        import re
        from typing import List

        def split_into_sentences(text: str) -> List[str]:
            # Define a pattern to split sentences while preserving spaces and newlines
            pattern = r'(?:(?<=\.)|(?<=\!)|(?<=\?))([\'"”’]?\s*)(?=[A-Z])|((?<=\.)|(?<=\!)|(?<=\?))([\'"”’]?\s*)(?=$)|(\n)'  # noqa: W605

            # Split the text using the pattern, this keeps delimiters (spaces/newlines) in the list
            units = re.split(pattern, text)

            # Filter out any None values that might occur in the result
            units = [unit for unit in units if unit is not None]

            # Combine text and delimiter into a single unit (pairs of text and spaces/newlines)
            combined_units = []
            i = 0
            while i < len(units):
                # Combine pairs of text and space/newline where applicable
                if i + 1 < len(units):
                    combined_units.append(units[i] + units[i + 1])
                    i += 2
                else:
                    combined_units.append(units[i])
                    i += 1

            return combined_units

        split_length = 10  # Start with 10 sentences
        while split_length > 0:
            splitter = DocumentSplitter(
                split_by="function",
                split_length=split_length,
                split_overlap=min(1, split_length - 1),
                split_threshold=min(3, split_length),
                splitting_function=split_into_sentences
            )
            split_docs = splitter.run(documents=[document])["documents"]

            # Check if all split documents fit within max_seq_length
            if all(self.count_tokens(doc.content) <= self._max_seq_length for doc in split_docs):
                return split_docs

            # If not, reduce split_length and try again
            split_length -= 1

        # If we get here, at least one sentence exceed max_seq_length, so we couldn't find a good split
        # So return splitter doing a word split
        chunk_size = 50
        split_length = self._max_seq_length//2
        while split_length > 0:
            splitter = DocumentSplitter(
                split_by="word",
                split_length=split_length,
                split_overlap=min(chunk_size, split_length // 4),
                split_threshold=min(chunk_size * 2, split_length // 2)
            )
            split_docs = splitter.run(documents=[document])["documents"]
            if all(self.count_tokens(doc.content) <= self._max_seq_length for doc in split_docs):
                return split_docs
            split_length -= chunk_size

        # So just let the splitter truncate the document
        # But give warning that document was truncated
        if self._verbose:
            print(f"Document was truncated to fit within max sequence length of {self._max_seq_length}: "
                  f"Actual length: {self.count_tokens(document.content)}")
            print(f"Problem Document: {document.content}")
        return [document]

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text, verbose=False))
