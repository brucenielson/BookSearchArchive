from __future__ import annotations
import wikipedia
# noinspection PyPackageRequirements
from google.api_core.exceptions import ResourceExhausted
import time
from wikipedia.exceptions import DisambiguationError, PageError
# noinspection PyPackageRequirements
import google.generativeai as genai
# noinspection PyPackageRequirements
from google.generativeai.types import Tool, FunctionDeclaration
# noinspection PyPackageRequirements
from google.generativeai.types.generation_types import GenerationConfig, GenerateContentResponse
# noinspection PyPackageRequirements
from google.generativeai import ChatSession
from typing import List, Dict, Any, Tuple
from doc_retrieval_pipeline import DocRetrievalPipeline
# noinspection PyPackageRequirements
from haystack import Document
from generator_model import get_secret


def format_document(doc, include_raw_info: bool = False) -> str:
    """
    Format a document by including its metadata followed by the quote.
    """
    formatted = ""
    # If the document has metadata, format each key-value pair.
    if hasattr(doc, 'meta') and doc.meta:
        meta_entries = ["Score: {:.4f}".format(doc.score) if hasattr(doc, 'score') else "Score: N/A"]
        # Add the original score if requested.
        if include_raw_info and hasattr(doc, 'orig_score'):
            meta_entries.append("Original Score: {:.4f}".format(doc.orig_score))
            meta_entries.append("Retrieved By: {}".format(doc.retrieval_method))
        # Define keys to ignore (customize as needed).
        ignore_keys = {"file_path", "item_#", "item_id"}
        for key, value in doc.meta.items():
            if key.lower() in ignore_keys or key.startswith('_') or key.startswith('split'):
                continue
            # Append each key-value pair.
            meta_entries.append(f"{key.replace('_', ' ').title()}: {value}")
        if meta_entries:
            formatted += "\n".join(meta_entries) + "\n"
    # Append the main quote.
    formatted += f"Quote: {doc.content}"
    return formatted


# Function declaration for performing a Wikipedia search.
search_wikipedia_declaration: Dict[str, Any] = {
    "name": "search_wikipedia",
    "description": (
        "Pass a Wikipedia page to search for and a question to be answered from that page. "
        "This will search Wikipedia for that page and either return an answer to the question from that page or "
        "if the page is ambiguous or missing, returns alternative search topics. "
        "Only use this function if you failed to find what you need from the datastore. Do not use this function "
        "before Action 6, as the datastore will provider better answers. Only use this function if you are "
        "you make it to Action 6 and you are not able to find what you need from the datastore. Going to Wikipedia "
        "is a last resort."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "wiki_page_search": {
                "type": "string",
                "description": "The search term to query on Wikipedia."
            },
            "question": {
                "type": "string",
                "description": "The question to be answered using the search results."
            },
        },
        "required": ["wiki_page_search", "question"],
    },
}


# Function declaration for searching within the document store
search_datastore_declaration: Dict[str, Any] = {
    "name": "search_datastore",
    "description": (
        "Pass phrase or question to search for within the document datastore to try to find an answer to "
        "the user's query. This function allows you to try to refine the search within the datastore or even look up "
        "multiple different datastore entries to use together to synthesize an answer not found in one spot. "
        "You should try to use this function before you use the search_wikipedia function, as it is more likely to "
        "return an authentic answer. "
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "datastore_query": {
                "type": "string",
                "description": "The search phrase to query the datastore."
            },
        },
        "required": ["datastore_query"],
    },
}


# Function declaration for providing the final answer.
answer_declaration: Dict[str, Any] = {
    "name": "answer",
    "description": (
        "Provides the final answer to the question and ends the conversation, while listing "
        "all the used information sources."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "final_answer": {
                "type": "string",
                "description": "The final answer produced by the model."
            }
        },
        "required": ["final_answer"],
    },
}


class ReActAgent:
    # Class-level attribute annotations
    model_name: str
    generative_model: genai.GenerativeModel
    chat: ChatSession  # genai Chat session object
    tools: List[Tool]
    _wikipedia_search_history: List[str]
    _wikipedia_search_urls: List[str]
    should_continue_prompting: bool

    def __init__(self, doc_retriever: DocRetrievalPipeline,
                 model: str = 'gemini-2.0-flash-exp',
                 password: str = None) -> None:
        if password:
            genai.configure(api_key=password)

        self.model_name: str = model
        self._doc_retriever: DocRetrievalPipeline = doc_retriever
        self.generative_model: genai.GenerativeModel = genai.GenerativeModel(model)
        self.chat: ChatSession = self.generative_model.start_chat(history=[])

        # Define the tools with our function declarations
        self.tools = [
            Tool(
                function_declarations=[
                    FunctionDeclaration(**search_wikipedia_declaration),
                    FunctionDeclaration(**search_datastore_declaration),
                    FunctionDeclaration(**answer_declaration),
                ]
            )
        ]

        # Initialize tracking variables
        self._wikipedia_search_history: List[str] = []
        self._wikipedia_search_urls: List[str] = []
        self._datastore_search_history: List[str] = []
        self._used_documents: List[Document] = []
        self.should_continue_prompting: bool = True
        self._iteration: int = 0

    @staticmethod
    def clean(text: str) -> str:
        """
        Cleans the input text by replacing newline characters with spaces.

        Args:
            text: The text to be cleaned.

        Returns:
            A single-line version of the input text.
        """
        return text.replace("\n", " ")

    def search_wikipedia(self, wiki_page_search: str, question: str) -> Dict[str, str]:
        wiki_page_search = wiki_page_search.strip()
        question = question.strip()
        try:
            # Attempt to retrieve a summary for the query from Wikipedia.
            wiki_page: wikipedia.WikipediaPage = wikipedia.page(wiki_page_search, auto_suggest=False)
            wiki_url: str = wiki_page.url

            # Clean the summary text.
            wiki_page_text: str = self.clean(wiki_page.content)

            # Generate a shorter version containing the first 2-3 sentences from the summary.
            observation: str = self.generate_content(
                f"You are looking for an answer to the question: '{question}'. \n"
                f"If the following text has an answer to that question, return an answer. "
                f"Otherwise return 'Answer Not Found': \n"
                f"{wiki_page_text}"
            )

            # Record the search details for later lookup.
            self._wikipedia_search_history.append(wiki_page_search)
            self._wikipedia_search_urls.append(wiki_url)
            print(f"Information Source: {wiki_url}")

        except (DisambiguationError, PageError):
            # Handle ambiguous or non-existent pages.
            observation = f'Could not find ["{wiki_page_search}"].'
            similar_topics: List[str] = wikipedia.search(wiki_page_search)
            observation += f' Similar: {similar_topics}. You should search for one of those instead.'

        # If a search for wikipedia was requested prematurely, still return the result, but remind the
        # model that it should not have used Wikipedia yet.
        if self._iteration <= 5:
            observation = ("You illegally requested Wikipedia information before Action 6. I'll give you the result, "
                           "but please try using the datastore first before answering, at least until Action 6:\n"
                           + observation)

        return {"result": observation}

    def search_datastore(self, datastore_query: str) -> Dict[str, str]:
        """
        Searches for a given query in the datastore and returns the result.

        Args:
            datastore_query: The search term to query on the datastore.

        Returns:
            A dictionary with the search result.
        """
        # Placeholder for actual datastore search logic
        self._datastore_search_history.append(datastore_query)
        retrieved_docs: List[Document]
        all_docs: List[Document]
        retrieved_docs, all_docs = self._doc_retriever.generate_response(datastore_query, min_score=0.6)
        # Add to the retrieved documents list what the search query was that found it
        for doc in retrieved_docs:
            if hasattr(doc, 'meta') and doc.meta:
                doc.meta['search_query'] = datastore_query
        self._used_documents.extend(retrieved_docs)
        # Format each retrieved document (quote + metadata).
        formatted_docs = [format_document(doc) for doc in retrieved_docs]
        retrieved_quotes = "\n\n".join(formatted_docs)
        # Generate a response based on the retrieved documents.
        observation: str = self.generate_content(
            f"Attempt to answer the question '{datastore_query}' using the following information. "
            f"If you can't answer the question entirely, answer it partially. If you can't even "
            f"give a partial answer, return 'Answer Not Found':\n{retrieved_quotes}"
        )
        return {"result": observation}

    def answer(self, final_answer: str) -> Dict[str, str]:
        """
        Marks the end of the conversation and returns the final answer along
        with sources.

        Args:
            final_answer: The final answer text.

        Returns:
            A dictionary with the final answer and sources.
        """
        self.should_continue_prompting = False
        sources: str = ", ".join(self._wikipedia_search_urls) if self._wikipedia_search_urls else "None"
        result: str = f"Answer: {final_answer}"
        print(f"Information Sources: {sources}")
        return {"result": result}

    def send_chat_message(self, message: str, **generation_kwargs: Any) -> GenerateContentResponse:
        """
        Sends a message to the chat session and returns the response.

        Args:
            message: The message to send.

        Returns:
            The response from the chat session.
        """
        # Set up the config with any provided generation parameters
        config: GenerationConfig = GenerationConfig(**generation_kwargs)

        try:
            return self.chat.send_message(message,
                                          generation_config=config,
                                          tools=self.tools)
        except ResourceExhausted:
            # Handle rate limit errors by pausing and retrying
            print()
            print(f"Rate limit exceeded. Retrying in 15 seconds...")
            time.sleep(15)
            return self.send_chat_message(message, **generation_kwargs)
        except Exception as e:
            print(f"Error during chat message sending: {e}")
            raise

    def generate_content(self, prompt: str, **generation_kwargs: Any) -> str:
        """
        Generates content using the generative model.

        Args:
            prompt: The prompt to send to the generative model.

        Returns:
            The generated text response.
        """
        config: GenerationConfig = GenerationConfig(**generation_kwargs)

        try:
            response = self.generative_model.generate_content(
                prompt,
                generation_config=config
            )
            return getattr(response, "text", None) or "[No response text]"
        except ResourceExhausted:
            print()
            print(f"Rate limit exceeded. Retrying in 15 seconds...")
            time.sleep(15)
            return self.generate_content(prompt, **generation_kwargs)
        except Exception as e:
            print(f"Error during content generation: {e}")
            raise

    def __call__(
            self,
            user_question: str,
            max_iterations: int = 25,
            **generation_kwargs: Any,
    ) -> Tuple[str, List[Document]]:
        """
        Orchestrates a ReAct-style multi-turn conversation until the "answer" function is invoked
        or the maximum number of iterations is reached.

        Args:
            user_question (str): The question or prompt for which an answer is to be generated.
            max_iterations (int, optional): Maximum number of interactions. Defaults to 25.
            **generation_kwargs (Any): Additional keyword arguments for generation.

        Returns:
            Tuple[str, List[Document]]: The final answer and the list of documents used.
        """
        assert 0 < max_iterations <= 25, "max_iterations must be between 1 and 25"

        prompt: str = (
            "Solve this question step by step using the search_datastore and answer functions. "
            "You can also use the search_wikipedia function starting at 'Action 6' if you can't find the answer in the "
            "datastore before then. But try to avoid using Wikipedia unless you have to. "
            "First, think how to approach the problem. "
            "When you have the final answer, use the answer function.\n\n"
            f"Question: {user_question}"
        )

        response: GenerateContentResponse = self.send_chat_message(prompt, **generation_kwargs)

        for iteration in range(1, max_iterations + 1):
            self._iteration = iteration
            # Ensure there's a valid response
            if not getattr(response, 'candidates', None):
                print("\nNo valid response from model.\n")
                break

            candidate = response.candidates[0]
            text_parts: List[str] = []
            func_call = None

            for part in candidate.content.parts:
                if getattr(part, 'text', None):
                    text_parts.append(part.text)
                if getattr(part, 'function_call', None):
                    func_call = part.function_call

            if text_parts:
                print(f"\nThought {iteration}: {' '.join(text_parts)}")

            if func_call:
                name: str = func_call.name
                args: Dict[str, Any] = func_call.args or {}
                print(f"\nAction {iteration}: <{name}>{list(args.values())[0] if args else ''}</{name}>")

                try:
                    method = getattr(self, name)
                    # Dynamically call the method with the provided arguments
                    result: Dict[str, str] = method(**args)
                except AttributeError:
                    result = {"result": f"Unknown function: {name}"}
                except Exception as e:
                    result = {"result": f"Error calling function {name}: {e}"}

                observation: str = result['result']
                if name == "answer":
                    return observation, self._used_documents
                if observation.strip() == "Answer Not Found":
                    if self._iteration < 5:
                        observation = observation.strip() + (". (Wikipedia is not yet available through the "
                                                             "search_wikipedia function. Do not use it yet.)\n")
                    else:
                        observation = observation.strip() + (". (Wikipedia is now available through the "
                                                             "search_wikipedia function.)\n")
                print(f"\nObservation {iteration}: {observation}")
                response = self.send_chat_message(observation, **generation_kwargs)
            else:
                print(f"\nNo function call detected. Ending conversation.")
                return getattr(response, 'text', None) or '', self._used_documents

        return "Maximum iterations reached.", self._used_documents


# -----------------------------------------------------------------------------
# Example usage of the ReActFunctionCaller class.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Define the question you want to ask.
    # users_question: str = (
    #     "What are the total of ages of the main trio from the new Percy Jackson and the Olympians TV series "
    #     "in real life?"
    # )
    # Uncomment alternate questions as needed.
    # users_question = "How many companions did Samuel Meladon have?"
    # users_question = "What is the most famous case of reincarnation in the world?"
    # users_question = "What are the total ages of everyone in the movie Star Wars: A New Hope?"
    # users_question = "Who was the mayor of Reykjavik in 2015 and what political party did they represent?"
    users_question = "Is induction valid in some cases, particularly when doing statistics?"
    # users_question = "What is Stephen King's birthday?"

    postgres_user_name: str = "postgres"
    postgres_db_name: str = "postgres"
    postgres_table_name: str = "popper_archive"
    postgres_host: str = 'localhost'
    postgres_port: int = 5432
    postgres_table_recreate: bool = False
    postgres_table_embedder_model_name: str = "BAAI/llm-embedder"
    db_password: str = get_secret(r'D:\Documents\Secrets\postgres_password.txt')
    gemini_password: str = get_secret(r'D:\Documents\Secrets\gemini_secret.txt')
    genai.configure(api_key=gemini_password)

    document_retriever: DocRetrievalPipeline = DocRetrievalPipeline(
        table_name=postgres_table_name,
        db_user_name=postgres_user_name,
        db_password=db_password,
        postgres_host=postgres_host,
        postgres_port=postgres_port,
        db_name=postgres_db_name,
        verbose=False,
        llm_top_k=5,
        retriever_top_k_docs=100,
        include_outputs_from=None,
        use_reranker=True,
        embedder_model_name=postgres_table_embedder_model_name,
    )

    # Instantiate the ReActFunctionCaller session using the defined model.
    gemini_react: ReActAgent = ReActAgent(doc_retriever=document_retriever)

    # Start the conversation using the provided question and generation parameters.
    # Feel free to adjust generation_kwargs such as temperature for varied results.
    answer: str
    docs: List[Document]
    answer, docs = gemini_react(users_question, temperature=0.2)
    print()
    print(f"\nFinal Answer:")
    print(answer)
