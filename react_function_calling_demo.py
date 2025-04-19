# https://ai.google.dev/gemini-api/docs/function-calling?example=meeting
# https://medium.com/google-cloud/building-react-agents-from-scratch-a-hands-on-guide-using-gemini-ffe4621d90ae
# https://www.philschmid.de/gemini-function-calling
# New approach
# https://atamel.dev/posts/2025/04-08_simplified_function_calling_gemini/
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

# Import the secret retrieval function from a module.
from generator_model import get_secret

from typing import List, Dict, Any

# -----------------------------------------------------------------------------
# Configure API Key for Gemini 2.0 using a secret file.
# -----------------------------------------------------------------------------
secret: str = get_secret(r'D:\Documents\Secrets\gemini_secret.txt')
genai.configure(api_key=secret)

# -----------------------------------------------------------------------------
# Define function declarations for the function calling flow.
# -----------------------------------------------------------------------------

# Function declaration for performing a Wikipedia search.
search_declaration: Dict[str, Any] = {
    "name": "search",
    "description": (
        "Searches Wikipedia for a given query and returns a summary of the page if found. "
        "If the page is ambiguous or missing, returns alternative search topics."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search term to query on Wikipedia."
            }
        },
        "required": ["query"],
    },
}

# Function declaration for looking up a phrase in the most recently searched Wikipedia page.
lookup_declaration: Dict[str, Any] = {
    "name": "lookup",
    "description": (
        "Looks up a phrase within the content of the most recent Wikipedia search result "
        "and returns a snippet of context around that phrase."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "phrase": {
                "type": "string",
                "description": "The phrase to search for within the page content."
            },
        },
        "required": ["phrase"],
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


# -----------------------------------------------------------------------------
# ReActFunctionCaller Class: Implements the ReAct framework using Gemini's
# function calling capabilities.
# -----------------------------------------------------------------------------
class ReActFunctionCaller:
    # Class-level attribute annotations
    model_name: str
    generative_model: genai.GenerativeModel
    chat: ChatSession  # genai Chat session object
    tools: List[Tool]
    _search_history: List[str]
    _search_urls: List[str]
    should_continue_prompting: bool

    def __init__(self, model: str) -> None:
        """
        Initializes the ReActFunctionCaller instance by setting up the Gemini model,
        configuring the function declarations, and initializing search history.

        Args:
            model: Name of the generative model (e.g., 'gemini-2.0-flash').
        """
        self.model_name = model
        self.generative_model = genai.GenerativeModel(model)
        self.chat = self.generative_model.start_chat(history=[])

        # Define the tools with our function declarations
        self.tools = [
            Tool(
                function_declarations=[
                    FunctionDeclaration(**search_declaration),
                    FunctionDeclaration(**lookup_declaration),
                    FunctionDeclaration(**answer_declaration),
                ]
            )
        ]

        # Initialize tracking variables
        self._search_history = []
        self._search_urls = []
        self.should_continue_prompting = True

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

    def search(self, query: str) -> Dict[str, str]:
        """
        Searches for a given query using the Wikipedia API and returns a
        summary or alternative suggestions if an error occurs.

        Args:
            query: The search term to query on Wikipedia.

        Returns:
            A dictionary with the search result.
        """
        query = query.strip()
        try:
            # Attempt to retrieve a summary for the query from Wikipedia.
            summary_text: str = wikipedia.summary(query, sentences=4, auto_suggest=False)
            wiki_page = wikipedia.page(query, auto_suggest=False)
            wiki_url: str = wiki_page.url

            # Clean the summary text.
            cleaned_summary: str = self.clean(summary_text)

            # Generate a shorter version containing the first 2-3 sentences from the summary.
            observation: str = self.generate_content(
                f'Return the first 2 or 3 sentences from the following text: {cleaned_summary}'
            )

            # Record the search details for later lookup.
            self._search_history.append(query)
            self._search_urls.append(wiki_url)
            print(f"Information Source: {wiki_url}")

        except (DisambiguationError, PageError):
            # Handle ambiguous or non-existent pages.
            observation = f'Could not find ["{query}"].'
            similar_topics: List[str] = wikipedia.search(query)
            observation += f' Similar: {similar_topics}. You should search for one of those instead.'

        return {"result": observation}

    def lookup(self, phrase: str, context_length: int = 200) -> Dict[str, str]:
        """
        Looks up a given phrase within the content of the most recently
        searched Wikipedia page and returns a context snippet around it.

        Args:
            phrase: The keyword or phrase to locate within the page content.
            context_length: The number of characters before and after the phrase
                            to include in the resulting context snippet.

        Returns:
            A dictionary with the lookup result.
        """
        if not self._search_history:
            return {"result": "No previous search available for lookup."}

        try:
            # Retrieve the most recently searched page's content.
            page = wikipedia.page(self._search_history[-1], auto_suggest=False)
            content: str = self.clean(page.content)
            start_index: int = content.find(phrase)

            # Compute the context snippet boundaries.
            snippet_start: int = max(0, start_index - context_length)
            snippet_end: int = start_index + len(phrase) + context_length
            result_snippet: str = content[snippet_start:snippet_end]

            print(f"Result: {result_snippet}")
            print(f"Information Source: {self._search_urls[-1]}")
            return {"result": result_snippet}
        except Exception as e:
            return {"result": f"Error during lookup: {e}"}

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
        sources: str = ", ".join(self._search_urls) if self._search_urls else "None"
        result: str = f"Final Answer: {final_answer}\nInformation Sources: {sources}"
        print(f"Information Sources: {self._search_urls}")
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
        except ResourceExhausted as e:
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
    ) -> str:
        """
        Orchestrates a ReAct-style multi-turn conversation until the "answer" function is invoked
        or the maximum number of iterations is reached.

        Args:
            user_question (str): The question or prompt for which an answer is to be generated.
            max_iterations (int, optional): Maximum number of interactions. Defaults to 25.
            **generation_kwargs (Any): Additional keyword arguments for generation.

        Returns:
            str: The final response text from the model.
        """
        assert 0 < max_iterations <= 25, "max_iterations must be between 1 and 25"

        prompt: str = (
            "Solve this question step by step using the search, lookup, and answer functions. "
            "First, think how to approach the problem. "
            "When you have the final answer, use the answer function.\n\n"
            f"Question: {user_question}"
        )

        response: GenerateContentResponse = self.send_chat_message(prompt, **generation_kwargs)

        for iteration in range(1, max_iterations + 1):
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
                    if name == "answer":
                        return result['result']
                except AttributeError:
                    result = {"result": f"Unknown function: {name}"}

                print(f"\nObservation {iteration}: {result['result']}")
                response = self.send_chat_message(result['result'], **generation_kwargs)
            else:
                print(f"\nNo function call detected. Ending conversation.")
                return getattr(response, 'text', None) or ''

        return "Maximum iterations reached."


# -----------------------------------------------------------------------------
# Example usage of the ReActFunctionCaller class.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Define the question you want to ask.
    question: str = (
        "What are the total of ages of the main trio from the new Percy Jackson and the Olympians TV series "
        "in real life?"
    )

    # Uncomment alternate questions as needed.
    # question = "How many companions did Samuel Meladon have?"
    # question = "What is the most famous case of reincarnation in the world?"
    # question = "What are the total ages of everyone in the movie Star Wars: A New Hope?"

    # Instantiate the ReActFunctionCaller session using the defined model.
    gemini_react: ReActFunctionCaller = ReActFunctionCaller(model='gemini-2.0-flash')

    # Start the conversation using the provided question and generation parameters.
    # Feel free to adjust generation_kwargs such as temperature for varied results.
    answer: str = gemini_react(question, temperature=0.2)
    print()
    print(f"\nFinal Answer:")
    print(answer)
