# https://ai.google.dev/gemini-api/docs/function-calling?example=meeting

from __future__ import annotations

import wikipedia
from wikipedia.exceptions import DisambiguationError, PageError

# Import Gemini generative AI and types for function calling.
# noinspection PyPackageRequirements
import google.generativeai as genai
# noinspection PyPackageRequirements
from google.genai import types
# noinspection PyPackageRequirements
from google.generativeai.types import Tool, FunctionDeclaration
# noinspection PyPackageRequirements
from google.generativeai.types.generation_types import GenerationConfig

# Import the secret retrieval function.
from generator_model import get_secret

# -----------------------------------------------------------------------------
# Configure API Key for Gemini 2.0 using a secret file.
# -----------------------------------------------------------------------------
secret = get_secret(r'D:\Documents\Secrets\gemini_secret.txt')
genai.configure(api_key=secret)

# -----------------------------------------------------------------------------
# Define function declarations for the built-in function calling flow.
#
# Each function declaration follows the OpenAPI-compatible schema.
# -----------------------------------------------------------------------------

# Function declaration for performing a Wikipedia search.
wikipedia_search_declaration = {
    "name": "wikipedia_search",
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
wikipedia_lookup_declaration = {
    "name": "wikipedia_lookup",
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
            "context_length": {
                "type": "integer",
                "description": (
                    "The number of characters before and after the found phrase to include "
                    "in the result snippet. Defaults to 200 if not provided."
                ),
            },
        },
        "required": ["phrase"],
    },
}

# Function declaration for providing the final answer.
final_answer_declaration = {
    "name": "final_answer",
    "description": (
        "Provides the final answer to the question and ends the conversation, while listing "
        "all the used information sources."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "The final answer produced by the model."
            }
        },
        "required": ["answer"],
    },
}


# -----------------------------------------------------------------------------
# GeminiFunctionCaller Class: Implements a question-answering workflow using
# Gemini’s built-in function calling.
#
# The flow is as follows:
#   1. The user question is sent to the model along with function declarations.
#   2. The model may return a function call suggestion (for search, lookup, or final answer).
#   3. The corresponding local function is executed (e.g. querying Wikipedia).
#   4. The result is fed back into the conversation.
#   5. After one or more iterations, the model provides a final answer to the user.
# -----------------------------------------------------------------------------
class GeminiFunctionCaller:
    def __init__(self, model_name: str):
        """
        Initializes the GeminiFunctionCaller with Gemini's API, function declarations,
        and internal state to track search history.

        Args:
            model_name: The name of the Gemini model (e.g., 'gemini-2.0-flash').
        """
        # Create the Gemini client.
        self.model_name = model_name
        self.model: genai.GenerativeModel = genai.GenerativeModel(model_name)
        self.chat: genai.ChatSession = self.model.start_chat(history=[])

        # Define the tools using our function declarations.
        # These tools let the model decide when to use our external functions.
        self.tools: list[Tool] = [
            Tool(
                function_declarations=[
                    FunctionDeclaration(**wikipedia_search_declaration),
                    FunctionDeclaration(**wikipedia_lookup_declaration),
                    FunctionDeclaration(**final_answer_declaration),
                ]
            )
        ]
        print("Tools initialized:", self.tools)

        self.config: GenerationConfig

        # Variables to keep track of the last Wikipedia search for use in lookup.
        self._search_history: list[str] = []
        self._search_urls: list[str] = []

    def wikipedia_search(self, query: str) -> dict:
        """
        Performs a Wikipedia search for a given query and returns a summary with an
        information source URL. In case of ambiguity or error, returns alternatives.

        Args:
            query: The search term for Wikipedia.

        Returns:
            A dictionary with a 'result' key containing the search observation.
        """
        query = query.strip()
        try:
            summary_text = wikipedia.summary(query, sentences=4, auto_suggest=False)
            wiki_page = wikipedia.page(query, auto_suggest=False)
            wiki_url = wiki_page.url

            # Clean the summary (replace newlines with spaces).
            cleaned_summary = summary_text.replace("\n", " ")

            # Record the successful search for later use in lookup.
            self._search_history.append(query)
            self._search_urls.append(wiki_url)

            result = f"Summary: {cleaned_summary}\nInformation Source: {wiki_url}"
        except (DisambiguationError, PageError):
            similar_topics = wikipedia.search(query)
            result = f'Could not find "{query}". Similar: {similar_topics}.'
        return {"result": result}

    def wikipedia_lookup(self, phrase: str, context_length: int = 200) -> dict:
        """
        Looks up a phrase within the most recently searched Wikipedia page and returns
        a snippet of context.

        Args:
            phrase: The phrase to locate.
            context_length: Number of characters around the phrase to return.

        Returns:
            A dictionary with a 'result' key containing the context snippet.
        """
        if not self._search_history:
            return {"result": "No previous search available for lookup."}

        try:
            page = wikipedia.page(self._search_history[-1], auto_suggest=False)
            content = page.content.replace("\n", " ")
            start_index = content.find(phrase)

            snippet_start = max(0, start_index - context_length)
            snippet_end = start_index + len(phrase) + context_length
            snippet = content[snippet_start:snippet_end]

            result = f"Lookup result: {snippet}\nInformation Source: {self._search_urls[-1]}"
        except Exception as e:
            result = f"Error during lookup: {e}"
        return {"result": result}

    def final_answer(self, answer: str) -> dict:
        """
        Constructs the final answer along with the list of information sources.

        Args:
            answer: The final answer string from the model.

        Returns:
            A dictionary with a 'result' key containing the final answer and sources.
        """
        sources = ", ".join(self._search_urls) if self._search_urls else "None"
        result = f"Final Answer: {answer}\nInformation Sources: {sources}"
        return {"result": result}

    def __call__(self, user_question: str, max_iterations: int = 3, **generation_kwargs) -> str:
        """
        Initiates a conversation with the Gemini model using built-in function calling.
        The workflow iteratively sends the current conversation along with function
        declarations. If the model returns a function call, the corresponding local
        function is executed and its output is appended to the conversation.

        Args:
            user_question: The initial question from the user.
            max_iterations: The maximum iterations for the function-calling loop.
            **generation_kwargs: Additional keyword arguments for generation (e.g., temperature).

        Returns:
            The final response text from the model.
        """
        # Create the initial conversation as a list of Content objects.
        messages = [
            {"role": "user", "parts": [{"text": user_question}]}
        ]

        # Reset self.config by building a new GenerateContentConfig that includes both the tools and any provided
        # generation settings.
        self.config = GenerationConfig(**generation_kwargs)

        for i in range(max_iterations):
            # Send the conversation to Gemini.
            response = self.model.generate_content(
                generation_config=self.config,
                tools=self.tools,
                contents=messages
            )

            # Inspect the candidate's first content part for a function call.
            candidate = response.candidates[0]
            part = candidate.content.parts[0]
            if part.function_call:
                # Extract function call details.
                function_call = part.function_call
                func_name = function_call.name
                args = function_call.args  # This is already a dict.

                # Execute the appropriate function based on the call.
                if func_name == "wikipedia_search":
                    result = self.wikipedia_search(**args)
                elif func_name == "wikipedia_lookup":
                    context_length = args.get("context_length", 200)
                    result = self.wikipedia_lookup(phrase=args["phrase"], context_length=context_length)
                elif func_name == "final_answer":
                    result = self.final_answer(**args)
                    # Append the function call and its result to messages, then break.
                    messages.append(types.Content(role="model", parts=[types.Part(function_call=function_call)]))
                    messages.append(types.Content(role="user",
                                                  parts=[types.Part.from_function_response(name=function_call.name,
                                                                                           response=result)]))
                    break
                else:
                    result = {"result": f"Unknown function call: {func_name}"}

                # Append the model's function call message.
                messages.append(types.Content(role="model", parts=[types.Part(function_call=function_call)]))
                # Append the function execution result as a user message.
                messages.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_function_response(name=function_call.name, response=result)]
                    )
                )
            else:
                # No function call was produced. Return the model's plain text response.
                print("Final response:", response.text)
                return response.text

        # After one or more iterations (i.e., after a function response has been fed back),
        # ask the model again to produce the final answer.
        final_response = self.model.generate_content(
            generation_config=self.config,
            tools=self.tools,
            contents=messages,
        )
        print("Final response:", final_response.text)
        return final_response.text


# -----------------------------------------------------------------------------
# Example usage of the GeminiFunctionCaller.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Define the question you want to ask.
    question = (
        "What are the total of ages of the main trio from the new Percy Jackson and the Olympians TV series "
        "in real life?"
    )

    # Instantiate the function-calling handler with the desired Gemini model.
    gemini_caller = GeminiFunctionCaller(model_name="gemini-2.0-flash")

    # Start the conversation with the question.
    final_output = gemini_caller(question, temperature=0.2)
    print("\n--- Final Output ---")
    print(final_output)
