# https://ai.google.dev/gemini-api/docs/function-calling?example=meeting
# https://medium.com/google-cloud/building-react-agents-from-scratch-a-hands-on-guide-using-gemini-ffe4621d90ae
# https://www.philschmid.de/gemini-function-calling
# New approach
# https://atamel.dev/posts/2025/04-08_simplified_function_calling_gemini/
from __future__ import annotations
import wikipedia
from wikipedia.exceptions import DisambiguationError, PageError
# noinspection PyPackageRequirements
import google.generativeai as genai
# noinspection PyPackageRequirements
from google.genai import types
# noinspection PyPackageRequirements
from google.generativeai.types import Tool, FunctionDeclaration
# noinspection PyPackageRequirements
from google.generativeai.types.generation_types import GenerationConfig

# Import the secret retrieval function from a module.
from generator_model import get_secret

# -----------------------------------------------------------------------------
# Configure API Key for Gemini 2.0 using a secret file.
# -----------------------------------------------------------------------------
secret = get_secret(r'D:\Documents\Secrets\gemini_secret.txt')
genai.configure(api_key=secret)

# -----------------------------------------------------------------------------
# Define function declarations for the function calling flow.
# -----------------------------------------------------------------------------

# Function declaration for performing a Wikipedia search.
search_declaration = {
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
lookup_declaration = {
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
answer_declaration = {
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
    def __init__(self, model: str):
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

    def search(self, query: str) -> dict:
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
            summary_text = wikipedia.summary(query, sentences=4, auto_suggest=False)
            wiki_page = wikipedia.page(query, auto_suggest=False)
            wiki_url = wiki_page.url

            # Clean the summary text.
            cleaned_summary = self.clean(summary_text)

            # Generate a shorter version containing the first 2-3 sentences from the summary.
            observation = self.generative_model.generate_content(
                f'Return the first 2 or 3 sentences from the following text: {cleaned_summary}'
            ).text

            # Record the search details for later lookup.
            self._search_history.append(query)
            self._search_urls.append(wiki_url)
            print(f"Information Source: {wiki_url}")

        except (DisambiguationError, PageError):
            # Handle ambiguous or non-existent pages.
            observation = f'Could not find ["{query}"].'
            similar_topics = wikipedia.search(query)
            observation += f' Similar: {similar_topics}. You should search for one of those instead.'

        return {"result": observation}

    def lookup(self, phrase: str, context_length: int = 200) -> dict:
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
            content = self.clean(page.content)
            start_index = content.find(phrase)

            # Compute the context snippet boundaries.
            snippet_start = max(0, start_index - context_length)
            snippet_end = start_index + len(phrase) + context_length
            result = content[snippet_start:snippet_end]

            print(f"Result: {result}")
            print(f"Information Source: {self._search_urls[-1]}")
            return {"result": result}
        except Exception as e:
            return {"result": f"Error during lookup: {e}"}

    def answer(self, final_answer: str) -> dict:
        """
        Marks the end of the conversation and returns the final answer along
        with sources.

        Args:
            final_answer: The final answer text.

        Returns:
            A dictionary with the final answer and sources.
        """
        self.should_continue_prompting = False
        sources = ", ".join(self._search_urls) if self._search_urls else "None"
        result = f"Final Answer: {final_answer}\nInformation Sources: {sources}"
        print(f"Information Sources: {self._search_urls}")
        return {"result": result}

    def __call__(self, user_question: str, max_iterations: int = 10, **generation_kwargs):
        """
        Initiates a multi-turn conversation with the generative model, using
        function calls to perform a series of actions until the answer is found
        or the maximum number of iterations is reached.

        Args:
            user_question: The question or prompt for which an answer is to be generated.
            max_iterations: Maximum number of interactions to attempt for resolving the query.
            **generation_kwargs: Additional keyword arguments for model generation.

        Returns:
            The final response from the model.
        """
        # Validate the maximum allowed number of iterations.
        assert 0 < max_iterations <= 10, "max_iterations must be between 1 and 10"

        # Create a prompt for the model that encourages ReAct-style reasoning
        react_prompt = (
            "Solve this question step by step using the search, lookup, and answer functions. "
            "First, think about how to approach the problem. "
            "Use search to find relevant information from Wikipedia. "
            "Use lookup to find specific details within the article. "
            "When you have the final answer, use the answer function.\n\n"
            f"Question: {user_question}"
        )

        # Create the initial conversation
        messages = [
            {"role": "user", "parts": [{"text": react_prompt}]}
        ]

        # Set up the config with any provided generation parameters
        config = GenerationConfig(**generation_kwargs)

        # Main conversation loop
        self.should_continue_prompting = True

        for i in range(max_iterations):
            if not self.should_continue_prompting:
                break

            # Send the current state to the model
            response = self.generative_model.generate_content(
                generation_config=config,
                tools=self.tools,
                contents=messages
            )

            # Check for function calls in the response
            candidate = response.candidates[0]
            content = candidate.content

            # Initialize variables to track whether we've found text or function calls
            found_text = False
            found_function_call = False

            # Process all parts in the content
            for part in content.parts:
                # Handle text part if present
                if hasattr(part, "text") and part.text:
                    found_text = True
                    print(f"\nThought {i + 1}:")
                    print(part.text)

                # Handle function call if present
                if hasattr(part, "function_call") and part.function_call:
                    found_function_call = True
                    function_call = part.function_call
                    func_name = function_call.name
                    args = function_call.args

                    print(f"\nAction {i + 1}:")
                    print(f"<{func_name}>{list(args.values())[0] if args else ''}")

                    # Execute the appropriate function
                    if func_name == "search":
                        result = self.search(**args)
                    elif func_name == "lookup":
                        context_length = args.get("context_length", 200)
                        result = self.lookup(phrase=args["phrase"], context_length=context_length)
                    elif func_name == "answer":
                        result = self.answer(**args)
                    else:
                        result = {"result": f"Unknown function: {func_name}"}

                    # Print the observation
                    print(f"\nObservation {i + 1}:")
                    print(result["result"])

                    # Append the model's function call and result to messages
                    messages.append(types.Content(role="model", parts=[types.Part(function_call=function_call)]))
                    messages.append(
                        types.Content(
                            role="user",
                            parts=[types.Part.from_function_response(name=function_call.name, response=result)]
                        )
                    )

                    # If answer was called, we're done
                    if func_name == "answer":
                        break

            # If no function call was found but we have text, treat it as a final response
            if not found_function_call and found_text:
                print("\nFinal response:")
                print(response.text)
                return response.text

            # If answer function was called, break the loop
            if not self.should_continue_prompting:
                break

        # After reaching the max iterations or completing the loop,
        # return the final model response if needed
        if self.should_continue_prompting:
            final_response = self.generative_model.generate_content(
                generation_config=config,
                tools=self.tools,
                contents=messages,
            )
            print("\nFinal response:")
            print(final_response.text)
            return final_response.text

        return "Conversation completed with final answer."


# -----------------------------------------------------------------------------
# Example usage of the ReActFunctionCaller class.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Define the question you want to ask.
    question = (
        "What are the total of ages of the main trio from the new Percy Jackson and the Olympians TV series "
        "in real life?"
    )

    # Uncomment alternate questions as needed.
    # question = "How many companions did Samuel Meladon have?"
    # question = "What is the most famous case of reincarnation in the world?"

    # Instantiate the ReActFunctionCaller session using the defined model.
    gemini_react = ReActFunctionCaller(model='gemini-2.0-flash')

    # Start the conversation using the provided question and generation parameters.
    # Feel free to adjust generation_kwargs such as temperature for varied results.
    gemini_react(question, temperature=0.2)
