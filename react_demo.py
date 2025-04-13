# Example of ReAct with Gemini 2.0 in Colab
# https://colab.research.google.com/drive/1lo7czGYVGgq1rfF69VBX2WhDPytW4906#scrollTo=cGsNrV_vTSGw
# https://github.com/google-gemini/cookbook/blob/main/examples/Search_Wikipedia_using_ReAct.ipynb
from __future__ import annotations
import re
import os
import wikipedia
from wikipedia.exceptions import DisambiguationError, PageError
# noinspection PyPackageRequirements
import google.generativeai as genai

# Import the secret retrieval function from a module.
from generator_model import get_secret

# -----------------------------------------------------------------------------
# Configure API Key for Gemini 2.0 using a secret file.
# -----------------------------------------------------------------------------
secret = get_secret(r'D:\Documents\Secrets\gemini_secret.txt')
genai.configure(api_key=secret)

# -----------------------------------------------------------------------------
# Define the model instructions and examples for the ReAct framework.
# This prompt guides Gemini to follow the ReAct pattern by interleaving
# thought, actions, and observations.
# -----------------------------------------------------------------------------
model_instructions = (
    "Solve a question answering task with interleaving Thought, Action, "
    "Observation steps. Thought can reason about the current situation, Observation "
    "is understanding relevant information from an Action's output and Action can be "
    "of three types:\n"
    "(1) <search>, which searches the exact entity on Wikipedia and returns the first "
    "paragraph if it exists. If not, it will return some similar entities to search and "
    "you can try to search the information from those topics.\n"
    "(2) <lookup>, which returns the next sentence containing keyword in the current "
    "context. This only does exact matches, so keep your searches short.\n"
    "(3) <answer>, which returns the answer and finishes the task.\n"
)

# noinspection SpellCheckingInspection
examples = """
Here is an example,


Example of Round 1:

Question
What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?

Thought 1
I need to search Colorado orogeny, find the area that the eastern sector of the Colorado orogeny extends into, then find the elevation range of the area.

Action 1
<search>Colorado orogeny


Example of Round 2:

Observation 1
The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas.

Thought 2
It does not mention the eastern sector. So I need to look up eastern sector.

Action 2
<lookup>eastern sector


Example of Round 3:

Observation 2
The eastern sector extends into the High Plains and is called the Central Plains orogeny.

Thought 3
The eastern sector of Colorado orogeny extends into the High Plains. So I need to search High Plains and find its elevation range.

Action 3
<lookup>High Plains


Example of Round 4:

Observation 3
High Plains refers to one of two distinct land regions

Thought 4
I need to instead search High Plains (United States).

Action 4
<search>High Plains (United States)


Example of Round 5:

Observation 4
The High Plains are a subregion of the Great Plains. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130m).

Thought 5
High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.

Action 5
<answer>1,800 to 7,000 ft

End of examples.

Important: When generating an Action, output exactly one line in the format <function>parameter (with no additional text) and then stop.

For example:
Action 5
<answer>1,800 to 7,000 ft


Question
{question}
"""  # noqa: E501

ReAct_prompt = model_instructions + examples


# -----------------------------------------------------------------------------
# ReAct Class: Implements a multi-turn conversation using the ReAct framework.
# -----------------------------------------------------------------------------
class ReAct:
    def __init__(self, model: str, react_prompt: str | os.PathLike):
        """
        Initializes the ReAct instance by setting up the Gemini model,
        loading the ReAct prompt (either directly or from a file), and
        initializing relevant search history.

        Args:
            model: Name of the generative model (e.g., 'gemini-2.0-flash').
            react_prompt: Either the actual ReAct prompt as a string or a path
                          to a file containing the prompt.
        """
        self.model = genai.GenerativeModel(model)
        self.chat = self.model.start_chat(history=[])
        self.should_continue_prompting = True
        self._search_history: list[str] = []
        self._search_urls: list[str] = []

        try:
            # Attempt to read the prompt from a file path.
            with open(react_prompt, 'r') as f:
                self._prompt = f.read()
        except FileNotFoundError:
            # If file is not found, assume the given parameter is the prompt itself.
            self._prompt = react_prompt

    @property
    def prompt(self) -> str:
        """
        Property getter for the ReAct prompt.

        Returns:
            The current ReAct prompt.
        """
        return self._prompt

    @classmethod
    def add_method(cls, func):
        """
        Dynamically add a method to the ReAct class.

        Args:
            func: The function to add as a method of the class.
        """
        setattr(cls, func.__name__, func)

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

    def search(self, query: str) -> str:
        """
        Searches for a given query using the Wikipedia API and returns a
        summary or alternative suggestions if an error occurs.

        Args:
            query: The search term to query on Wikipedia.

        Returns:
            A summary of the Wikipedia page corresponding to the query or a
            message with similar suggestions if the page is ambiguous or missing.
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
            observation = self.model.generate_content(
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

        return observation

    def lookup(self, phrase: str, context_length: int = 200) -> str:
        """
        Looks up a given phrase within the content of the most recently
        searched Wikipedia page and returns a context snippet around it.

        Args:
            phrase: The keyword or phrase to locate within the page content.
            context_length: The number of characters before and after the phrase
                            to include in the resulting context snippet.

        Returns:
            A context snippet from the Wikipedia page surrounding the phrase.
        """
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
        return result

    def answer(self, _ignored: str) -> None:
        """
        Marks the end of the conversation by setting a flag to stop further prompts.
        Also prints all the gathered information sources (URLs) that were used during the session.

        Args:
            _ignored: This parameter is ignored as the answer function does not require input.
        """
        self.should_continue_prompting = False
        print(f"Information Sources: {self._search_urls}")

    def __call__(self, user_question: str, max_calls: int = 10, **generation_kwargs):
        """
        Initiates a multi-turn conversation with the generative model, using
        function calls to perform a series of actions until the answer is found
        or the maximum number of calls is reached.

        Args:
            user_question: The question or prompt for which an answer is to be generated.
            max_calls: Maximum number of interactions to attempt for resolving the query.
            **generation_kwargs: Additional keyword arguments for model generation such as:
                                 temperature, max_output_tokens, top_p, etc.

        Raises:
            AssertionError: If max_calls is set to a value not within the allowed range.
        """
        # Validate the maximum allowed number of calls.
        assert 0 < max_calls <= 10, "max_calls must be between 1 and 10"

        # Determine the initial prompt to send to the model.
        if len(self.chat.history) == 0:
            model_prompt = self.prompt.format(question=user_question)
        else:
            model_prompt = user_question

        # Define stop sequences to mimic function calling behavior.
        generation_kwargs.update({'stop_sequences': ['<stop>']})
        self.should_continue_prompting = True

        for idx in range(max_calls):
            # Send the current prompt to the model.
            self.response = self.chat.send_message(
                content=[model_prompt],
                generation_config=generation_kwargs,
                stream=False
            )

            # Print the full response for debugging/inspection.
            for chunk in self.response:
                print(chunk.text, end=' ')

            # Retrieve the model's latest message which should contain a function call.
            response_cmd = self.chat.history[-1].parts[-1].text

            try:
                # Extract the function name (e.g., search, lookup, answer) from the response.
                cmd = re.search(r'<(\w+)>', response_cmd).group(1)
                # Extract the parameter after the function tag.
                query = response_cmd.split(f'<{cmd}>')[-1].strip()

                # Dynamically call the function based on the extracted command.
                observation = self.__getattribute__(cmd)(query)

                # If the answer function was triggered, break the loop.
                if not self.should_continue_prompting:
                    break

                # Prepare the observation to send back as context for the next round.
                stream_message = f"\nObservation {idx + 1}\n{observation}"
                print(stream_message)
                # Update the prompt with both the action and its observation.
                model_prompt = f"<{cmd}>{query}{cmd}>'s Output: {stream_message}"

            except (IndexError, AttributeError):
                # If the expected function call format is not adhered to, send an error prompt.
                model_prompt = (
                    "You failed to make a function call by following the function calling format. "
                    "Please try again. When generating an Action, output exactly one line "
                    "in the format <function>parameter (with no additional text) and then stop. "
                    "For example:\n"
                    "Action 5\n"
                    "<answer>1,800 to 7,000 ft\n"
                )


# -----------------------------------------------------------------------------
# Example usage of the ReAct class.
# You can change the question and generational configuration parameters below.
# -----------------------------------------------------------------------------
question: str = (
    "What are the total of ages of the main trio from the new Percy Jackson and the Olympians TV series "
    "in real life?"
)

# Uncomment alternate questions as needed.
# question = "How many companions did Samuel Meladon have?"
question = "What is the most famous case of reincarnation in the world?"

# Instantiate the ReAct session using the defined model and prompt.
gemini_ReAct_chat = ReAct(model='gemini-2.0-flash', react_prompt=ReAct_prompt)

# Start the conversation using the provided question and generation parameters.
# Feel free to adjust generation_kwargs such as temperature for varied results.
gemini_ReAct_chat(question, temperature=0.2)
