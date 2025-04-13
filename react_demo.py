# Example of ReAct with Gemini 2.0 in Colab
# https://colab.research.google.com/drive/1lo7czGYVGgq1rfF69VBX2WhDPytW4906#scrollTo=cGsNrV_vTSGw
# https://github.com/google-gemini/cookbook/blob/main/examples/Search_Wikipedia_using_ReAct.ipynb
from __future__ import annotations

import re
import os
from generator_model import get_secret
import wikipedia
from wikipedia.exceptions import DisambiguationError, PageError
# noinspection PyPackageRequirements
import google.generativeai as genai

secret = get_secret(r'D:\Documents\Secrets\gemini_secret.txt')
genai.configure(api_key=secret)

model_instructions = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, Observation is understanding relevant information from an Action's output and Action can be of three types:
(1) <search>, which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search and you can try to search the information from those topics.
(2) <lookup>, which returns the next sentence containing keyword in the current context. This only does exact matches, so keep your searches short.
(3) <answer>, which returns the answer and finishes the task.
"""  # noqa: E501

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

Important: When generating an Action, output exactly one line in the format <function>parameter (with no additional 
text) and then stop.

For example:
Action 5
<answer>1,800 to 7,000 ft


Question
{question}"""  # noqa: E501

ReAct_prompt = model_instructions + examples


class ReAct:
    def __init__(self, model: str, react_prompt: str | os.PathLike):
        """Prepares Gemini to follow a `Few-shot ReAct prompt` by imitating
        `function calling` technique to generate both reasoning traces and
        task-specific actions in an interleaved manner.

        Args:f
            model: name to the model.
            react_prompt: ReAct prompt OR path to the ReAct prompt.
        """
        self.model = genai.GenerativeModel(model)
        self.chat = self.model.start_chat(history=[])
        self.should_continue_prompting = True
        self._search_history: list[str] = []
        self._search_urls: list[str] = []

        try:
            # try to read the file
            with open(react_prompt, 'r') as f:
                self._prompt = f.read()
        except FileNotFoundError:
            # assume that the parameter represents prompt itself rather than path to the prompt file.
            self._prompt = react_prompt

    @property
    def prompt(self):
        return self._prompt

    @classmethod
    def add_method(cls, func):
        setattr(cls, func.__name__, func)

    @staticmethod
    def clean(text: str):
        """Helper function for responses."""
        text = text.replace("\n", " ")
        return text

    def search(self, query: str):
        """Performs search on `query` via Wikipedia api and returns its summary.

        Args:
            self: Instance of the ReAct class.
            query: Search parameter to query the Wikipedia API with.

        Returns:
            observation: Summary of Wikipedia search for `query` if found else
            similar search results.
        """
        observation: str
        query = query.strip()
        try:
            # try to get the summary for requested `query` from the Wikipedia
            observation = wikipedia.summary(query, sentences=4, auto_suggest=False)
            wiki_url = wikipedia.page(query, auto_suggest=False).url
            observation = self.clean(observation)

            # if successful, return the first 2-3 sentences from the summary as model's context
            observation = self.model.generate_content(f'Return the first 2 or 3 \
            sentences from the following text: {observation}').text

            # keep track of the model's search history
            self._search_history.append(query)
            self._search_urls.append(wiki_url)
            print(f"Information Source: {wiki_url}")

        # if the page is ambiguous/does not exist, return similar search phrases for model's context
        except (DisambiguationError, PageError):
            observation = f'Could not find ["{query}"].'
            # get a list of similar search topics
            search_results = wikipedia.search(query)
            observation += f' Similar: {search_results}. You should search for one of those instead.'

        return observation

    def lookup(self, phrase: str, context_length=200):
        """Searches for the `phrase` in the latest Wikipedia search page
        and returns number of sentences which is controlled by the
        `context_length` parameter.

        Args:
            self: Instance of the ReAct class.
            phrase: Lookup phrase to search for within a page. Generally
            attributes to some specification of any topic.

            context_length: Number of words to consider
            while looking for the answer.

        Returns:
            result: Context related to the `phrase` within the page.
        """
        # get the last searched Wikipedia page and find `phrase` in it.
        page = wikipedia.page(self._search_history[-1], auto_suggest=False)
        page = page.content
        page = self.clean(page)
        start_index = page.find(phrase)

        # extract sentences considering the context length defined
        result = page[max(0, start_index - context_length):start_index+len(phrase)+context_length]
        print(f"Information Source: {self._search_urls[-1]}")
        return result

    def answer(self, _):
        """Finishes the conversation on encountering  token by
        setting the `self.should_continue_prompting` flag to `False`.
        """
        self.should_continue_prompting = False
        print(f"Information Sources: {self._search_urls}")

    def __call__(self, user_question, max_calls: int = 10, **generation_kwargs):
        """Starts multi-turn conversation with the chat models with function calling

        Args:
            max_calls: max calls made to the model to get the final answer.

            generation_kwargs: Same as genai.GenerativeModel.GenerationConfig
                    candidate_count: (int | None) = None,
                    stop_sequences: (Iterable[str] | None) = None,
                    max_output_tokens: (int | None) = None,
                    temperature: (float | None) = None,
                    top_p: (float | None) = None,
                    top_k: (int | None) = None

        Raises:
            AssertionError: if max_calls is not between 1 and 8
        """

        # hyperparameter fine-tuned according to the paper
        assert 0 < max_calls <= 10, "max_calls must be between 1 and 8"

        if len(self.chat.history) == 0:
            model_prompt = self.prompt.format(question=user_question)
        else:
            model_prompt = user_question

        # stop_sequences for the model to imitate function calling
        callable_entities = ['<stop>']

        generation_kwargs.update({'stop_sequences': callable_entities})

        self.should_continue_prompting = True
        for idx in range(max_calls):

            self.response = self.chat.send_message(content=[model_prompt],
                                                   generation_config=generation_kwargs, stream=False)

            for chunk in self.response:
                print(chunk.text, end=' ')

            response_cmd = self.chat.history[-1].parts[-1].text

            try:
                # regex to extract
                cmd = re.findall(r'<(.*)>', response_cmd)[-1]
                print(f'{cmd}>')
                # regex to extract param
                query = response_cmd.split(f'<{cmd}>')[-1].strip()
                # call to appropriate function
                observation = self.__getattribute__(cmd)(query)

                if not self.should_continue_prompting:
                    break

                stream_message = f"\nObservation {idx + 1}\n{observation}"
                print(stream_message)
                # send function's output as user's response
                model_prompt = f"<{cmd}>{query}{cmd}>'s Output: {stream_message}"

            except (IndexError, AttributeError):
                model_prompt = ("You failed to make a function call by following the function calling format. "
                                "Please try again. When generating an Action, output exactly one line "
                                "in the format <function>parameter (with no additional text) and then stop. "
                                "For example:\n"
                                "Action 5\n"
                                "<answer>1,800 to 7,000 ft\n")


question: str = ("What are the total of ages of the main trio from the new Percy Jackson and the Olympians TV series "
                 "in real life?")
# question = "How many companions did Samuel Meladon have?"
gemini_ReAct_chat = ReAct(model='gemini-2.0-flash', react_prompt=ReAct_prompt)
# Note: try different combinations of generational_config parameters for variational results
gemini_ReAct_chat(
    question,
    temperature=0.2)
