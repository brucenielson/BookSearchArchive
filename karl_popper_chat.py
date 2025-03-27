import time
import gradio as gr
# noinspection PyPackageRequirements
# from google.genai import Client
# noinspection PyPackageRequirements
from google.genai.types import GenerateContentConfig
import generator_model as gen
# noinspection PyPackageRequirements
import google.generativeai as genai
# Import your DocRetrievalPipeline and SearchMode (adjust import paths as needed)
from doc_retrieval_pipeline import DocRetrievalPipeline, SearchMode
# noinspection PyPackageRequirements
from haystack import Document
from typing import Optional, List, Dict, Any


class KarlPopperChat:
    def __init__(self):
        # Initialize Gemini Chat with a system instruction to act like philosopher Karl Popper.
        google_secret: str = gen.get_secret(r'D:\Documents\Secrets\gemini_secret.txt')
        genai.configure(api_key=google_secret)
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            system_instruction="You are philosopher Karl Popper. Answer questions with philosophical insights, "
                               "and use the provided quotes along with their metadata as reference.")
        self.model = model

        # Initialize the document retrieval pipeline with top-5 quote retrieval.
        password: str = gen.get_secret(r'D:\Documents\Secrets\postgres_password.txt')
        user_name: str = "postgres"
        db_name: str = "postgres"
        self.doc_pipeline = DocRetrievalPipeline(
            table_name="popper_archive",
            db_user_name=user_name,
            db_password=password,
            postgres_host='localhost',
            postgres_port=5432,
            db_name=db_name,
            verbose=False,
            llm_top_k=5,
            retriever_top_k_docs=100,
            include_outputs_from=None,
            search_mode=SearchMode.HYBRID,
            use_reranker=True,
            embedder_model_name="BAAI/llm-embedder"
        )
        self.doc_pipeline.draw_pipeline()

    @staticmethod
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

    # Taken from https://medium.com/latinxinai/simple-chatbot-gradio-google-gemini-api-4ce02fbaf09f
    @staticmethod
    def transform_history(history) -> List[Dict[str, Any]]:
        new_history = []
        for chat_response in history:
            new_history.append({"parts": [{"text": chat_response[0]}], "role": "user"})
            new_history.append({"parts": [{"text": chat_response[1]}], "role": "model"})
        return new_history

    def ask_llm_question(self, prompt: str, chat_history: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Ask a question to the LLM (Large Language Model) and get a response.

        Args:
            prompt (str): The question or prompt to send to the LLM.
            chat_history (Optional[str]): The chat history to include in the session. Defaults to None.
            Format is assumed to be correct for Gemini (for now).
            If None, a new chat session is started without history.

        Returns:
            str: The response from the LLM.
        """
        if chat_history is None:
            chat_history = []
        # Start a new chat session with no history for this check.
        chat_session = self.model.start_chat(history=chat_history)
        chat_response = chat_session.send_message(prompt)
        # Extract numbers from Gemini's response.
        return chat_response.text.strip()

    def ask_llm_for_quote_relevance(self, message: str, docs: List[Document]) -> str:
        """
        Given a question and a list of retrieved documents, ask the LLM to determine which quotes are relevant.
        This function formats the question and documents into a prompt for the LLM, which will then return a list of
        relevant quote numbers based on order of Documents in the docs list. e.g. "1,3,5".
        You can then use this list to filter the documents to which the LLM found most relevant to the question.

        Args:
            message (str): The question or prompt to evaluate.
            docs (List[Document]): The list of documents containing quotes to be reviewed.

        Returns:
            str: A comma-separated list of numbers indicating the relevant quotes, or an empty string if none are
            relevant.
        """
        prompt = (
            f"Given the question: '{message}', review the following numbered quotes and "
            "return a comma-separated list of the numbers for the quotes that you believe will help answer the "
            "question. If there are no quotes relevant to the question, return an empty string. "
            "Answer with only the numbers or an empty string, for example: '1,3,5' or ''.\n\n"
        )
        for i, doc in enumerate(docs, start=1):
            prompt += f"{i}. {doc.content}\n\n"

        return self.ask_llm_question(prompt)

    def ask_llm_for_improved_query(self, message: str, chat_history: List[Dict[str, Any]]) -> str:
        prompt = (
            f"Given the query: '{message}' and the current chat history, the database of relevant quotes found none "
            f"that were a strong match. This might be due to poor wording on the user's part. "
            f"Reviewing the chat history, determine if you can provide a better wording for the query that might "
            f"yield better results. If you can improve the query, return the improved "
            f"query. If you cannot improve the question, return an empty string and we'll continue with the user's "
            f"original query. There is no need to explain your thinking if you want to return an empty string.\n\n"
            f"You must either return a single sentence or phrase that is the new query or an empty string. "
            f"For example: 'Alternative views of induction' or ''"
            f"\n\n"
        )

        return self.ask_llm_question(prompt, chat_history)

    @staticmethod
    def get_max_score(docs: Optional[List[Document]]) -> float:
        """
        Get the maximum score from a list of documents.

        Args:
            docs (List[Document]): The list of documents to evaluate.

        Returns:
            float: The maximum score found in the documents.
        """
        # Find the largest score
        max_score: float = 0.0
        if docs:
            max_score = max(doc.score for doc in docs if hasattr(doc, 'score'))
        return max_score

    def respond(self, message: Optional[str], chat_history: List[Optional[List[str]]]):
        # --- Step 1: Retrieve the top-5 quotes with metadata ---
        if message.strip() == "" or message is None:
            # This is a kludge to deal with the fact that Gradio sometimes get a race condition, and we lose the message
            # To correct, try to get the last message from chat history
            if chat_history and len(chat_history) > 0 and chat_history[-1][1] is None:
                # If the last message has no response, then grab the message portion and remove it.
                # It will get added back again below.
                # There has got to be a better way to do this, but this will work for now
                message = chat_history[-1][0]
                # Remove last message from chat history
                chat_history = chat_history[:-1]

        # Put the chat_history into the correct format for Gemini
        gemini_chat_history: List[Dict[str, Any]] = self.transform_history(chat_history)

        retrieved_docs: List[Document]
        all_docs: List[Document]
        retrieved_docs, all_docs = self.doc_pipeline.generate_response(message)

        # Find the largest score
        max_score: float = self.get_max_score(retrieved_docs)

        if max_score is not None and max_score < 0.50:
            # If we don't have any good quotes, ask the LLM if it wants to do its own search
            improved_query: str = self.ask_llm_for_improved_query(message, gemini_chat_history)
            # The LLM is sometimes stupid and takes my example too literally and returns "''" instead of "" for an
            # empty string. So we need to check for that and convert it to an empty string.
            # Unfortunately, dropping that instruction tends to cause it to think out loud before returning an empty
            # string at the end. Which sort of defeats the purpose.
            if improved_query == "''":
                improved_query = ""
            new_retrieved_docs: List[Document]
            temp_all_docs: List[Document]
            if improved_query != "":
                new_retrieved_docs, temp_all_docs = self.doc_pipeline.generate_response(improved_query)
                new_max_score: float = self.get_max_score(new_retrieved_docs)
                if new_max_score > max_score:
                    # If the new max score is better than the old one, use the new docs
                    retrieved_docs = new_retrieved_docs
                    all_docs = temp_all_docs
                    max_score = new_max_score

        if max_score is not None and max_score < 0.30:
            # If there are no quotes with a score at least 0.30,
            # then we ask Gemini in one go which quotes are relevant.
            response_text = self.ask_llm_for_quote_relevance(message, retrieved_docs)
            # Split by commas, remove any extra spaces, and convert to integers.
            try:
                relevant_numbers = [int(num.strip()) for num in response_text.split(',') if num.strip().isdigit()]
            except Exception as parse_e:
                print(f"Error parsing Gemini response: {parse_e}")
                time.sleep(1)
                relevant_numbers = []

            # Filter docs based on the numbered positions.
            ranked_docs = [doc for idx, doc in enumerate(retrieved_docs, start=1) if idx in relevant_numbers]
        else:
            # Drop any quotes with a score less than 0.20 if we have at least 3 quotes above 20
            # Otherwise drop any quotes with a score less than 0.10
            # Count how many quotes have a score >= 0.20.
            threshold: float = 0.20
            num_high = len([doc for doc in retrieved_docs if hasattr(doc, 'score') and doc.score >= threshold])
            # If we have at least 3 such quotes, drop any with a score less than 0.20.
            # Otherwise, drop quotes with a score less than 0.10.
            threshold = 0.20 if num_high >= 3 else 0.10
            ranked_docs = [doc for doc in retrieved_docs if hasattr(doc, 'score') and doc.score >= threshold]

        # Format each retrieved document (quote + metadata).
        formatted_docs = [self.format_document(doc) for doc in ranked_docs]
        retrieved_quotes = "\n\n".join(formatted_docs)
        formatted_docs = [self.format_document(doc, include_raw_info=True) for doc in all_docs]
        all_quotes = "\n\n".join(formatted_docs)

        modified_query: str = ""
        if not retrieved_quotes or retrieved_quotes.strip() == "":
            modified_query = message
        elif max_score > 0.50:
            modified_query = (
                f"Use the following quotes with their metadata as reference in your answer:\n\n{retrieved_quotes}\n\n"
                f"Reference the quotes and their metadata in your answer where possible. "
                f"Now, answer the following question: {message}"
            )
        else:
            modified_query = (
                f"The following quotes are available. You may use them as reference to answer my question"
                f"if you find them relevant:\n\n{retrieved_quotes}\n\n"
                f"Reference the quotes and their metadata in your answer if used. But don't "
                f"feel obligated to use the quotes if they are not relevant. "
                f"Now, answer the following question: {message}"
            )
        # We start a new chat session each time so that we can control the chat history and remove all the rag docs
        # We just want questions and answers in the chat history
        chat_session = self.model.start_chat(history=gemini_chat_history)
        # Send the modified query to Gemini.
        chat_response = chat_session.send_message(modified_query, stream=True)
        answer_text = ""
        # # --- Step 3: Stream the answer character-by-character ---
        for chunk in chat_response:
            if hasattr(chunk, 'text'):
                answer_text += chunk.text
                yield chat_history + [(message, answer_text)], retrieved_quotes, all_quotes


def build_interface():
    karl_chat = KarlPopperChat()
    css: str = """
    #retrieved_quotes, #raw_quotes {
        height: calc(100vh - 100px);
        overflow-y: auto;
        white-space: pre-wrap;
    }
    """

    with gr.Blocks(css=css) as chat_interface:
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("# Karl Popper Chatbot")
                gr.Markdown(
                    "Chat with AI Karl Popper. He'll respond in the chat box on the left and utilize and cite "
                    "sources from the box on the right.")
                chatbot = gr.Chatbot(label="Chat")
                msg = gr.Textbox(placeholder="Ask your question...", label="Your Message")
                clear = gr.Button("Clear Chat")

            with gr.Column(scale=1):
                with gr.Tab("Retrieved Quotes"):
                    retrieved_quotes_box = gr.Markdown(label="Retrieved Quotes & Metadata", value="",
                                                       elem_id="retrieved_quotes")
                with gr.Tab("Raw Quotes"):
                    raw_quotes_box = gr.Markdown(label="Raw Quotes & Metadata", value="", elem_id="raw_quotes")

        def user_message(message, chat_history):
            # print(f"user_message: User submitted message: '{message}'")
            # Append the user's message to the chat history
            updated_history = chat_history + [(message, None)]
            return "", updated_history

        def process_message(message, chat_history):
            # print(f"process_message: User submitted message: '{message}'")
            for updated_history, ranked_docs, all_docs in karl_chat.respond(message, chat_history):
                yield updated_history, ranked_docs, all_docs

        msg.submit(user_message, [msg, chatbot], [msg, chatbot], queue=True)
        msg.submit(process_message, [msg, chatbot], [chatbot, retrieved_quotes_box, raw_quotes_box], queue=True)
        clear.click(lambda: ([], ""), None, [chatbot, retrieved_quotes_box, raw_quotes_box], queue=False)

    return chat_interface


if __name__ == "__main__":
    popper_chat = build_interface()
    popper_chat.launch(debug=True)
