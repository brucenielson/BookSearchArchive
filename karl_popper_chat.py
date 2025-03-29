import time
import os
import gradio as gr
# noinspection PyPackageRequirements
# from google.genai import Client
# noinspection PyPackageRequirements
# from google.genai.types import GenerateContentConfig
import generator_model as gen
# noinspection PyPackageRequirements
import google.generativeai as genai
# Import your DocRetrievalPipeline and SearchMode (adjust import paths as needed)
from doc_retrieval_pipeline import DocRetrievalPipeline, SearchMode
from document_processor import DocumentProcessor
# noinspection PyPackageRequirements
from haystack import Document
from typing import Optional, List, Dict, Any, Iterator


class RagChat:
    def __init__(self, system_instruction: Optional[str] = None):
        # Initialize Gemini Chat with a system instruction to act like philosopher Karl Popper.
        google_secret: str = gen.get_secret(r'D:\Documents\Secrets\gemini_secret.txt')
        genai.configure(api_key=google_secret)
        self._initialize_model(system_instruction=system_instruction)

        # Initialize the document retrieval pipeline with top-5 quote retrieval.
        self._password: str = gen.get_secret(r'D:\Documents\Secrets\postgres_password.txt')
        self._user_name: str = "postgres"
        self._db_name: str = "postgres"
        self._table_name: str = "book_archive"
        self._doc_pipeline = DocRetrievalPipeline(
            table_name=self._table_name,
            db_user_name=self._user_name,
            db_password=self._password,
            postgres_host='localhost',
            postgres_port=5432,
            db_name=self._db_name,
            verbose=False,
            llm_top_k=5,
            retriever_top_k_docs=100,
            include_outputs_from=None,
            search_mode=SearchMode.HYBRID,
            use_reranker=True,
            embedder_model_name="BAAI/llm-embedder"
        )
        self._load_pipeline: Optional[DocumentProcessor] = None

    def _initialize_model(self, system_instruction: Optional[str]):
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            system_instruction=system_instruction
        )
        self._model = model

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
        chat_session = self._model.start_chat(history=chat_history)
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
            f"Reviewing the query and chat history, determine if you can provide a better wording for the query "
            f"that might yield better results. If you can improve the query, return the improved "
            f"query. If you cannot improve the question, return an empty string (without quotes around it) and we'll "
            f"continue with the user's original query. There is no need to explain your thinking if you want to return "
            f"an empty string. Do not return quotes around your answer.\n\n"
            f"You must either return a single sentence or phrase that is the new query (without quotes around it) or "
            f"an empty string (without quotes around it). Keep the new query as concise as possible to improve matches."
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

    def load_documents(self, files: List[str]) -> Iterator[None]:
        if self._load_pipeline is None:
            self._load_pipeline: DocumentProcessor = DocumentProcessor(
                table_name=self._table_name,
                recreate_table=False,
                embedder_model_name="BAAI/llm-embedder",
                file_folder_path_or_list=files,
                db_user_name=self._user_name,
                db_password=self._password,
                postgres_host='localhost',
                postgres_port=5432,
                db_name=self._db_name,
                min_section_size=3000,
                min_paragraph_size=300,
            )
        # Load the documents into the database.
        for _ in self._load_pipeline.run(files, use_iterator=True):
            yield

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
        retrieved_docs, all_docs = self._doc_pipeline.generate_response(message)

        # Find the largest score
        max_score: float = self.get_max_score(retrieved_docs)

        if max_score is not None and max_score < 0.50:
            # If we don't have any good quotes, ask the LLM if it wants to do its own search
            improved_query: str = self.ask_llm_for_improved_query(message, gemini_chat_history)
            # The LLM is sometimes stupid and takes my example too literally and returns "''" instead of "" for an
            # empty string. So we need to check for that and convert it to an empty string.
            # Unfortunately, dropping that instruction tends to cause it to think out loud before returning an empty
            # string at the end. Which sort of defeats the purpose.

            # Strip off double or single quotes if the improved query starts and ends with them.
            if improved_query.startswith(('"', "'")) and improved_query.endswith(('"', "'")):
                improved_query = improved_query[1:-1]
            if improved_query.lower() == "empty string":
                improved_query = ""

            new_retrieved_docs: List[Document]
            temp_all_docs: List[Document]
            if improved_query != "":
                new_retrieved_docs, temp_all_docs = self._doc_pipeline.generate_response(improved_query)
                new_max_score: float = self.get_max_score(new_retrieved_docs)
                if new_max_score > max(max_score * 1.1, max_score + 0.05):
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

        modified_query: str
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
        chat_session = self._model.start_chat(history=gemini_chat_history)
        # Send the modified query to Gemini.
        chat_response = chat_session.send_message(modified_query, stream=True)
        answer_text = ""
        # # --- Step 3: Stream the answer character-by-character ---
        for chunk in chat_response:
            if hasattr(chunk, 'text'):
                answer_text += chunk.text
                yield chat_history + [(message, answer_text)], retrieved_quotes, all_quotes

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


def build_interface(title: str = 'RAG Chat', system_instruction: Optional[str] = None) -> gr.Interface:
    karl_chat = RagChat(system_instruction=system_instruction)
    css: str = """
    #QuoteBoxes {
        height: calc(100vh - 175px);
        overflow-y: auto;
        white-space: pre-wrap;
    """
    with gr.Blocks(css=css) as chat_interface:
        with gr.Tab("Chat"):
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Row():
                        with gr.Column(scale=3):
                            gr.Markdown("## " + title)
                            gr.Markdown("Chat on the left. "
                                        "It will cite sources from the retrieved quotes on the right.")
                        with gr.Column(scale=1):
                            clear = gr.Button("Clear Chat")
                    chatbot = gr.Chatbot(label="Chat")
                    msg = gr.Textbox(placeholder="Ask your question...", label="Your Message")
                with gr.Column(scale=1):
                    with gr.Tab("Retrieved Quotes"):
                        retrieved_quotes_box = gr.Markdown(label="Retrieved Quotes & Metadata", value="",
                                                           elem_id="QuoteBoxes")
                    with gr.Tab("Raw Quotes"):
                        raw_quotes_box = gr.Markdown(label="Raw Quotes & Metadata", value="", elem_id="QuoteBoxes")
        with gr.Tab("Load"):
            gr.Markdown("Drag and drop your files here to load them into the database. ")
            gr.Markdown("Supported file types: PDF and EPUB.")
            file_input = gr.File(file_count="multiple", label="Upload a file", interactive=True)
            load_button = gr.Button("Load")

        def user_message(message, chat_history):
            updated_history = chat_history + [(message, None)]
            return "", updated_history

        def process_message(message, chat_history):
            for updated_history, ranked_docs, all_docs in karl_chat.respond(message, chat_history):
                yield updated_history, ranked_docs.strip(), all_docs.strip()

        def process_with_custom_progress(files, progress=gr.Progress()):
            if files is None or len(files) == 0:
                # If no files, immediately yield cleared file list and 0% progress.
                return

            # Call the load_documents method, which now yields progress (a float between 0 and 1)
            file_enumerator = karl_chat.load_documents(files)
            for i, file in enumerate(files):
                file_name = os.path.basename(file)
                desc = f"Processing {file_name}"
                prog = i / len(files)
                progress(prog, desc=desc)
                next(file_enumerator)
            progress(1.0, desc="Finished processing")
            time.sleep(0.5)
            return "Finished processing"

        def update_progress(files):
            # Process the files and return a progress message along with an empty list to clear the widget
            process_with_custom_progress(files)
            return []

        load_button.click(update_progress, inputs=file_input, outputs=file_input)

        msg.submit(user_message, [msg, chatbot], [msg, chatbot], queue=True)
        msg.submit(process_message, [msg, chatbot],
                   [chatbot, retrieved_quotes_box, raw_quotes_box], queue=True)
        clear.click(lambda: ([], "", ""), None, [chatbot, retrieved_quotes_box, raw_quotes_box], queue=False)

    return chat_interface


if __name__ == "__main__":
    sys_instruction: str = ("You are philosopher Karl Popper. Answer questions with philosophical insights, and use "
                            "the provided quotes along with their metadata as reference.")
    rag_chat = build_interface(title="Karl Popper Chatbot", system_instruction=sys_instruction)
    rag_chat.launch(debug=True, max_file_size=100 * gr.FileSize.MB)
