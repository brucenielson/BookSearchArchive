import time
import os
import gradio as gr
# noinspection PyPackageRequirements
from google.generativeai.types import generation_types
# noinspection PyPackageRequirements
# from google.genai import Client
# noinspection PyPackageRequirements
# from google.genai.types import GenerateContentConfig
# noinspection PyPackageRequirements
import google.generativeai as genai
# Import your DocRetrievalPipeline and SearchMode (adjust import paths as needed)
from doc_retrieval_pipeline import DocRetrievalPipeline, SearchMode
from document_processor import DocumentProcessor
# noinspection PyPackageRequirements
from haystack import Document
from typing import Optional, List, Dict, Any, Iterator, Union


class RagChat:
    def __init__(self,
                 google_secret: str,
                 postgres_password: str,
                 postgres_user_name: str = "postgres",
                 postgres_db_name: str = "postgres",
                 postgres_table_name: str = "book_archive",
                 postgres_host: str = 'localhost',
                 postgres_port: int = 5432,
                 postgres_table_recreate: bool = False,
                 postgres_table_embedder_model_name: str = "BAAI/llm-embedder",
                 system_instruction: Optional[str] = None, ):

        # Initialize Gemini Chat with a system instruction to act like philosopher Karl Popper.
        self._model: Optional[genai.GenerativeModel] = None
        self._system_instruction: Optional[str] = system_instruction
        self._google_secret: str = google_secret
        self.initialize_model(system_instruction=system_instruction, google_secret=google_secret)

        # Initialize the document retrieval pipeline with top-5 quote retrieval.
        self._postgres_password: str = postgres_password
        self._postgres_user_name: str = postgres_user_name
        self._postgres_db_name: str = postgres_db_name
        self._postgres_table_name: str = postgres_table_name
        self._postgres_table_recreate: bool = postgres_table_recreate
        self._postgres_table_embedder_model_name: str = postgres_table_embedder_model_name
        self._postgres_host: str = postgres_host
        self._postgres_port: int = postgres_port
        # Initialize the document retrieval pipeline.
        self._doc_pipeline = DocRetrievalPipeline(
            table_name=self._postgres_table_name,
            db_user_name=self._postgres_user_name,
            db_password=self._postgres_password,
            postgres_host=self._postgres_host,
            postgres_port=self._postgres_port,
            db_name=self._postgres_db_name,
            verbose=False,
            llm_top_k=5,
            retriever_top_k_docs=100,
            include_outputs_from=None,
            search_mode=SearchMode.HYBRID,
            use_reranker=True,
            embedder_model_name=self._postgres_table_embedder_model_name,
        )
        self._load_pipeline: Optional[DocumentProcessor] = None

    def initialize_model(self, system_instruction: Optional[str] = None, google_secret: Optional[str] = None):
        genai.configure(api_key=google_secret)
        self._google_secret = google_secret
        model: genai.GenerativeModel = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            system_instruction=system_instruction
        )
        self._model = model

    def ask_llm_question(self, prompt: str,
                         chat_history: Optional[List[Dict[str, Any]]] = None,
                         stream: bool = False) -> Union[generation_types.GenerateContentResponse, str]:
        if chat_history is None:
            chat_history = []
        if self._google_secret is not None and self._google_secret != "":
            # Start a new chat session with no history for this check.
            chat_session = self._model.start_chat(history=chat_history)
            chat_response = chat_session.send_message(prompt, stream=stream)
            # If streaming is enabled, return the response object.
            if stream:
                return chat_response
            # If streaming is not enabled, return the full response text.
            else:
                return chat_response.text.strip()
        else:
            # If no secret is provided, throw an error
            raise ValueError("Google secret is not provided. Please provide a valid API key.")

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
                table_name=self._postgres_table_name,
                recreate_table=False,
                embedder_model_name="BAAI/llm-embedder",
                file_folder_path_or_list=files,
                db_user_name=self._postgres_user_name,
                db_password=self._postgres_password,
                postgres_host='localhost',
                postgres_port=5432,
                db_name=self._postgres_db_name,
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
        # Send the modified query to Gemini.
        chat_response = self.ask_llm_question(modified_query, chat_history=gemini_chat_history, stream=True)
        # chat_response = chat_session.send_message(modified_query, stream=True)
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


# --- Chat Tab ---
def build_chat_tab(title: str, default_tab: str):
    with gr.Tab(label="Chat", id="Chat", interactive=(default_tab == "Chat")) as chat_tab:
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row():
                    with gr.Column(scale=3):
                        title_md = gr.Markdown("## " + title)
                        gr.Markdown("Chat on the left. It will cite sources from the retrieved quotes on the right.")
                    with gr.Column(scale=1):
                        clear = gr.Button("Clear Chat")
                chatbot = gr.Chatbot(label="Chat")
                msg = gr.Textbox(placeholder="Ask your question...", label="Your Message")
            with gr.Column(scale=1):
                with gr.Tab("Retrieved Quotes"):
                    retrieved_quotes_box = gr.Markdown(
                        label="Retrieved Quotes & Metadata", value="", elem_id="QuoteBoxes"
                    )
                with gr.Tab("Raw Quotes"):
                    raw_quotes_box = gr.Markdown(
                        label="Raw Quotes & Metadata", value="", elem_id="QuoteBoxes"
                    )
    return {
        "chat_tab": chat_tab,
        "title_md": title_md,
        "clear": clear,
        "chatbot": chatbot,
        "msg": msg,
        "retrieved_quotes_box": retrieved_quotes_box,
        "raw_quotes_box": raw_quotes_box,
    }


# --- Load Tab ---
def build_load_tab(default_tab: str):
    with gr.Tab(label="Load", id="Load", interactive=(default_tab == "Chat")) as load_tab:
        gr.Markdown("Drag and drop your files here to load them into the database.")
        gr.Markdown("Supported file types: PDF and EPUB.")
        file_input = gr.File(file_count="multiple", label="Upload a file", interactive=True)
        load_button = gr.Button("Load")
    return {
        "load_tab": load_tab,
        "file_input": file_input,
        "load_button": load_button,
    }


# --- Config Tab ---
def build_config_tab(config_data: dict):
    with gr.Tab(label="Config", id="Config") as config_tab:
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("Settings for chat and load.")
                gr.Markdown("### Chat Settings")
            with gr.Column(scale=1):
                save_settings = gr.Button("Save Settings")
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    chat_title_tb = gr.Textbox(
                        label="Chat Title",
                        placeholder="Enter the title for the chat",
                        value=config_data["title"],
                        interactive=True,
                    )
                    sys_inst_box_tb = gr.Textbox(
                        label="System Instructions",
                        placeholder="Enter your system instructions here",
                        value=config_data["system_instructions"],
                        interactive=True,
                    )
                gr.Markdown("### API Keys")
                with gr.Group():
                    google_secret_tb = gr.Textbox(
                        label="Gemini API Key",
                        placeholder="Enter your Gemini API key here",
                        value=config_data["google_password"],
                        type="password",
                        interactive=True,
                    )
                gr.Markdown("### Postgres Settings")
                with gr.Group():
                    postgres_secret_tb = gr.Textbox(
                        label="Postgres Password",
                        placeholder="Enter your Postgres password here",
                        value=config_data["postgres_password"],
                        type="password",
                        interactive=True,
                    )
                    postgres_user_tb = gr.Textbox(
                        label="Postgres User",
                        placeholder="Enter your Postgres user here",
                        value=config_data["postgres_user_name"],
                        interactive=True,
                    )
                    postgres_db_tb = gr.Textbox(
                        label="Postgres DB",
                        placeholder="Enter your Postgres DB name here",
                        value=config_data["postgres_db_name"],
                        interactive=True,
                    )
                    postgres_table_tb = gr.Textbox(
                        label="Postgres Table",
                        placeholder="Enter your Postgres table name here",
                        value=config_data["postgres_table_name"],
                        interactive=True,
                    )
                    postgres_host_tb = gr.Textbox(
                        label="Postgres Host",
                        placeholder="Enter your Postgres host here",
                        value=config_data["postgres_host"],
                        interactive=True,
                    )
                    postgres_port_tb = gr.Textbox(
                        label="Postgres Port",
                        placeholder="Enter your Postgres port here",
                        value=str(config_data["postgres_port"]),
                        interactive=True,
                    )
    return {
        "config_tab": config_tab,
        "save_settings": save_settings,
        "chat_title_tb": chat_title_tb,
        "sys_inst_box_tb": sys_inst_box_tb,
        "google_secret_tb": google_secret_tb,
        "postgres_secret_tb": postgres_secret_tb,
        "postgres_user_tb": postgres_user_tb,
        "postgres_db_tb": postgres_db_tb,
        "postgres_table_tb": postgres_table_tb,
        "postgres_host_tb": postgres_host_tb,
        "postgres_port_tb": postgres_port_tb,
    }


def build_interface(title: str = 'RAG Chat',
                    system_instructions: str = "You are a helpful assistant.") -> gr.Interface:
    def load_rag_chat(google_secret_param: str,
                      postgres_password_param: str,
                      postgres_user_name_param: str,
                      postgres_db_name_param: str,
                      postgres_table_name_param: str,
                      postgres_host_param: str,
                      postgres_port_param: int,
                      system_instructions_param: str) -> RagChat:
        return RagChat(
            google_secret=google_secret_param,
            postgres_password=postgres_password_param,
            postgres_user_name=postgres_user_name_param,
            postgres_db_name=postgres_db_name_param,
            postgres_table_name=postgres_table_name_param,
            postgres_host=postgres_host_param,
            postgres_port=postgres_port_param,
            system_instruction=system_instructions_param
        )

    # noinspection PyShadowingNames
    def load_config_data():
        google_password: str = ""
        postgres_password: str = ""
        postgres_user_name: str = "postgres"
        postgres_db_name: str = "postgres"
        postgres_table_name: str = "book_archive"
        postgres_host: str = "localhost"
        postgres_port: int = 5432
        title: str = ""
        system_instructions: str = ""
        if os.path.exists("config.txt"):
            with open("config.txt", "r") as f:
                lines = f.readlines()
                if len(lines) >= 9:
                    google_password = lines[0].strip()
                    postgres_password = lines[1].strip()
                    postgres_user_name = lines[2].strip()
                    postgres_db_name = lines[3].strip()
                    postgres_table_name = lines[4].strip()
                    postgres_host = lines[5].strip()
                    postgres_port = int(lines[6].strip())
                    title = lines[7].strip()
                    system_instructions = lines[8].strip()
        return {
            "google_password": google_password,
            "postgres_password": postgres_password,
            "postgres_user_name": postgres_user_name,
            "postgres_db_name": postgres_db_name,
            "postgres_table_name": postgres_table_name,
            "postgres_host": postgres_host,
            "postgres_port": int(postgres_port),
            "system_instructions": system_instructions,
            "title": title,
        }

    def load_event():
        nonlocal config_data, rag_chat
        load_config()
        # Return an update for each Textbox in the same order as the outputs list below.
        return (
            gr.update(value=config_data["title"]),
            gr.update(value=config_data["system_instructions"]),
            gr.update(value=config_data["google_password"]),
            gr.update(value=config_data["postgres_password"]),
            gr.update(value=config_data["postgres_user_name"]),
            gr.update(value=config_data["postgres_db_name"]),
            gr.update(value=config_data["postgres_table_name"]),
            gr.update(value=config_data["postgres_host"]),
            gr.update(value=str(config_data["postgres_port"])),
            gr.update(interactive=(rag_chat is not None)),
            gr.update(interactive=(rag_chat is not None)),
        )

    def load_config():
        nonlocal rag_chat, config_data, title, system_instructions
        # Load the config data from the file
        config_data = load_config_data()
        if config_data["title"] is None or config_data["title"] == "":
            config_data["title"] = title
        if config_data["system_instructions"] is None or config_data["system_instructions"] == "":
            config_data["system_instructions"] = system_instructions

        # Check if google_password and postgres_password are not empty
        if not (config_data["google_password"] == "" or config_data["postgres_password"] == ""):
            # Attempt to load RagChat with loaded values
            try:
                rag_chat = load_rag_chat(config_data["google_password"],
                                         config_data["postgres_password"],
                                         config_data["postgres_user_name"],
                                         config_data["postgres_db_name"],
                                         config_data["postgres_table_name"],
                                         config_data["postgres_host"],
                                         int(config_data["postgres_port"]),
                                         config_data["system_instructions"])
            except Exception as e:
                rag_chat = None
        # If RagChat was not loaded (None) then simply return the default values

    rag_chat: Optional[RagChat] = None
    config_data: dict = {}
    load_config()
    default_tab: str = "Chat"
    if not rag_chat:
        # No config settings yet, so set Config tab as default
        default_tab: str = "Config"

    css: str = """
    #QuoteBoxes {
        height: calc(100vh - 175px);
        overflow-y: auto;
        white-space: pre-wrap;
    """
    with gr.Blocks(css=css) as chat_interface:
        with gr.Tabs(selected=default_tab) as tabs:
                    postgres_port_tb, chat_tab, load_tab,
            chat_components = build_chat_tab(title, default_tab)
            load_components = build_load_tab(default_tab)
            config_components = build_config_tab(config_data)

        # Unpack Chat Tab components
        chat_tab = chat_components["chat_tab"]
        title_md = chat_components["title_md"]
        clear = chat_components["clear"]
        chatbot = chat_components["chatbot"]
        msg = chat_components["msg"]
        retrieved_quotes_box = chat_components["retrieved_quotes_box"]
        raw_quotes_box = chat_components["raw_quotes_box"]

        # Unpack Load Tab components
        load_tab = load_components["load_tab"]
        file_input = load_components["file_input"]
        load_button = load_components["load_button"]

        # Unpack Config Tab components
        config_tab = config_components["config_tab"]
        save_settings = config_components["save_settings"]
        chat_title_tb = config_components["chat_title_tb"]
        sys_inst_box_tb = config_components["sys_inst_box_tb"]
        google_secret_tb = config_components["google_secret_tb"]
        postgres_secret_tb = config_components["postgres_secret_tb"]
        postgres_user_tb = config_components["postgres_user_tb"]
        postgres_db_tb = config_components["postgres_db_tb"]
        postgres_table_tb = config_components["postgres_table_tb"]
        postgres_host_tb = config_components["postgres_host_tb"]
        postgres_port_tb = config_components["postgres_port_tb"]

        # Attach the load event on the Blocks container:
        chat_interface.load(
            load_event,
            outputs=[
                chat_title_tb, sys_inst_box_tb, google_secret_tb, postgres_secret_tb,
                postgres_user_tb, postgres_db_tb, postgres_table_tb, postgres_host_tb,
                postgres_port_tb, chat_tab, load_tab,
            ]
        )

        def user_message(message, chat_history):
            updated_history = chat_history + [(message, None)]
            return "", updated_history

        def process_message(message, chat_history):
            for updated_history, ranked_docs, all_docs in rag_chat.respond(message, chat_history):
                yield updated_history, ranked_docs.strip(), all_docs.strip()

        def process_with_custom_progress(files, progress=gr.Progress()):
            if files is None or len(files) == 0:
                # If no files, immediately yield cleared file list and 0% progress.
                return

            # Call the load_documents method, which now yields progress (a float between 0 and 1)
            file_enumerator = rag_chat.load_documents(files)
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

        def update_config(google_password_param: str,
                          postgres_password_param: str,
                          postgres_user_name_param: str,
                          postgres_db_name_param: str,
                          postgres_table_name_param: str,
                          postgres_host_param: str,
                          postgres_port_param: str,
                          title_param: str,
                          system_instructions_param: str):
            nonlocal rag_chat

            # Save the settings to a file
            with open("config.txt", "w") as file:
                file.write(f"{google_password_param}\n")
                file.write(f"{postgres_password_param}\n")
                file.write(f"{postgres_user_name_param}\n")
                file.write(f"{postgres_db_name_param}\n")
                file.write(f"{postgres_table_name_param}\n")
                file.write(f"{postgres_host_param}\n")
                file.write(f"{int(postgres_port_param)}\n")
                file.write(f"{title_param}\n")
                file.write(f"{system_instructions_param}\n")

            # Reset the RagChat instance with the new settings
            rag_chat = load_rag_chat(google_password_param,
                                     postgres_password_param,
                                     postgres_user_name_param,
                                     postgres_db_name_param,
                                     postgres_table_name_param,
                                     postgres_host_param,
                                     int(postgres_port_param),
                                     system_instructions_param)

            return (
                google_password_param,
                postgres_password_param,
                postgres_user_name_param,
                postgres_db_name_param,
                postgres_table_name_param,
                postgres_host_param,
                postgres_port_param,
                title_param,
                system_instructions_param,
                "## " + title_param,
                gr.update(interactive=True),
                gr.update(interactive=True),
            )

        load_button.click(update_progress, inputs=file_input, outputs=file_input)

        msg.submit(user_message, [msg, chatbot], [msg, chatbot], queue=True)
        msg.submit(process_message, [msg, chatbot],
                   [chatbot, retrieved_quotes_box, raw_quotes_box], queue=True)
        clear.click(lambda: ([], "", ""), None, [chatbot, retrieved_quotes_box, raw_quotes_box], queue=False)
        save_settings.click(update_config,
                            inputs=[google_secret_tb, postgres_secret_tb, postgres_user_tb, postgres_db_tb,
                                    postgres_table_tb, postgres_host_tb, postgres_port_tb, chat_title_tb,
                                    sys_inst_box_tb],
                            outputs=[
                                google_secret_tb, postgres_secret_tb, postgres_user_tb, postgres_db_tb,
                                postgres_table_tb, postgres_host_tb, postgres_port_tb, chat_title_tb,
                                sys_inst_box_tb, title_md, chat_tab, load_tab,
                            ],
                            queue=True)

    return chat_interface


if __name__ == "__main__":
    sys_instruction: str = ("You are philosopher Karl Popper. Answer questions with philosophical insights, and use "
                            "the provided quotes along with their metadata as reference.")
    rag_chat_ui = build_interface(title="Karl Popper Chatbot",
                                  system_instructions=sys_instruction)
    rag_chat_ui.launch(debug=True, max_file_size=100 * gr.FileSize.MB)
