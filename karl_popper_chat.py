import gradio as gr
# noinspection PyPackageRequirements
from google.genai import Client
# noinspection PyPackageRequirements
from google.genai.types import GenerateContentConfig
import generator_model as gen
# Import your DocRetrievalPipeline and SearchMode (adjust import paths as needed)
from doc_retrieval_pipeline import DocRetrievalPipeline, SearchMode
# noinspection PyPackageRequirements
from haystack import Document


class KarlPopperChat:
    def __init__(self):
        # Initialize Gemini Chat with a system instruction to act like philosopher Karl Popper.
        google_secret: str = gen.get_secret(r'D:\Documents\Secrets\gemini_secret.txt')
        client: Client = Client(api_key=google_secret)
        config: GenerateContentConfig = GenerateContentConfig(
            system_instruction="You are philosopher Karl Popper. Answer questions with philosophical insights, "
                               "and use the provided quotes along with their metadata as reference."
        )
        self.chat = client.chats.create(model="gemini-1.5-flash", config=config)

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

    @staticmethod
    def format_document(doc) -> str:
        """
        Format a document by including its metadata followed by the quote.
        """
        formatted = ""
        # If the document has metadata, format each key-value pair.
        if hasattr(doc, 'meta') and doc.meta:
            meta_entries = []
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

    def respond(self, message, chat_history):
        # --- Step 1: Retrieve the top-5 quotes with metadata ---
        # inputs = {
        #     "query_input": {"query": message, "llm_top_k": self.doc_pipeline.llm_top_k}
        # }
        # results = self.doc_pipeline._pipeline.run(inputs)
        if message.strip() == "" or message is None:
            # This is a kludge to deal with the fact that Gradio sometimes get a race condition and we lose the message
            # To correct, try to get the last message from chat history
            if chat_history and len(chat_history) > 0 and chat_history[-1][1] is None:
                # If the last message has no response, then grab the message portion and remove it
                # It will get added back again below
                # There has got to be a better way to do this, but this will work for now
                message = chat_history[-1][0]
                # Remove last message from chat history
                chat_history = chat_history[:-1]

        docs: list[Document] = self.doc_pipeline.generate_response(message)

        # Format each retrieved document (quote + metadata).
        formatted_docs = [self.format_document(doc) for doc in docs]
        quotes_text = "\n\n".join(formatted_docs)

        modified_query = (
            f"Use the following quotes with their metadata as reference in your answer:\n\n{quotes_text}\n\n"
            f"Reference the quotes and their metadata in your answer where possible. "
            f"Now, answer the following question: {message}"
        )

        # Send the modified query to Gemini.
        chat_response = self.chat.send_message(modified_query)
        answer_text = chat_response.text

        # --- Step 3: Stream the answer character-by-character ---
        return chat_history + [(message, answer_text)], quotes_text
        # How to do streaming
        # current_text = ""
        # for char in answer_text:
        #     current_text += char
        #     yield chat_history + [(message, current_text)], quotes_text


def build_interface():
    karl_chat = KarlPopperChat()

    with gr.Blocks() as chat_interface:
        gr.Markdown("# Karl Popper Chatbot")
        gr.Markdown(
            "This chatbot retrieves quotes with metadata from a document store and uses them as context "
            "for its Gemini-powered responses. The quotes and metadata are displayed on the right.")
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Chat")
                msg = gr.Textbox(placeholder="Ask your question...", label="Your Message")
                clear = gr.Button("Clear Chat")
            with gr.Column(scale=1):
                quotes_box = gr.Textbox(label="Retrieved Quotes & Metadata", interactive=False, lines=15)

        def user_message(message, chat_history):
            # print(f"user_message: User submitted message: '{message}'")
            # Append the user's message to the chat history
            updated_history = chat_history + [(message, None)]
            return "", updated_history

        def process_message(message, chat_history):
            # print(f"process_message: User submitted message: '{message}'")
            updated_history, quotes_text = karl_chat.respond(message, chat_history)
            yield updated_history, quotes_text
            # How to do streaming
            # for updated_history, quotes_text in karl_chat.respond(message, chat_history):
            #     yield updated_history, quotes_text

        msg.submit(user_message, [msg, chatbot], [msg, chatbot], queue=True)
        msg.submit(process_message, [msg, chatbot], [chatbot, quotes_box], queue=True)
        clear.click(lambda: ([], ""), None, [chatbot, quotes_box], queue=False)

    return chat_interface


if __name__ == "__main__":
    popper_chat = build_interface()
    popper_chat.launch(debug=True)
