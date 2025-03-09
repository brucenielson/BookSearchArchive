import gradio as gr
# noinspection PyPackageRequirements
from google.genai import Client
# noinspection PyPackageRequirements
from google.genai.types import GenerateContentConfig
import generator_model as gen

# Import your DocRetrievalPipeline and SearchMode (adjust import paths as needed)
from doc_retrieval_pipeline import DocRetrievalPipeline, SearchMode


class KarlPopperChat:
    def __init__(self):
        # Initialize Gemini Chat with a system instruction to act like philosopher Karl Popper.
        google_secret: str = gen.get_secret(r'D:\Documents\Secrets\gemini_secret.txt')
        client: Client = Client(api_key=google_secret)
        config: GenerateContentConfig = GenerateContentConfig(
            system_instruction="You are philosopher Karl Popper. Answer questions with philosophical insights, and use the provided quotes as reference."
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
            retriever_top_k_docs=5,
            include_outputs_from=None,
            search_mode=SearchMode.HYBRID,
            use_reranker=True,
            embedder_model_name="BAAI/llm-embedder"
        )

    def respond(self, message, chat_history):
        # --- Step 1: Retrieve the top-5 quotes ---
        # Prepare inputs for the document retrieval pipeline.
        inputs = {
            "query_input": {"query": message, "llm_top_k": self.doc_pipeline.llm_top_k}
        }
        # Run the pipeline (this returns a dictionary with the retrieved documents).
        results = self.doc_pipeline._pipeline.run(inputs)
        # Retrieve quotes from the results (assuming they’re under "reranker" -> "documents").
        docs = results["reranker"]["documents"]
        # Extract the content of each document.
        quotes = [doc.content for doc in docs]
        # Join quotes into a single text block for both display and inclusion in the prompt.
        quotes_text = "\n\n".join([f"Quote: {quote}" for quote in quotes])

        # --- Step 2: Prepare modified prompt for Gemini ---
        modified_query = (
            f"You are philosopher Karl Popper. Use the following quotes as reference in your answer:\n\n"
            f"{quotes_text}\n\nNow, answer the following question: {message}"
        )

        # Send the modified query to Gemini.
        chat_response = self.chat.send_message(modified_query)
        answer_text = chat_response.text

        # --- Step 3: Stream the answer character-by-character ---
        current_text = ""
        # Stream each character so that the chat appears to be typing out the answer.
        for char in answer_text:
            current_text += char
            # Update the chat history with the current answer progress.
            yield (chat_history + [(message, current_text)], quotes_text)


def build_interface():
    karl_chat = KarlPopperChat()

    with gr.Blocks() as demo:
        gr.Markdown("# Karl Popper Chatbot")
        gr.Markdown(
            "This chatbot retrieves relevant quotes from a document store and uses them as context for its Gemini-powered responses. The quotes are displayed on the right.")
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Chat")
                msg = gr.Textbox(placeholder="Ask your question...", label="Your Message")
                clear = gr.Button("Clear Chat")
            with gr.Column(scale=1):
                quotes_box = gr.Textbox(label="Retrieved Quotes", interactive=False, lines=15)

        def user_message(message, chat_history):
            # When the user sends a message, simply append it to the history (the answer will be added in the streaming function).
            return "", chat_history + [(message, None)]

        def process_message(message, chat_history):
            # This generator yields both the updated chat history and the quotes text.
            for updated_history, quotes_text in karl_chat.respond(message, chat_history):
                yield updated_history, quotes_text

        msg.submit(user_message, [msg, chatbot], [msg, chatbot], queue=False)
        msg.submit(process_message, [msg, chatbot], [chatbot, quotes_box], queue=True)
        clear.click(lambda: ([], ""), None, [chatbot, quotes_box], queue=False)

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch(debug=True)
