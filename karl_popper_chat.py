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
from typing import Optional, List


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

    @staticmethod
    def format_document(doc) -> str:
        """
        Format a document by including its metadata followed by the quote.
        """
        formatted = ""
        # If the document has metadata, format each key-value pair.
        if hasattr(doc, 'meta') and doc.meta:
            meta_entries = ["Score: {:.4f}".format(doc.score) if hasattr(doc, 'score') else "Score: N/A"]
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
    def transform_history(history):
        new_history = []
        for chat_response in history:
            new_history.append({"parts": [{"text": chat_response[0]}], "role": "user"})
            new_history.append({"parts": [{"text": chat_response[1]}], "role": "model"})
        return new_history

    def respond(self, message, chat_history):
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

        docs: List[Document] = self.doc_pipeline.generate_response(message)

        # Find the largest score
        max_score: float = 100.0
        if docs:
            max_score = max(doc.score for doc in docs if hasattr(doc, 'score'))

        if max_score is not None and max_score < 0.30:
            # If there are no quotes with a scare at least 0.30, then we need to do additional checks
            # Let's ask Gemini outside a chat session if each quote is relevant to the question or not
            relevant_docs = []
            for doc in docs:
                try:
                    prompt = (
                        f"Given the question: '{message}', determine if the following quote will help answer "
                        f"the question. \n\nQuote: {doc.content}\n\nAnswer with only 'yes' or 'no'."
                    )
                    # Start a new chat session with no history for this check.
                    chat_session = self.model.start_chat(history=[])
                    chat_response = chat_session.send_message(prompt)
                    # If Gemini's response includes 'yes', consider the quote relevant.
                    if "yes" in chat_response.text.lower():
                        relevant_docs.append(doc)
                except Exception as e:
                    # Capture any exceptions. Likely to exceeding quota for Gemini rate limits.
                    # Wait for a few seconds before continuing.
                    print(f"Error during doc relevance check: {e}")
                    time.sleep(1)
                    break
            docs = relevant_docs
        else:
            # Drop any quotes with a score less than 0.20 if we have at least 3 quotes above 20
            # Otherwise drop any quotes with a score less than 0.10
            # Count how many quotes have a score >= 0.20.
            threshold: float = 0.20
            num_high = len([doc for doc in docs if hasattr(doc, 'score') and doc.score >= threshold])
            # If we have at least 3 such quotes, drop any with a score less than 0.20.
            # Otherwise, drop quotes with a score less than 0.10.
            threshold = 0.20 if num_high >= 3 else 0.10
            docs = [doc for doc in docs if hasattr(doc, 'score') and doc.score >= threshold]

        # Format each retrieved document (quote + metadata).
        formatted_docs = [self.format_document(doc) for doc in docs]
        quotes_text = "\n\n".join(formatted_docs)

        modified_query: str = ""
        if not quotes_text or quotes_text.strip() == "":
            modified_query = message
        elif max_score > 0.50:
            modified_query = (
                f"Use the following quotes with their metadata as reference in your answer:\n\n{quotes_text}\n\n"
                f"Reference the quotes and their metadata in your answer where possible. "
                f"Now, answer the following question: {message}"
            )
        else:
            modified_query = (
                f"The following quotes are available. You may use them as reference to answer my question"
                f"if you find them relevant:\n\n{quotes_text}\n\n"
                f"Reference the quotes and their metadata in your answer if used. But don't "
                f"feel obligated to use the quotes if they are not relevant. "
                f"Now, answer the following question: {message}"
            )
        # Put the chat_history into the correct format for Gemini
        gemini_chat_history = self.transform_history(chat_history)
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
                yield chat_history + [(message, answer_text)], quotes_text


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
            for updated_history, quotes_text in karl_chat.respond(message, chat_history):
                yield updated_history, quotes_text

        msg.submit(user_message, [msg, chatbot], [msg, chatbot], queue=True)
        msg.submit(process_message, [msg, chatbot], [chatbot, quotes_box], queue=True)
        clear.click(lambda: ([], ""), None, [chatbot, quotes_box], queue=False)

    return chat_interface


if __name__ == "__main__":
    popper_chat = build_interface()
    popper_chat.launch(debug=True)
