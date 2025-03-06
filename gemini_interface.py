import gradio as gr
# noinspection PyPackageRequirements
from google.genai import Client
# noinspection PyPackageRequirements
from google.genai.types import GenerateContentConfig
# noinspection PyPackageRequirements
from google.genai.chats import Chat
import generator_model as gen


class RPGChat:
    def __init__(self):
        # Initialize your chat model
        google_secret: str = gen.get_secret(r'D:\Documents\Secrets\gemini_secret.txt')
        client: Client = Client(api_key=google_secret)
        config: GenerateContentConfig = GenerateContentConfig(
            system_instruction="You are a Dungeon Master that will play a game with me.")
        # Create a Gemini chat instance and store it as a property of the class
        self.chat: Chat = client.chats.create(model="gemini-1.5-flash", config=config)

    # Transform Gradio history to Gemini format
    def transform_history(self, history):
        new_history = []
        for chat_response in history:
            new_history.append({"parts": [{"text": chat_response[0]}], "role": "user"})
            new_history.append({"parts": [{"text": chat_response[1]}], "role": "model"})
        return new_history

    def response(self, message, history):
        # The history will be the same as in Gradio; the 'Undo' and 'Clear' buttons work correctly.
        self.chat.history = self.transform_history(history)
        chat_response = self.chat.send_message(message)
        # Yield the answer character by character
        for i in range(len(chat_response.text)):
            yield chat_response.text[: i + 1]


def main():
    rpg_chat = RPGChat()
    demo = gr.ChatInterface(
        fn=rpg_chat.response,
        title='RPG Chat',
        textbox=gr.Textbox(placeholder="Chat to the Dungeon Master"),
    )
    demo.launch(debug=True)


if __name__ == "__main__":
    main()
