import gradio as gr
from google import genai
from google.genai import types
import generator_model as gen


google_secret: str = gen.get_secret(r'D:\Documents\Secrets\gemini_secret.txt')  # Put your path here # noqa: F841
client = genai.Client(api_key=google_secret)
config = types.GenerateContentConfig(system_instruction="You are a Dungeon Master that will play a game with me.")
chat = client.chats.create(model="gemini-1.5-flash", config=config)


# Transform Gradio history to Gemini format
def transform_history(history):
    new_history = []
    for chat_response in history:
        new_history.append({"parts": [{"text": chat_response[0]}], "role": "user"})
        new_history.append({"parts": [{"text": chat_response[1]}], "role": "model"})
    return new_history


def response(message, history):
    global chat
    # The history will be the same as in Gradio, the 'Undo' and 'Clear' buttons will work correctly.
    chat.history = transform_history(history)
    chat_response = chat.send_message(message)

    # Each character of the answer is displayed
    for i in range(len(chat_response.text)):
        yield chat_response.text[: i+1]


def main():
    demo = gr.ChatInterface(response,
                            title='RPG Chat',
                            textbox=gr.Textbox(placeholder="Chat to the Dungeon Master"),
                            )
    demo.launch(debug=True)


if __name__ == "__main__":
    main()
