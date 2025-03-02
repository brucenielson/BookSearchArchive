import gradio as gr
from google import genai
from google.genai import types


def get_secret(secret_file: str) -> str:
    """
    Read a secret from a file.

    Args:
        secret_file (str): Path to the file containing the secret.

    Returns:
        str: The content of the secret file, or an empty string if an error occurs.
    """
    try:
        with open(secret_file, 'r') as file:
            secret_text: str = file.read().strip()
    except FileNotFoundError:
        print(f"The file '{secret_file}' does not exist.")
        secret_text = ""
    except Exception as e:
        print(f"An error occurred: {e}")
        raise e

    return secret_text


google_secret: str = get_secret(r'D:\Documents\Secrets\gemini_secret.txt')  # Put your path here # noqa: F841
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
