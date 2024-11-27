import gradio as gr
from transformers import pipeline

# Load the TTS pipeline
tts = pipeline("text-to-speech", model="facebook/mms-tts-eng")

def synthesize_speech(text):
    audio = tts(text)
    return audio["audio"]

# Create the Gradio interface
iface = gr.Interface(
    fn=synthesize_speech,
    inputs=gr.inputs.Textbox(label="Input Text"),
    outputs=gr.outputs.Audio(label="Generated Speech"),
    title="Text-to-Speech Application"
)

# Launch the application
iface.launch()
