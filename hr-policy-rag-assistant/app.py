import os
import gradio as gr
import google.generativeai as genai

genai.configure(
    api_key=os.environ["GEMINI_API_KEY"]
)

model = genai.GenerativeModel(
    "gemini-2.5-flash"
)

def ask(question):

    response = model.generate_content(question)

    return response.text

demo = gr.Interface(
    fn=ask,
    inputs=gr.Textbox(lines=2),
    outputs=gr.Textbox(lines=10),
    title="HR Policy Assistant"
)

demo.launch()