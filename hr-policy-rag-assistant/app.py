import os
import gradio as gr
import google.generativeai as genai

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Gemini

genai.configure(
api_key=os.getenv("GEMINI_API_KEY")
)

model = genai.GenerativeModel(
"gemini-2.5-flash"
)

# Embeddings

embeddings = HuggingFaceEmbeddings(
model_name="BAAI/bge-base-en-v1.5"
)

# FAISS

vectorstore = FAISS.load_local(
"enterprise_faiss",
embeddings,
allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(
search_kwargs={"k":3}
)

def ask(question):

    docs = retriever.invoke(question)

    return "\n\n=================\n\n".join(
        d.page_content[:1000]
        for d in docs
    )

    prompt = f"""
You are an HR policy assistant.

Answer ONLY using the provided context.

If the answer is not explicitly present,
respond exactly:

I could not find this information in the documents.

Context:
{context}

Question:
{question}

Answer:
"""

    response = model.generate_content(
        prompt
    )

    answer = response.text

    sources = []

    for d in docs:

        source_file = d.metadata.get(
            "source_file",
            "Unknown"
        )

        page = d.metadata.get(
            "page",
            0
        )

        sources.append(
            f"{source_file} (Page {page+1})"
        )

    return (
        answer
        + "\n\nSources:\n"
        + "\n".join(sources)
    )


demo = gr.Interface(
fn=ask,
inputs=gr.Textbox(
lines=2,
label="Ask a policy question"
),
outputs=gr.Textbox(
lines=15,
label="Answer"
),
title="HR Policy Assistant (RAG)",
description="Enterprise HR Policy Assistant using FAISS + Gemini"
)

demo.launch()
