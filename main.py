import gradio as gr
from graph_builder import build_graph
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader,
    UnstructuredWordDocumentLoader, UnstructuredMarkdownLoader,
    UnstructuredImageLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import shutil
import os

# OCR settings (if needed for images)
os.environ["TESSDATA_PREFIX"] = r"C:\Users\mansi.patil\AppData\Local\Programs\Tesseract-OCR\tessdata"
os.environ["PATH"] += os.pathsep + r"C:\Users\mansi.patil\AppData\Local\Programs\Tesseract-OCR"

# File loader mapping
LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".md": UnstructuredMarkdownLoader,
    ".png": UnstructuredImageLoader,
    ".jpg": UnstructuredImageLoader,
    ".jpeg": UnstructuredImageLoader,
}

# Build RAG Agent graph (workflow)
graph = build_graph()

def process_file_and_store(file_obj):
    suffix = os.path.splitext(file_obj.name)[1]
    loader_cls = LOADER_MAP.get(suffix.lower())

    if not loader_cls:
        return f"Unsupported file type: {suffix}", None

    tmp_path = file_obj.name
    docs = loader_cls(tmp_path).load()

    if not docs:
        return "Failed to load document or document is empty.", None

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    if not chunks:
        return "No readable text found in the document.", None

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.exists("db"):
        try:
            shutil.rmtree("db")
        except PermissionError:
            pass

    Chroma.from_documents(chunks, embeddings, persist_directory="db", collection_name="rag_collection")

    with open("current_file.txt", "w", encoding="utf-8") as f:
        f.write(os.path.basename(file_obj.name))

    print(f"Stored {len(chunks)} chunks for '{os.path.basename(file_obj.name)}'")
    return f"Stored {len(chunks)} chunks for '{os.path.basename(file_obj.name)}'.", True

def chat(query, history):
    result = graph.invoke({"input": query})
    answer = result.get("answer", "I cannot find this information in the document.")

    history = history or []
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": answer})
    return history, history

with gr.Blocks() as demo:
    gr.Markdown("LangGraph RAG Agent Conversational Chat")

    file_upload = gr.File(
        label="Upload a document",
        file_types=[".pdf", ".txt", ".docx", ".md", ".png", ".jpg", ".jpeg"]
    )
    upload_status = gr.Textbox(label="Upload Status", interactive=False)

    chatbot = gr.Chatbot(label="Conversation", type="messages")
    msg = gr.Textbox(label="Ask a question about the uploaded document:")

    state = gr.State([])

    file_upload.upload(process_file_and_store, inputs=[file_upload], outputs=[upload_status])
    msg.submit(chat, [msg, state], [chatbot, state]).then(lambda: "", None, [msg])

if __name__ == "__main__":
    demo.launch()
