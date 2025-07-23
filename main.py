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
import hashlib
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

def get_doc_hash(file_path):
    """Generate a unique hash for the document to avoid duplicate uploads."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def process_file_and_store(file_obj):
    suffix = os.path.splitext(file_obj.name)[1]
    loader_cls = LOADER_MAP.get(suffix.lower())

    if not loader_cls:
        return f"Unsupported file type: {suffix}", None

    tmp_path = file_obj.name
    file_hash = get_doc_hash(tmp_path)

    # Check if already uploaded
    if os.path.exists("uploaded_docs.txt"):
        with open("uploaded_docs.txt", "r", encoding="utf-8") as f:
            uploaded_hashes = {line.strip() for line in f.readlines()}

        if file_hash in uploaded_hashes:
            return f"'{os.path.basename(file_obj.name)}' is already uploaded. Skipping re-upload.", True
    else:
        uploaded_hashes = set()

    # Process new document
    docs = loader_cls(tmp_path, encoding="utf-8").load()

    if not docs:
        return "Failed to load document or document is empty.", None

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    if not chunks:
        return "No readable text found in the document.", None

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.exists("db") and os.listdir("db"):
        vectordb = Chroma(persist_directory="db", embedding_function=embeddings)
        vectordb.add_documents(chunks)
        print("Total documents stored (after adding):", vectordb._collection.count())
    else:
        vectordb = Chroma.from_documents(chunks, embeddings, persist_directory="db")
        print("Total documents stored (new DB):", vectordb._collection.count())

    # Save hash to avoid future duplicates
    with open("uploaded_docs.txt", "a", encoding="utf-8") as f:
        f.write(file_hash + "\n")

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
