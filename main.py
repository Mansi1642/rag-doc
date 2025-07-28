import gradio as gr
import os
import json
import hashlib
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredHTMLLoader,
    UnstructuredImageLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from bs4 import BeautifulSoup
from graph_builder import build_graph
 
# Setup Tesseract
os.environ["TESSDATA_PREFIX"] = r"C:\Users\mansi.patil\AppData\Local\Programs\Tesseract-OCR\tessdata"
os.environ["PATH"] += os.pathsep + r"C:\Users\mansi.patil\AppData\Local\Programs\Tesseract-OCR"
 
# Loader mapping
LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".html": UnstructuredHTMLLoader,
    ".png": UnstructuredImageLoader,
    ".jpg": UnstructuredImageLoader,
    ".jpeg": UnstructuredImageLoader,
}
 
# Init state
UPLOAD_LOG = "uploaded_docs.json"
if not os.path.exists(UPLOAD_LOG):
    with open(UPLOAD_LOG, "w") as f:
        json.dump({}, f)
 
with open(UPLOAD_LOG, "r") as f:
    file_hash_map = json.load(f)
 
CURRENT_DOC_ID = {"doc": None}
 
# Setup DB and graph
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory="db", embedding_function=embeddings)
graph = build_graph()
 
 
def get_doc_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()
 
def extract_html_metadata(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
        title = soup.title.string.strip() if soup.title else "Unknown"
        return title 
 
 
def process_file_and_store(file_obj):
    suffix = os.path.splitext(file_obj.name)[1]
    loader_cls = LOADER_MAP.get(suffix.lower())
    if not loader_cls:
        return f"Unsupported file type: {suffix}", None, []

    tmp_path = file_obj.name
    file_hash = get_doc_hash(tmp_path)
    CURRENT_DOC_ID["doc"] = file_hash

    if file_hash in file_hash_map:
        return f"'{os.path.basename(file_obj.name)}' already uploaded. Using existing embeddings.", [], []

    # Load and split
    docs = loader_cls(tmp_path, encoding="utf-8").load() if suffix == ".txt" else loader_cls(tmp_path).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Filter empty chunks
    chunks = [c for c in chunks if c.page_content.strip()]
    if not chunks:
        return f"No valid text found in '{os.path.basename(file_obj.name)}'.", [], []

    # Add metadata BEFORE storing
    if suffix == ".html":
        title = extract_html_metadata(tmp_path)
        for chunk in chunks:
            chunk.metadata = {
                "doc_id": file_hash,
                "title": title,
                "file_type": "HTML Document"
            }
    else:
        for chunk in chunks:
            chunk.metadata = {
                "doc_id": file_hash,
                "file_type": suffix.upper().replace(".", "")
            }

    # Store in VectorDB
    vectordb.add_documents(chunks, ids=[f"{file_hash}_{i}" for i in range(len(chunks))])

    # Update log
    file_hash_map[file_hash] = os.path.basename(file_obj.name)
    with open(UPLOAD_LOG, "w") as f:
        json.dump(file_hash_map, f)

    return f"Stored {len(chunks)} chunks for '{os.path.basename(file_obj.name)}'.", [], []


def chat(query, history):
    # Check if any known filename is mentioned in the query
    selected_doc_id = CURRENT_DOC_ID.get("doc")
    lowered_query = query.lower()
 
    for doc_hash, filename in file_hash_map.items():
        if filename.lower() in lowered_query:
            selected_doc_id = doc_hash
            break
 
    result = graph.invoke({"input": query, "selected_doc": selected_doc_id})
    answer = result.get("answer", "I cannot find this information in the document.")
    history = history or []
    history.append((query, answer))
    return history, history
 
 
 
# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 📄 LangGraph RAG Chat with Per-Document Retrieval")
 
    file_upload = gr.File(label="Upload Document", file_types=list(LOADER_MAP.keys()))
    upload_status = gr.Textbox(label="Upload Status", interactive=False)
 
    chatbot = gr.Chatbot(label="Chat")
    msg = gr.Textbox(label="Ask something about this document")
    state = gr.State([])
 
    file_upload.upload(process_file_and_store, inputs=[file_upload], outputs=[upload_status, state, chatbot])
    msg.submit(chat, [msg, state], [chatbot, state]).then(lambda: "", None, [msg])
 
demo.launch()
 


