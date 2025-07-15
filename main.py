import gradio as gr
from graph_builder import build_graph
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

# File loader mapping
LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".md": UnstructuredMarkdownLoader,
}

graph = build_graph()

def process_file_and_store(file_obj):
    suffix = os.path.splitext(file_obj.name)[1]
    loader_cls = LOADER_MAP.get(suffix.lower())

    if not loader_cls:
        return f"Unsupported file type: {suffix}", None

    tmp_path = file_obj.name

    docs = loader_cls(tmp_path).load()
    if not docs:
        return "Failed to load document.", None

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory="db")
    vectordb.persist()

    return f"Stored {len(chunks)} chunks in vector DB.", True

def chat(query, history):
    if not query.strip():
        history.append({"role": "assistant", "content": "Please ask a question."})
        return history, history

    result = graph.invoke({"input": query})
    answer = result["answer"]

    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": answer})
    return history, history


with gr.Blocks() as demo:
    gr.Markdown("## 📄🧠 LangGraph RAG Conversational Chat")

    file_upload = gr.File(label="Upload a document", file_types=[".pdf", ".txt", ".docx", ".md"])
    upload_status = gr.Textbox(label="Upload Status", interactive=False)

    chatbot = gr.Chatbot(label="Conversation", type="messages")
    msg = gr.Textbox(label="Ask a question about the uploaded document:")

    state = gr.State([])

    file_upload.upload(process_file_and_store, inputs=[file_upload], outputs=[upload_status])
    msg.submit(chat, [msg, state], [chatbot, state]).then(
        lambda: "", None, [msg]
    )

if __name__ == "__main__":
    demo.launch()
