import streamlit as st
from graph_builder import build_graph
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import tempfile
import os

st.set_page_config(page_title="LangGraph RAG Agent")
st.title("📄🧠 LangGraph RAG with File Upload")

LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".md": UnstructuredMarkdownLoader,
}

def load_document(file_path, suffix):
    loader_cls = LOADER_MAP.get(suffix.lower())
    if loader_cls:
        return loader_cls(file_path).load()
    else:
        st.error(f"Unsupported file type: {suffix}")
        return []

uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt", "docx", "md"])

if uploaded_file:
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.success(f"Uploaded: {uploaded_file.name}")
    docs = load_document(tmp_path, suffix)

    if docs:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = Chroma.from_documents(chunks, embeddings, persist_directory="db")
        vectordb.persist()
        os.remove(tmp_path)
        st.success(f"Stored {len(chunks)} chunks in vector DB.")
    else:
        st.error("Failed to load document.")

graph = build_graph()
user_input = st.text_input("Ask a question about the uploaded document:")

if user_input:
    with st.spinner("Thinking..."):
        result = graph.invoke({"input": user_input})
        st.write("### Answer:")
        st.markdown(result["answer"])