from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os

def ingest_documents():
    loader = DirectoryLoader("docs", glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory="db")
    vectordb.persist()
    print(f"✅ Embedded and stored {len(chunks)} chunks.")

if __name__ == "__main__":
    ingest_documents()