# import pypdf
# from sentence_transformers import SentenceTransformer
# import chromadb
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import docx
# import os


# # def read_pdf(file_path):
# #     """Reads all pages from the PDF and returns combined text."""
# #     reader = pypdf.PdfReader(file_path)
# #     all_text = ""
# #     for i, page in enumerate(reader.pages):
# #         text = page.extract_text()
# #         if text:
# #             all_text += text + "\n"
# #         else:
# #             print(f"No text found on page {i + 1}")

# #     return all_text.strip()

# def read_pdf(file_path):
#     reader=pypdf.PdfReader(file_path)
#     return "\n".join([page.extract_text() or "" for page in reader.pages])

# def read_docx(file_path):
#     doc=docx.Document(file_path)
#     return "\n".join([para.text for para in doc.paragraphs])

# def read_txt(file_path):
#     with open(file_path, 'r', encoding='utf-8') as f:
#         return f.read()

# def read_file(file_path):
#     ext = os.path.splitext(file_path)[-1].lower()
#     if ext == ".pdf":
#         return read_pdf(file_path)
#     elif ext == ".docx":
#         return read_docx(file_path)
#     elif ext == ".txt":
#         return read_txt(file_path)
#     else:
#         print(f"Unsupported file type: {ext}")

# def store_data(text):
    
#     model= SentenceTransformer('multi-qa-mpnet-base-dot-v1')

#     text_splitter=RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=50,separators=["\n\n", "\n", ".", " "])
#     chunks=text_splitter.split_text(text)
#     # print(chunks)
#     embeddings=model.encode(chunks).tolist()
#     client=chromadb.PersistentClient(path="./chroma_db")
#     client.delete_collection("my_documents")
#     collection=client.get_or_create_collection(name="my_documents") 
#     ids = [f"document-{i}" for i in range(len(chunks))]
#     collection.add(documents=chunks,embeddings=embeddings,ids=ids)
#     print("Data Stored....")

# def search_ans(query,k=3):
#     model=SentenceTransformer("multi-qa-mpnet-base-dot-v1")
#     query_embeddings=model.encode([query]).tolist()

#     client=chromadb.PersistentClient(path="./chroma_db")
#     collection=client.get_or_create_collection("my_documents")

#     results=collection.query(query_embeddings=query_embeddings,n_results=k)
#     return results['documents'][0]


# file_path="microsoft_report.txt"
# text=read_file(file_path)
# if text:
#     store_data(text)

#     question="What is the annual revenue of microsoft??"
#     answers=search_ans(question)

#     print(f"answer is: " )
#     for i,ans in enumerate(answers,1):
#         print(f"\nAnswer: \n{ans}")

import os
import chromadb
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

def read_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return PyPDFLoader(file_path).load()
    if ext == ".docx":
        return Docx2txtLoader(file_path).load()
    if ext == ".txt":
        return TextLoader(file_path, encoding="utf-8").load()
    raise ValueError(f"Unsupported file type: {ext}")

def store_data(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100, separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)
    embeddings = SentenceTransformerEmbeddings(model_name="multi-qa-mpnet-base-dot-v1")
    vectordb = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="./chroma_db",
        collection_name="my_documents"
    )
    vectordb.persist()
    return vectordb

def build_qa_chain(vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    llm = ChatGroq(model_name="llama3-70b-8192",api_key=GROQ_API_KEY, temperature=0)
    return RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",retriever=retriever)















































































