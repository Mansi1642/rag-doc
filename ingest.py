# # #ingest.py

# # from langchain.document_loaders import DirectoryLoader, PyPDFLoader
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # from langchain.embeddings import OpenAIEmbeddings
# # from langchain.vectorstores import Chroma
# # import os

# # def ingest_documents():
# #     loader = DirectoryLoader("docs", glob="**/*.pdf", loader_cls=PyPDFLoader)
# #     docs = loader.load()

# #     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
# #     chunks = splitter.split_documents(docs)

# #     embeddings = OpenAIEmbeddings()
# #     vectordb = Chroma.from_documents(chunks, embeddings, persist_directory="db")
# #     vectordb.persist()
# #     print(f"✅ Embedded and stored {len(chunks)} chunks.")

# # if __name__ == "__main__":
# #     ingest_documents()

# from langchain.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import Chroma
# import sys
# import os

# def ingest_single_pdf(file_path: str):
#     if not os.path.exists(file_path):
#         print(f"❌ File does not exist: {file_path}")
#         return

#     print(f"📄 Loading file: {file_path}")
#     loader = PyPDFLoader(file_path)
#     docs = loader.load()

#     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#     chunks = splitter.split_documents(docs)
#     print(f"✂️ Split into {len(chunks)} chunks.")

#     print("🔐 Using OpenAIEmbeddings (make sure OPENAI_API_KEY is set)...")
#     embeddings = OpenAIEmbeddings()

#     vectordb = Chroma.from_documents(chunks, embeddings, persist_directory="db")
#     vectordb.persist()
#     print(f"✅ Embedded and stored {len(chunks)} chunks to vector DB.")

# if __name__ == "__main__":
#     if len(sys.argv) != 2:

#         print("Usage: python ingest.py D:\OneDrive - Aligned Automation Services Private Limited\Desktop\RAG-AA\rag-doc\AI+Agents+-+Build+AI+Agents.pdf")
#     else:
#         ingest_single_pdf(sys.argv[1])
