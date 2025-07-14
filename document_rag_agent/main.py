import pypdf
from sentence_transformers import SentenceTransformer
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter

def read_pdf(file_path):
    """Reads all pages from the PDF and returns combined text."""
    reader = pypdf.PdfReader(file_path)
    all_text = ""
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            all_text += text + "\n"
        else:
            print(f"No text found on page {i + 1}")

    return all_text.strip()

def store_data(text):
    
    model= SentenceTransformer('multi-qa-mpnet-base-dot-v1')

    text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_overlap=50,separators=["\n\n", "\n", ".", " "])
    chunks=text_splitter.split_text(text)
    # print(chunks)
    embeddings=model.encode(chunks).tolist()
    client=chromadb.PersistentClient(path="./chroma_db")
    client.delete_collection("my_documents")
    collection=client.get_or_create_collection(name="my_documents") 
    ids = [f"document-{i}" for i in range(len(chunks))]
    collection.add(documents=chunks,embeddings=embeddings,ids=ids)
    print("Data Stored....")

def search_ans(query,k=3):
    model=SentenceTransformer("multi-qa-mpnet-base-dot-v1")
    query_embeddings=model.encode([query]).tolist()

    client=chromadb.PersistentClient(path="./chroma_db")
    collection=client.get_or_create_collection("my_documents")

    results=collection.query(query_embeddings=query_embeddings,n_results=k)
    return results['documents'][0]


file_path="2024_Annual_Report.pdf"
text=read_pdf(file_path)
if text:
    store_data(text)

    question="Which was pivotal year for microsoft??"
    answers=search_ans(question)

    print(f"answer is: " )
    for i,ans in enumerate(answers,1):
        print(f"\nAnswer: \n{ans}")















































































