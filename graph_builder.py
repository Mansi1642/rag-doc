from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os
import re

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# LangGraph state definition
class GraphState(dict):
    input: str
    documents: list
    answer: str

def retrieve_chunks(state: GraphState):
    query = state.get("input", "")

    if not os.path.exists("db") or not os.listdir("db"):
        return {"input": query, "documents": []}

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory="db", embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(query)
    return {"input": query, "documents": docs}

def generate_answer(state: GraphState):
    query = state["input"]
    docs = state["documents"]
    content = "\n\n".join([doc.page_content for doc in docs]) if docs else ""

    llm = ChatGroq(model="llama3-70b-8192", api_key=GROQ_API_KEY)

    if not content:
        return {"answer": "I cannot find this information in the document."}

    messages = [
        SystemMessage(content=(
            "You are a strict RAG assistant. "
            "ONLY answer using the provided documents. "
            "If the answer is not in the documents, say: "
            "'I cannot find this information in the document.'"
        )),
        HumanMessage(content=f"Documents:\n{content}\n\nQuestion:\n{query}")
    ]

    response = llm.invoke(messages).content
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    return {"answer": cleaned}

def build_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("Retrieve", retrieve_chunks)
    workflow.add_node("Generate", generate_answer)
    workflow.set_entry_point("Retrieve")
    workflow.add_edge("Retrieve", "Generate")
    workflow.add_edge("Generate", END)
    return workflow.compile()
