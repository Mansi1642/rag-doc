from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import SystemMessage
from langchain.schema.messages import HumanMessage
from dotenv import load_dotenv
import os
import re

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Shared instances
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory="db", embedding_function=embeddings)
llm = ChatGroq(model="deepseek-r1-distill-llama-70b", api_key=GROQ_API_KEY)

# LangGraph state definition
class GraphState(dict):
    input: str
    documents: list
    answer: str

# Node 1: Retrieve from vector DB
def retrieve_chunks(state: GraphState):
    query = state.get("input", "")
    retriever = vectordb.as_retriever()
    docs = retriever.get_relevant_documents(query)
    return {"input": query, "documents": docs}

# Node 2: Generate response from docs
def generate_answer(state: GraphState):
    query = state["input"]
    docs = state["documents"]
    content = "\n\n".join([doc.page_content for doc in docs])

    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions based on the provided documents."),
        HumanMessage(content=f"Documents:\n{content}\n\nQuestion:\n{query}")
    ]
    response = llm(messages).content
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    return {"answer": cleaned}


# Build LangGraph workflow
def build_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("Retrieve", retrieve_chunks)
    workflow.add_node("Generate", generate_answer)
    workflow.set_entry_point("Retrieve")
    workflow.add_edge("Retrieve", "Generate")
    workflow.add_edge("Generate", END)
    return workflow.compile()
