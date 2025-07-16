# from langgraph.graph import StateGraph, END
# from langchain_groq import ChatGroq
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.schema import SystemMessage
# from langchain.schema.messages import HumanMessage
# from dotenv import load_dotenv
# import os
# import re

# # Load environment variables
# load_dotenv()
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# # Shared instances
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# vectordb = Chroma(persist_directory="db", embedding_function=embeddings)
# llm = ChatGroq(model="deepseek-r1-distill-llama-70b", api_key=GROQ_API_KEY)

# # LangGraph state definition
# class GraphState(dict):
#     input: str
#     documents: list
#     answer: str

# # Node 1: Retrieve from vector DB
# def retrieve_chunks(state: GraphState):
#     query = state.get("input", "")
#     retriever = vectordb.as_retriever()
#     docs = retriever.invoke(query)
#     return {"input": query, "documents": docs}

# # Node 2: Generate response from docs
# def generate_answer(state: GraphState):
#     query = state["input"]
#     docs = state["documents"]
#     content = "\n\n".join([doc.page_content for doc in docs])

#     messages = [
#         SystemMessage(content="You are a helpful assistant that answers questions based on the provided documents."),
#         HumanMessage(content=f"Documents:\n{content}\n\nQuestion:\n{query}")
#     ]
#     response = llm.invoke(messages).content
#     cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
#     return {"answer": cleaned}


# # Build LangGraph workflow
# def build_graph():
#     workflow = StateGraph(GraphState)
#     workflow.add_node("Retrieve", retrieve_chunks)
#     workflow.add_node("Generate", generate_answer)
#     workflow.set_entry_point("Retrieve")
#     workflow.add_edge("Retrieve", "Generate")
#     workflow.add_edge("Generate", END)
#     return workflow.compile()


# from langgraph.graph import StateGraph, END
# from langchain_groq import ChatGroq
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.schema import SystemMessage
# from langchain.schema.messages import HumanMessage
# from langchain.agents import initialize_agent, AgentType
# from langchain.tools import Tool
# from dotenv import load_dotenv
# import os
# import re

# # Load environment variables
# load_dotenv()
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# # Shared instances
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# vectordb = Chroma(persist_directory="db", embedding_function=embeddings)
# llm = ChatGroq(model="deepseek-r1-distill-llama-70b", api_key=GROQ_API_KEY)

# # LangGraph state definition
# class GraphState(dict):
#     input: str
#     documents: list
#     answer: str

# # Node 1: Retrieve from vector DB
# def retrieve_chunks(state: GraphState):
#     query = state.get("input", "")
#     retriever = vectordb.as_retriever()
#     docs = retriever.invoke(query)
#     return {"input": query, "documents": docs}

# # Node 2: Agent reasoning over documents
# def agent_node(state: GraphState):
#     query = state["input"]
#     docs = state["documents"]

#     # Prepare a single combined document text for the agent tool
#     docs_content = "\n\n".join([doc.page_content for doc in docs]) if docs else "No relevant documents found."

#     # Define a tool that the agent can use to "search" documents
#     tools = [
#         Tool(
#             name="Document Search",
#             func=lambda q: docs_content,
#             description="Provides the retrieved document content relevant to the user's query."
#         )
#     ]

#     # Initialize an agent with zero-shot reasoning over tools
#     agent = initialize_agent(
#         tools=tools,
#         llm=llm,
#         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#         verbose=False,
#         handle_parsing_errors=True
#     )

#     # Run the agent to generate an answer
#     response = agent.run(query)

#     # Clean unwanted <think> tags if Groq adds them
#     cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
#     return {"answer": cleaned}

# # Build LangGraph workflow
# def build_graph():
#     workflow = StateGraph(GraphState)
#     workflow.add_node("Retrieve", retrieve_chunks)
#     workflow.add_node("Agent", agent_node)
#     workflow.set_entry_point("Retrieve")
#     workflow.add_edge("Retrieve", "Agent")
#     workflow.add_edge("Agent", END)
#     return workflow.compile()


from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.tools import Tool
from dotenv import load_dotenv
import os
import re

from langgraph.prebuilt import create_react_agent
from langchain.schema import HumanMessage

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
    answer: str

# ---- Tools ----

# Tool 1: Detect Document Type
def detect_document_type(query: str) -> str:
    # ✅ Read stored file name (for title queries)
    file_name = "unknown"
    if os.path.exists("current_file.txt"):
        with open("current_file.txt", "r", encoding="utf-8") as f:
            file_name = f.read().strip()

    # ✅ Retrieve sample document content
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke("document content")
    sample_text = "\n".join([d.page_content for d in docs]) if docs else "No document content found."

    # ✅ Ask LLM to classify based on real content
    prompt = (
        f"You are a document classification expert.\n\n"
        f"Document File Name: {file_name}\n"
        f"Document Content (sample):\n{sample_text}\n\n"
        f"Classify the document type (invoice, contract, resume, research paper, email, report, etc.).\n"
        f"Answer strictly in this format:\n"
        f"The document is a [TYPE]. Answer: [RESULT]."
    )

    response = llm.invoke(prompt).content
    return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

def get_document_title(_: str) -> str:
    if os.path.exists("current_file.txt"):
        with open("current_file.txt", "r", encoding="utf-8") as f:
            file_name = f.read().strip()
        return f"The document title is '{file_name}'."
    return "No document has been uploaded yet."


# Tool 2: Vector DB Search
def vector_db_search(query: str) -> str:
    retriever = vectordb.as_retriever()
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant information found in the document."
    return "\n\n".join([doc.page_content for doc in docs])

# ---- Multi-Step Agent Node ----
def agent_node(state: GraphState):
    query = state["input"]

    tools = [
        Tool(
            name="Detect Document Type",
            func=detect_document_type,
            description="Use this when the user asks about the type or classification of the uploaded document."
        ),
        Tool(
            name="Vector DB Search",
            func=vector_db_search,
            description="Use this when the user asks questions about the document content."
        ),
        Tool(
            name="Get Document Title",
            func=get_document_title,
            description="Use this when the user asks for the file name or title of the document."
        )
    ]

    # ✅ Let LangGraph handle tool reasoning automatically (no forced prompt)
    agent = create_react_agent(llm, tools)

    # ✅ Directly pass the user query as a normal conversation
    result = agent.invoke({"messages": [HumanMessage(content=query)]})

    # ✅ Clean output
    final_answer = result["messages"][-1].content
    cleaned = re.sub(r"<think>.*?</think>", "", final_answer, flags=re.DOTALL).strip()

    return {"answer": cleaned}

# ---- Build LangGraph workflow ----
def build_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("Agent", agent_node)
    workflow.set_entry_point("Agent")
    workflow.add_edge("Agent", END)
    return workflow.compile()
