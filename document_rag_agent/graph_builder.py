import os
import json
from typing import TypedDict
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, START, END
from langchain.memory import ConversationBufferMemory
from langgraph.checkpoint.memory import InMemorySaver
from rag_processing import read_file, store_data, build_qa_chain, check_similar_question

HISTORY_FILE = "conversation_history.json"

# Load/save
def save_history_to_file(history_dict, filename=HISTORY_FILE):
    print("History saver is called..")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(history_dict, f, indent=2, ensure_ascii=False)

def load_history_from_file(filename=HISTORY_FILE):
    if not os.path.exists(filename):
        return {}
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

# State
class State(TypedDict):
    query: str
    answer: str

class ConfigSchema(TypedDict):
    thread_id: str

# Globals
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
conversation_history = load_history_from_file()
vdb = None

def init_vectordb(file_path: str):
    global vdb
    if vdb is None:
        docs = read_file(file_path)
        vdb = store_data(docs)

def qa_node(state: State, config) -> dict:
    init_vectordb(os.getenv("RAG_DOC_PATH", ""))  # or pass directly
    qa = build_qa_chain(vdb)
    thread_id = config["configurable"]["thread_id"]

    global conversation_history
    if thread_id not in conversation_history:
        conversation_history[thread_id] = []

    conversation_history[thread_id].append({"role": "user", "content": state["query"]})

    chat_history = memory.load_memory_variables({"configurable": {"thread_id": thread_id}}).get("chat_history", [])
    answer = qa.invoke({"query": state["query"], "chat_history": chat_history})

    memory.save_context({"input": state["query"]}, {"output": answer["result"]})
    conversation_history[thread_id].append({"role": "assistant", "content": answer["result"]})
    save_history_to_file(conversation_history)

    return {"answer": answer["result"]}

def build_graph():
    graph = StateGraph(State)
    graph.add_node("qa", qa_node)
    graph.add_edge(START, "qa")
    graph.add_edge("qa", END)
    checkpointer = InMemorySaver()
    return graph.compile(checkpointer=checkpointer)

def run_question(question: str, file_path: str, thread_id: str):
    history = load_history_from_file()

    if thread_id not in history:
        history[thread_id] = []

    for entry in history[thread_id]:
        if entry.get("role") == "user" and entry.get("content") == question:
            print("Answer from cache:")
            for e in history[thread_id]:
                if e.get("role") == "assistant":
                    return e.get("content")
    embedding_model = HuggingFaceEmbeddings(model_name="multi-qa-mpnet-base-dot-v1")
    similar_answer = check_similar_question(question, embedding_model)
    if similar_answer:
        print("Answer from cache (VectorDB):")
        return similar_answer

    init_vectordb(file_path)
    app = build_graph()
    out = app.invoke({"query": question, "answer": ""}, config={"configurable": {"thread_id": thread_id}})
    answer = out["answer"]

    history[thread_id].append({"role": "user", "content": question})
    history[thread_id].append({"role": "assistant", "content": answer})
    save_history_to_file(history)

    return answer
