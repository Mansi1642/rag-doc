import os
import tempfile
import streamlit as st
from rag_processing import read_file,store_data,build_qa_chain

st.set_page_config("RAG Document Chat", layout="wide")
st.title("📄 Chat With Your Document")

uploaded = st.file_uploader("Upload PDF, DOCX or TXT", type=["pdf", "docx", "txt"])
if uploaded:
    suffix=os.path.splitext(uploaded.name)[1]
    with tempfile.NamedTemporaryFile(delete=False,suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        tmp.flush()
        temp_path=tmp.name

    docs = read_file(temp_path)
    os.remove(temp_path)


    vectordb =store_data(docs)
    qa = build_qa_chain(vectordb)

    query = st.chat_input("Ask your question…")
    if query:
        st.chat_message("user").write(query)
        answer = qa.run(query)
        st.chat_message("assistant").write(answer)
















































