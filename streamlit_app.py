# streamlit_app.py
import streamlit as st
from app.rag import run_rag, add_user_documents
from app.document_loader import load_document

st.title("RAG Chatbot with Document Upload")

# File uploader
uploaded_file = st.file_uploader("Upload your document (PDF, TXT, DOCX)", type=["pdf", "txt", "docx"])
if uploaded_file is not None:
    try:
        documents = load_document(uploaded_file, uploaded_file.name)
        add_user_documents(documents)
        st.success("Document uploaded and indexed!")
    except Exception as e:
        st.error(f"Error processing document: {e}")

query = st.text_input("Enter your question:")

use_groq = st.checkbox("Use Groq AI Inference")
groq_api_key = None
if use_groq:
    groq_api_key = st.text_input("Enter your Groq API key (optional, leave blank to use .env)", type="password")

if st.button("Get Answer"):
    with st.spinner("Generating answer..."):
        answer = run_rag(query, use_groq=use_groq, groq_api_key=groq_api_key)
    st.write("Answer:")
    st.write(answer)
