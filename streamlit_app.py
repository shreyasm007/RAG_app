# streamlit_app.py
# Streamlit front-end for interacting with the RAG pipeline.

import streamlit as st
from app.rag import run_rag

st.title("RAG Project with LangChain, ChromaDB, Groq AI Inference, and Hugging Face Embeddings")

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
