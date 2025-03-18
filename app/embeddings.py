# app/embeddings.py
# This module handles generating embeddings using Hugging Face.
# It uses LangChain's integration with Hugging Face embeddings.

from langchain.embeddings import HuggingFaceEmbeddings

def get_embeddings(texts):
    """
    Given a list of texts, return their embeddings.
    Uses the "sentence-transformers/all-MiniLM-L6-v2" model.
    """
    hf = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return hf.embed_documents(texts)
