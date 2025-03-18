# app/vector_store.py
# This module sets up and interacts with a ChromaDB vector store.

import chromadb
from chromadb.config import Settings

def init_vector_store(collection_name="documents"):
    """
    Initialize or retrieve a ChromaDB collection.
    The persistence directory is set to './chroma_db'.
    """
    client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="./chroma_db"
    ))
    collection = client.get_or_create_collection(name=collection_name)
    return collection

def add_documents(collection, documents):
    """
    Add documents to the vector store.
    Each document is expected to be a dict with keys: "id", "text", "metadata".
    """
    ids = [doc["id"] for doc in documents]
    texts = [doc["text"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]
    collection.add(documents=texts, metadatas=metadatas, ids=ids)

def search_documents(collection, query_embedding, n_results=5):
    """
    Search the vector store for documents similar to the provided query embedding.
    """
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    return results
