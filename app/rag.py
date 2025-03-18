# app/rag.py
# This is the main RAG module that ties together the vector store,
# embeddings, LangChain chain, and Groq AI Inference.

from .vector_store import init_vector_store, add_documents, search_documents
from .embeddings import get_embeddings
from .langchain_chain import create_chain, run_chain
from .groq_inference import groq_infer

def setup_documents():
    """
    For demonstration purposes, add some sample documents to the vector store.
    In a real project, you would pre-index your own documents.
    """
    collection = init_vector_store()
    documents = [
        {"id": "1", "text": "The capital of France is Paris.", "metadata": {"source": "wiki"}},
        {"id": "2", "text": "Python is a programming language.", "metadata": {"source": "wiki"}},
        {"id": "3", "text": "The sky is blue on a clear day.", "metadata": {"source": "observation"}},
    ]
    add_documents(collection, documents)
    return collection

def retrieve_context(collection, query: str):
    """
    Retrieve context for a query by generating its embedding and searching the vector store.
    """
    query_embedding = get_embeddings([query])[0]
    results = search_documents(collection, query_embedding)
    # Assuming the results dict contains a key "documents" with the texts.
    context = " ".join(results.get("documents", []))
    return context

def run_rag(query: str, use_groq: bool = False, groq_api_key: str = None) -> str:
    """
    End-to-end function to run the RAG pipeline:
      1. Initialize vector store and index documents.
      2. Retrieve context based on the query.
      3. Generate an answer using either:
         - A LangChain chain (default)
         - Groq AI Inference if `use_groq` is True.
    
    The Groq API key can be provided via the function parameter (overriding the .env setting).
    """
    collection = init_vector_store()
    # Setup documents if not already indexed; in production, index your documents once.
    setup_documents()
    context = retrieve_context(collection, query)
    if context.strip() == "":
        context = "No relevant context found."
    
    if use_groq:
        prompt = f"Context: {context}\nQuestion: {query}"
        answer = groq_infer(prompt, api_key=groq_api_key)
    else:
        chain = create_chain()
        answer = run_chain(chain, context, query)
    
    return answer
