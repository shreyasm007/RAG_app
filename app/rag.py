# app/rag.py
# Main RAG pipeline integrating vector storage, document retrieval, and response generation.

from .vector_store import init_vector_store, add_documents, search_documents
from .embeddings import get_embeddings
from .groq_inference import groq_infer  # Use Groq AI for response generation

def add_user_documents(docs):
    """
    Convert LangChain Document objects from user uploads into the format expected by the vector store
    and add them.
    """
    collection = init_vector_store()
    formatted_docs = []
    for i, doc in enumerate(docs):
        formatted_docs.append({
            "id": f"user_{i}",
            "text": doc.page_content,
            "metadata": {"source": "upload"}
        })
    add_documents(collection, formatted_docs)

def retrieve_context(collection, query: str):
    """
    Retrieve relevant documents for a query by generating its embedding and searching ChromaDB.
    """
    query_embedding = get_embeddings([query])[0]
    results = search_documents(collection, query_embedding)
    
    # Extract retrieved document text
    context = " ".join(results.get("documents", []))
    return context if context.strip() else "No relevant context found."

def run_rag(query: str, groq_api_key: str = None) -> str:
    """
    End-to-end RAG process:
    1. Retrieve relevant document context from ChromaDB.
    2. Use Groq AI to generate an answer based on retrieved context.
    """
    collection = init_vector_store()
    context = retrieve_context(collection, query)

    # Use Groq AI for final response generation
    prompt = f"Context: {context}\nQuestion: {query}"
    answer = groq_infer(prompt, api_key=groq_api_key)
    
    return answer
