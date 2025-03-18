# app/langchain_chain.py
# Uses Groq AI for response generation instead of OpenAI.

from app.groq_inference import groq_infer  # Import Groq AI Inference function

def create_chain():
    """
    Instead of OpenAI, we use Groq AI Inference for answering queries.
    """
    def run(prompt):
        return groq_infer(prompt)

    return run  # Return function reference instead of an LLMChain object

def run_chain(chain, context: str, question: str) -> str:
    """
    Run the RAG chain with the given context and question.
    Uses Groq AI for inference.
    """
    prompt = f"Given the context: {context}\nAnswer the question: {question}"
    return chain(prompt)  # Calls groq_infer()
