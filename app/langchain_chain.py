# app/langchain_chain.py
# This module creates a simple LangChain chain.
# For demonstration, we use OpenAI's LLM (adjust as needed).
# You can swap this with a chain that calls groq_infer if preferred.

from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

def create_chain():
    """
    Creates an LLM chain that uses a prompt template.
    Here, OpenAI's LLM is used for text generation.
    Ensure your environment is configured with the necessary API key.
    """
    llm = OpenAI(temperature=0.7)  # Set up your LLM; alternatively, integrate groq_infer
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="Given the context: {context}\nAnswer the question: {question}"
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    return chain

def run_chain(chain, context: str, question: str) -> str:
    """
    Run the chain with the provided context and question.
    """
    return chain.run({"context": context, "question": question})
