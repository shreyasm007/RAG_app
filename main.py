# main.py
# Entry point for the command-line interface.

from app.rag import run_rag

def main():
    query = input("Enter your question: ")
    
    use_groq_input = input("Do you want to use Groq AI Inference? (y/n): ")
    use_groq = use_groq_input.lower().strip() == 'y'
    
    groq_api_key = None
    if use_groq:
        groq_api_key = input("Enter your Groq API key (leave blank to use .env): ").strip() or None

    answer = run_rag(query, use_groq=use_groq, groq_api_key=groq_api_key)
    print("Answer:", answer)

if __name__ == "__main__":
    main()
