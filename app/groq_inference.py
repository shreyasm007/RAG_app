# app/groq_inference.py
# This module provides a function for Groq AI Inference.
# It loads API keys from the .env file and allows an override via function parameters.

import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Use the GROQ_API_URL from .env (with a default if not provided)
GROQ_API_URL = os.getenv("GROQ_API_URL", "https://api.groq.ai/inference")

def groq_infer(prompt: str, api_key: str = None) -> str:
    """
    Send a prompt to the Groq AI Inference API and return the generated text.
    If api_key is not provided, it is loaded from the environment variables.
    """
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "No API key provided for Groq AI Inference."
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "prompt": prompt,
        # Add additional parameters as required by Groq's API
    }
    response = requests.post(GROQ_API_URL, json=payload, headers=headers)
    if response.status_code == 200:
        result = response.json()
        return result.get("generated_text", "")
    else:
        return f"Error: {response.status_code} - {response.text}"
