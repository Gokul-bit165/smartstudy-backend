# backend/utils/rag_pipeline.py
import os
from groq import Groq
from typing import List, Generator

# --- Initialize the Groq Client ---
# It will automatically find the GROQ_API_KEY in your Render environment variables
try:
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
except Exception as e:
    print(f"Error initializing Groq client: {e}")
    groq_client = None

def generate_stream(context: List[str], query: str, llm_model: str = "llama3-8b-8192") -> Generator:
    """Generates a streaming response for the chat using the Groq API."""
    if not groq_client:
        yield "Error: Groq API key is not configured or the client failed to initialize."
        return

    context_str = "\n".join(context)
    
    prompt = f"""
    Use the following context to answer the question.
    If the context doesn't contain the answer, say that you couldn't find the relevant information.
    
    Context:
    {context_str}
    
    Question: {query}
    """

    try:
        stream = groq_client.chat.completions.create(
            model=llm_model,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.7,
            stream=True,
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    except Exception as e:
        print(f"An error occurred while connecting to Groq: {e}")
        yield "Error: Could not connect to the LLM service. Please check the server logs."


def generate_non_stream_answer(prompt: str, llm_model: str = "llama3-8b-8192") -> str:
    """Generates a complete, non-streaming response using the Groq API."""
    if not groq_client:
        return "Error: Groq API key is not configured."

    try:
        chat_completion = groq_client.chat.completions.create(
            model=llm_model,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.7,
            stream=False,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"An error occurred while connecting to Groq: {e}")
        return "Error: Could not get a response from the LLM service."