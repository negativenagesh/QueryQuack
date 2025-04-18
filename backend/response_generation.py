from google.generativeai import GenerativeModel
import streamlit as st

def generate_response(query, chunks):
    """Generate a response using Gemini LLM with source attribution."""
    try:
        llm = GenerativeModel('gemini-1.5-pro')
        context = "\n".join([f"{pdf} (Chunk {chunk}): {text}" for pdf, chunk, text in chunks])
        prompt = (
            f"Query: {query}\n\n"
            f"Context:\n{context}\n\n"
            "Answer concisely, citing the source PDF and chunk number. "
            "If no relevant information is found, say 'No relevant information found.'"
        )
        response = llm.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "No relevant information found."