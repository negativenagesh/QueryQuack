from sentence_transformers import SentenceTransformer
from google.generativeai import GenerativeModel
import streamlit as st

def rewrite_query(query):
    """Rewrite ambiguous queries using Gemini LLM."""
    try:
        llm = GenerativeModel('gemini-1.5-pro')
        prompt = f"Rewrite the following query to be clear and precise:\nQuery: {query}\nRewritten Query:"
        response = llm.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.warning(f"Query rewriting failed: {str(e)}. Using original query.")
        return query

def process_query(query, rewrite=True):
    """Convert query to embedding, optionally rewriting it."""
    try:
        if rewrite:
            query = rewrite_query(query)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode([query], convert_to_numpy=True)[0]
        return query_embedding, query
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        return None, query