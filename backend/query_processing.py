import streamlit as st
import re
from langchain.prompts import PromptTemplate
from backend.model_utils import ensure_model_exists

custom_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

def rewrite_query(query):
    """Optionally rewrite the query to improve retrieval."""
    query = query.strip()
    
    query = re.sub(r'\?+$', '', query)
    
    fillers = [
        r'^(please\s+)?tell\s+me\s+about\s+',
        r'^(please\s+)?explain\s+',
        r'^what\s+is\s+',
        r'^how\s+to\s+'
    ]
    
    for filler in fillers:
        query = re.sub(filler, '', query, flags=re.IGNORECASE)
    
    return query.strip()

def process_query(query, rewrite=True):
    """
    Process the query for improved retrieval.
    
    Args:
        query: User query text
        rewrite: Whether to rewrite the query
        
    Returns:
        query_embedding: Query embedding vector
        processed_query: Processed query text
        original_query: Original query text
    """
    try:
        original_query = query
        
        if 'original_query' not in st.session_state:
            st.session_state['original_query'] = original_query
            
        if rewrite:
            processed_query = rewrite_query(query)
        else:
            processed_query = query
            
        model_path = ensure_model_exists("all-MiniLM-L6-v2")
        if not model_path:
            st.error("Failed to load embedding model")
            return None, processed_query, original_query
            
        from langchain_community.embeddings import HuggingFaceEmbeddings
        
        embeddings_model = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={'device': 'cpu'}
        )
        
        query_embedding = embeddings_model.embed_query(processed_query)
        
        return query_embedding, processed_query, original_query
        
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        return None, query, original_query