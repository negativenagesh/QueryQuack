import pinecone
import streamlit as st
import os
from uuid import uuid4

def initialize_pinecone():
    """Initialize Pinecone connection."""
    try:
        pinecone.init(
            api_key=os.getenv('PINECONE_API_KEY', 'YOUR_PINECONE_API_KEY'),
            environment='us-west1-gcp'
        )
        return pinecone.Index('pdf-embeddings')
    except Exception as e:
        st.error(f"Failed to initialize Pinecone: {str(e)}")
        return None

def store_embeddings(embeddings, chunk_metadata, namespace='default'):
    """Store embeddings in Pinecone with metadata."""
    index = initialize_pinecone()
    if index is None:
        return False
    
    try:
        # Batch upsert embeddings
        vectors = [
            (f"{meta['filename']}_{meta['chunk_index']}_{uuid4()}", emb.tolist(), meta)
            for emb, meta in zip(embeddings, chunk_metadata)
        ]
        index.upsert(vectors=vectors, namespace=namespace)
        return True
    except Exception as e:
        st.error(f"Error storing embeddings: {str(e)}")
        return False