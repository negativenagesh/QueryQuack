from pinecone import Pinecone
import streamlit as st
import os
from uuid import uuid4
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def initialize_pinecone():
    """Initialize Pinecone connection with queryquack index."""
    try:
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            st.error("PINECONE_API_KEY not found in environment variables")
            return None
        
        environment = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
        pc = Pinecone(api_key=api_key, environment=environment)
        
        # Get or create index
        index_name = "queryquack"
        
        # List available indexes
        indexes = pc.list_indexes()
        
        if not index_name in [index.name for index in indexes]:
            # Create a new index
            pc.create_index(
                name=index_name,
                dimension=384,  # Dimension for all-MiniLM-L6-v2
                metric="cosine"
            )
            st.success(f"Created new Pinecone index: {index_name}")
        
        # Get index
        index = pc.Index(index_name)
        return index
    
    except Exception as e:
        st.error(f"Error initializing Pinecone: {str(e)}")
        return None

def store_embeddings(embeddings, chunk_metadata, namespace='default'):
    """Store embeddings in Pinecone with metadata."""
    index = initialize_pinecone()
    if index is None:
        return False
    
    try:
        # Prepare vectors for upsert
        vectors = []
        for i, (emb, meta) in enumerate(zip(embeddings, chunk_metadata)):
            # Create a unique ID for each vector
            vector_id = f"{meta['filename']}_{meta['chunk_index']}_{uuid4()}"
            
            # Ensure embeddings are in list format (handle both numpy arrays and lists)
            if hasattr(emb, 'tolist'):
                emb_list = emb.tolist()
            else:
                emb_list = emb  # Already a list
                
            # Create vector record
            vector = {
                "id": vector_id,
                "values": emb_list,
                "metadata": meta
            }
            vectors.append(vector)
        
        # Batch upsert embeddings
        index.upsert(vectors=vectors, namespace=namespace)
        st.success(f"Successfully stored {len(vectors)} embeddings in Pinecone")
        return True
    except Exception as e:
        st.error(f"Error storing embeddings: {str(e)}")
        return False