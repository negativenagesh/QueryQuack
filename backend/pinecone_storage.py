import streamlit as st
import uuid
import time
import os
import numpy as np
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from backend.model_utils import ensure_model_exists

# Load environment variables from .env file
load_dotenv()

def initialize_pinecone():
    """Initialize and return Pinecone index."""
    # Get API key from environment variables (loaded from .env)
    api_key = os.environ.get("PINECONE_API_KEY")
    index_name = os.environ.get("PINECONE_INDEX", "queryquack")
    
    # If still no API key, show error message
    if not api_key:
        st.error("Pinecone API key not found. Please set the PINECONE_API_KEY in your .env file.")
        return None
    
    try:
        # Initialize Pinecone with the new API
        pc = Pinecone(api_key=api_key)
        
        # Check if index exists
        existing_indexes = pc.list_indexes().names()
        if index_name not in existing_indexes:
            # Create index if it doesn't exist (serverless)
            pc.create_index(
                name=index_name,
                dimension=384,  # Dimension for all-MiniLM-L6-v2
                metric="cosine"
            )
            st.info(f"Created new Pinecone index: {index_name}")
        
        # Connect to index using the new API
        index = pc.Index(index_name)
        st.success("Successfully connected to Pinecone")
        return index
    except Exception as e:
        st.error(f"Failed to connect to Pinecone: {str(e)}")
        return None

def store_embeddings(embeddings: List, metadata_list: List[Dict], namespace: str = "default", batch_size: int = 50):
    """
    Store embeddings in Pinecone with batching to avoid size limits.
    
    Args:
        embeddings: List of embedding vectors
        metadata_list: List of metadata dictionaries
        namespace: Pinecone namespace
        batch_size: Number of vectors per batch to stay under 2MB limit
    
    Returns:
        bool: Success status
    """
    index = initialize_pinecone()
    if not index:
        return False
    
    # Fix: Properly check if embeddings or metadata_list is empty
    is_empty_embeddings = False
    is_empty_metadata = False
    
    # Check embeddings - handle NumPy arrays and regular lists differently
    if isinstance(embeddings, np.ndarray):
        is_empty_embeddings = embeddings.size == 0
    else:
        is_empty_embeddings = len(embeddings) == 0 if embeddings is not None else True
    
    # Check metadata list
    is_empty_metadata = len(metadata_list) == 0 if metadata_list is not None else True
    
    if is_empty_embeddings or is_empty_metadata:
        st.warning("No embeddings to store")
        return False
    
    total_vectors = len(embeddings) if not isinstance(embeddings, np.ndarray) else embeddings.shape[0]
    
    try:
        # Process in batches to stay under Pinecone's 2MB limit
        for i in range(0, total_vectors, batch_size):
            batch_end = min(i + batch_size, total_vectors)
            
            # Create batch of vectors to upsert
            vectors_batch = []
            for j in range(i, batch_end):
                # Create a unique ID for each embedding
                vector_id = str(uuid.uuid4())
                # Include original chunk index in metadata
                metadata = metadata_list[j].copy()
                metadata["chunk_index"] = j
                
                # Get the embedding vector - handle NumPy arrays
                if isinstance(embeddings, np.ndarray):
                    vector = embeddings[j].tolist()
                else:
                    vector = embeddings[j]
                
                vectors_batch.append({
                    "id": vector_id,
                    "values": vector,
                    "metadata": metadata
                })
            
            # Upsert the batch to Pinecone
            index.upsert(vectors=vectors_batch, namespace=namespace)
            
            # Add a small delay between batches to avoid rate limiting
            if batch_end < total_vectors:
                time.sleep(0.5)
            
            # Update progress if in a Streamlit app
            if i == 0:
                progress_bar = st.progress(0)
            progress = min(1.0, batch_end / total_vectors)
            if 'progress_bar' in locals():
                progress_bar.progress(progress)
        
        # Clear progress bar
        if 'progress_bar' in locals():
            progress_bar.empty()
            
        st.success(f"Successfully stored {total_vectors} embeddings in Pinecone")
        return True
        
    except Exception as e:
        st.error(f"Error storing embeddings: {str(e)}")
        return False

def get_langchain_retriever(namespace="default"):
    """
    Create a LangChain retriever from Pinecone.
    
    Args:
        namespace: Pinecone namespace
        
    Returns:
        retriever: LangChain retriever
    """
    try:
        # Get API key from environment variables
        api_key = os.environ.get("PINECONE_API_KEY")
        index_name = os.environ.get("PINECONE_INDEX", "queryquack")
        
        if not api_key:
            st.error("Pinecone API key not found")
            return None
        
        # Ensure embedding model exists
        model_path = ensure_model_exists("all-MiniLM-L6-v2")
        if not model_path:
            st.error("Failed to load embedding model")
            return None
            
        # Create embeddings model
        embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize Pinecone correctly for LangChain compatibility
        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)
        
        # Create LangChain Pinecone vectorstore - updated for compatibility
        vectorstore = LangchainPinecone.from_existing_index(
            index_name=index_name,
            embedding=embeddings,
            text_key="text",
            namespace=namespace,
            pinecone_kwargs={"api_key": api_key}
        )
        
        # Create retriever
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 8  # Top 8 similar chunks
            }
        )
        
        return retriever
        
    except Exception as e:
        st.error(f"Error creating LangChain retriever: {str(e)}")
        return None

def delete_namespace(namespace: str = "default"):
    """Delete all vectors in a namespace."""
    index = initialize_pinecone()
    if not index:
        return False
    
    try:
        index.delete(delete_all=True, namespace=namespace)
        st.success(f"Successfully deleted all vectors in namespace: {namespace}")
        return True
    except Exception as e:
        st.error(f"Error deleting namespace: {str(e)}")
        return False