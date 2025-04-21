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

load_dotenv()

def initialize_pinecone():
    """Initialize and return Pinecone index."""
    api_key = os.environ.get("PINECONE_API_KEY")
    index_name = os.environ.get("PINECONE_INDEX", "queryquack")
    
    if not api_key:
        st.error("Pinecone API key not found. Please set the PINECONE_API_KEY in your .env file.")
        return None
    
    try:
        pc = Pinecone(api_key=api_key)
        
        existing_indexes = pc.list_indexes().names()
        if index_name not in existing_indexes:
            pc.create_index(
                name=index_name,
                dimension=384,
                metric="cosine"
            )
            st.info(f"Created new Pinecone index: {index_name}")
        
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
    
    is_empty_embeddings = False
    is_empty_metadata = False
    
    if isinstance(embeddings, np.ndarray):
        is_empty_embeddings = embeddings.size == 0
    else:
        is_empty_embeddings = len(embeddings) == 0 if embeddings is not None else True
    
    is_empty_metadata = len(metadata_list) == 0 if metadata_list is not None else True
    
    if is_empty_embeddings or is_empty_metadata:
        st.warning("No embeddings to store")
        return False
    
    total_vectors = len(embeddings) if not isinstance(embeddings, np.ndarray) else embeddings.shape[0]
    
    try:
        for i in range(0, total_vectors, batch_size):
            batch_end = min(i + batch_size, total_vectors)
            
            vectors_batch = []
            for j in range(i, batch_end):
                vector_id = str(uuid.uuid4())
                metadata = metadata_list[j].copy()
                metadata["chunk_index"] = j
                
                if isinstance(embeddings, np.ndarray):
                    vector = embeddings[j].tolist()
                else:
                    vector = embeddings[j]
                
                vectors_batch.append({
                    "id": vector_id,
                    "values": vector,
                    "metadata": metadata
                })
            
            index.upsert(vectors=vectors_batch, namespace=namespace)
            
            if batch_end < total_vectors:
                time.sleep(0.5)
            
            if i == 0:
                progress_bar = st.progress(0)
            progress = min(1.0, batch_end / total_vectors)
            if 'progress_bar' in locals():
                progress_bar.progress(progress)
        
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
        api_key = os.environ.get("PINECONE_API_KEY")
        index_name = os.environ.get("PINECONE_INDEX", "queryquack")
        
        if not api_key:
            st.error("Pinecone API key not found")
            return None
        
        model_path = ensure_model_exists("all-MiniLM-L6-v2")
        if not model_path:
            st.error("Failed to load embedding model")
            return None
            
        embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={'device': 'cpu'}
        )
        
        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)
        
        vectorstore = LangchainPinecone.from_existing_index(
            index_name=index_name,
            embedding=embeddings,
            text_key="text",
            namespace=namespace,
            pinecone_kwargs={"api_key": api_key}
        )
        
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 8
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