import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from backend.model_utils import ensure_model_exists

def chunk_and_embed(text, metadata=None):
    """
    Split text into chunks and create embeddings using LangChain.
    
    Args:
        text: Text to chunk and embed
        metadata: Document metadata
        
    Returns:
        chunks: Text chunks
        embeddings: Embeddings for each chunk
        chunk_metadata: Metadata for each chunk
    """
    if not text or not isinstance(text, str):
        st.warning("No valid text to process")
        return [], [], []
    
    try:
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        chunks = text_splitter.split_text(text)
        
        if not chunks:
            st.warning("No chunks created")
            return [], [], []
        
        model_path = ensure_model_exists("all-MiniLM-L6-v2")
        if not model_path:
            st.error("Failed to load embedding model")
            return chunks, [], []
        
        embeddings_model = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={'device': 'cpu'}
        )
        
        chunk_metadata = []
        for i, chunk in enumerate(chunks):
            chunk_meta = {
                "text": chunk,
                "chunk_index": i
            }
            
            if metadata and isinstance(metadata, dict):
                for key, value in metadata.items():
                    if key != "text" and key != "chunk_index":
                        chunk_meta[key] = value
            
            chunk_metadata.append(chunk_meta)
        
        raw_embeddings = embeddings_model.embed_documents(chunks)
        
        return chunks, raw_embeddings, chunk_metadata
        
    except Exception as e:
        st.error(f"Error chunking and embedding text: {str(e)}")
        return [], [], []