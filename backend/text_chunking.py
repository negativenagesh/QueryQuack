from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
import os
import warnings

# Try importing SentenceTransformer with a fallback mechanism
EMBEDDING_AVAILABLE = False
try:
    import numpy
    from sentence_transformers import SentenceTransformer
    from backend.model_utils import ensure_model_exists, MODELS_DIR
    EMBEDDING_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    st.warning(f"Advanced embedding functionality unavailable: {str(e)}")
    warnings.warn(f"SentenceTransformer or NumPy initialization failed: {e}")

def simple_text_splitter(text, chunk_size=350, overlap=100):
    """Simple text chunking when advanced methods fail"""
    chunks = []
    # Split by double newlines first
    paragraphs = text.split("\n\n")
    for para in paragraphs:
        # If paragraph is too large, split it further
        if len(para) > chunk_size:
            words = para.split()
            current_chunk = []
            current_length = 0
            
            for word in words:
                current_length += len(word) + 1  # +1 for space
                current_chunk.append(word)
                
                if current_length >= chunk_size:
                    chunks.append(" ".join(current_chunk))
                    # Keep overlap words for context
                    overlap_words = current_chunk[-int(len(current_chunk) * overlap/chunk_size):]
                    current_chunk = overlap_words
                    current_length = len(" ".join(current_chunk)) 
            
            if current_chunk:
                chunks.append(" ".join(current_chunk))
        else:
            chunks.append(para)
    
    # If we have no chunks, create at least one with the full text
    if not chunks and text:
        chunks = [text]
        
    return chunks

def chunk_and_embed(text, metadata, chunk_size=350, chunk_overlap=50):
    """Split text into chunks and optionally generate embeddings if available."""
    try:
        if EMBEDDING_AVAILABLE:
            # Using advanced chunking with langchain
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = splitter.split_text(text)
            
            if not chunks:
                st.warning(f"No chunks created for {metadata['filename']}.")
                return [], [], []

            try:
                # Try to load model and generate embeddings
                model_path = ensure_model_exists("all-MiniLM-L6-v2")
                if not model_path:
                    raise ValueError("Model path not available")
                    
                model = SentenceTransformer(model_path)
                embeddings = model.encode(chunks, convert_to_numpy=True)
                
                chunk_metadata = []
                for i, chunk in enumerate(chunks):
                    chunk_meta = metadata.copy()
                    chunk_meta['chunk_index'] = i
                    chunk_meta['text'] = chunk
                    chunk_metadata.append(chunk_meta)
                
                return chunks, embeddings, chunk_metadata
            
            except Exception as model_error:
                st.warning(f"Embedding generation failed: {str(model_error)}. Using simple text chunks instead.")
                # Continue with chunks but use placeholder embeddings
        
        # Fallback to simple chunking when advanced methods aren't available
        chunks = simple_text_splitter(text, chunk_size, chunk_overlap)
        
        if not chunks:
            st.warning(f"No chunks created for {metadata['filename']}.")
            return [], [], []
        
        # Create placeholder embeddings (random values) when embedding fails
        import random
        random.seed(42)  # For reproducibility
        embed_dim = 384  # Standard dimension for all-MiniLM-L6-v2
        placeholder_embeddings = []
        
        for _ in chunks:
            # Create random vector normalized to unit length
            vec = [random.uniform(-1, 1) for _ in range(embed_dim)]
            # Normalize to unit length (simulates proper embeddings)
            magnitude = sum(x*x for x in vec) ** 0.5
            vec = [x/magnitude for x in vec]
            placeholder_embeddings.append(vec)
        
        # Create metadata for each chunk
        chunk_metadata = []
        for i, chunk in enumerate(chunks):
            chunk_meta = metadata.copy()
            chunk_meta['chunk_index'] = i
            chunk_meta['text'] = chunk
            chunk_meta['is_placeholder'] = True  # Mark as placeholder embedding
            chunk_metadata.append(chunk_meta)
        
        st.warning("Using fallback chunking method without semantic embeddings. Search results may be less accurate.")
        return chunks, placeholder_embeddings, chunk_metadata

    except Exception as e:
        st.error(f"Error in chunking and embedding: {str(e)}")
        return [], [], []