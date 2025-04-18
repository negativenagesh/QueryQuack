from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import streamlit as st
import numpy as np

def chunk_and_embed(text, metadata, chunk_size=500, chunk_overlap=100):
    """Split text into chunks and generate embeddings."""
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        chunks = splitter.split_text(text)
        if not chunks:
            st.warning(f"No chunks created for {metadata['filename']}.")
            return [], [], metadata

        # Generate embeddings
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(chunks, convert_to_numpy=True)
        
        # Update metadata for each chunk
        chunk_metadata = [
            {**metadata, "chunk_index": i, "text": chunk}
            for i, chunk in enumerate(chunks)
        ]
        
        return chunks, embeddings, chunk_metadata
    except Exception as e:
        st.error(f"Error chunking or embedding {metadata['filename']}: {str(e)}")
        return [], [], metadata