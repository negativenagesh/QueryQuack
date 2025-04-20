import streamlit as st
import os
import numpy as np

# Try importing the needed modules with a fallback
ADVANCED_RETRIEVAL = False
try:
    from sentence_transformers import CrossEncoder
    from backend.model_utils import ensure_model_exists, MODELS_DIR
    ADVANCED_RETRIEVAL = True
except (ImportError, RuntimeError):
    st.warning("Advanced retrieval features unavailable")

def retrieve_chunks(query_embedding, namespace='default', top_k=5):
    """Retrieve top-k chunks from Pinecone and re-rank them if possible."""
    try:
        from backend.pinecone_storage import initialize_pinecone
        index = initialize_pinecone()
        if not index:
            st.error("Failed to connect to Pinecone.")
            return []
            
        # Make sure query_embedding is a list (not numpy array)
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
            
        # Initial retrieval
        results = index.query(
            vector=query_embedding,
            top_k=top_k * 2,  # Retrieve more for re-ranking
            include_metadata=True,
            namespace=namespace
        )
        
        if not results or 'matches' not in results or not results['matches']:
            return []
            
        chunks = [
            (match['metadata']['filename'], match['metadata'].get('chunk_index', 0), match['metadata']['text'])
            for match in results['matches']
            if 'metadata' in match and 'text' in match['metadata']
        ]
        
        if not chunks:
            return []
        
        # Only perform re-ranking if advanced features are available
        if ADVANCED_RETRIEVAL:
            try:
                model_path = ensure_model_exists("ms-marco-MiniLM-L-6-v2")
                if model_path:
                    cross_encoder = CrossEncoder(model_path)
                    
                    # Use the query as the first element for comparison
                    query_text = results['matches'][0]['metadata'].get('query_text', '')
                    if not query_text:
                        # If query_text isn't in metadata, use the first chunk's text as context
                        query_text = chunks[0][2]
                        
                    pairs = [(query_text, chunk[2]) for chunk in chunks]
                    scores = cross_encoder.predict(pairs)
                    
                    # Sort by score and return top-k
                    ranked_chunks = [
                        chunks[i] for i in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                    ][:top_k]
                    
                    return ranked_chunks
            except Exception as e:
                st.warning(f"Re-ranking failed: {str(e)}. Using basic retrieval instead.")
                
        # Fallback to basic top-k if re-ranking is unavailable or fails
        return chunks[:top_k]
        
    except Exception as e:
        st.error(f"Error retrieving chunks: {str(e)}")
        return []