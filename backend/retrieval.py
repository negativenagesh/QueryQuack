import pinecone
from sentence_transformers import CrossEncoder
import streamlit as st

def retrieve_chunks(query_embedding, namespace='default', top_k=5):
    """Retrieve top-k chunks from Pinecone and re-rank them."""
    index = pinecone.Index('pdf-embeddings')
    try:
        # Initial retrieval
        results = index.query(
            vector=query_embedding.tolist(),
            top_k=top_k * 2,  # Retrieve more for re-ranking
            include_metadata=True,
            namespace=namespace
        )
        chunks = [
            (match['metadata']['pdf'], match['metadata']['chunk_index'], match['metadata']['text'])
            for match in results['matches']
        ]
        
        # Re-rank with cross-encoder
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        query_text = results['matches'][0]['metadata']['text']  # Approximate query context
        pairs = [(query_text, chunk[2]) for chunk in chunks]
        scores = cross_encoder.predict(pairs)
        
        # Sort by score and return top-k
        ranked_chunks = [
            chunks[i] for i in np.argsort(scores)[::-1][:top_k]
        ]
        return ranked_chunks
    except Exception as e:
        st.error(f"Error retrieving chunks: {str(e)}")
        return []