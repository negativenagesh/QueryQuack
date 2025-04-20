import streamlit as st
import numpy as np
from backend.pinecone_storage import initialize_pinecone

def retrieve_chunks(query_embedding, query_text=None, namespace="default", top_k=5):
    """
    Retrieve relevant chunks using vector similarity search.
    
    Args:
        query_embedding: Query embedding vector
        query_text: Optional query text for hybrid search/reranking
        namespace: Pinecone namespace
        top_k: Number of results to return
        
    Returns:
        chunks: List of relevant text chunks with metadata
    """
    try:
        # Initialize Pinecone index
        index = initialize_pinecone()
        if not index:
            st.error("Failed to initialize Pinecone for retrieval")
            return []
        
        # Perform vector search
        search_results = index.query(
            namespace=namespace,
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # Extract and format results
        chunks = []
        for match in search_results.matches:
            # Skip results with no metadata
            if not hasattr(match, 'metadata') or not match.metadata:
                continue
                
            metadata = match.metadata
            
            # Ensure text field exists
            if 'text' not in metadata:
                continue
                
            # Create chunk object
            chunk = {
                'text': metadata['text'],
                'metadata': metadata,
                'score': match.score
            }
            
            # Store source information for attribution
            if 'sources_used' not in st.session_state:
                st.session_state['sources_used'] = []
                
            if 'filename' in metadata and 'chunk_index' in metadata:
                source_info = (metadata['filename'], metadata['chunk_index'])
                if source_info not in st.session_state['sources_used']:
                    st.session_state['sources_used'].append(source_info)
            
            chunks.append(chunk)
        
        # Print the number of chunks retrieved for debugging
        if len(chunks) > 0:
            st.success(f"Retrieved {len(chunks)} relevant chunks from document.")
        else:
            st.warning("No relevant chunks found in the document.")
        
        # Store debug info
        if 'debug_info' not in st.session_state:
            st.session_state['debug_info'] = {}
            
        st.session_state['debug_info']['retrieval'] = {
            'query_text': query_text,
            'top_k': top_k,
            'num_results': len(chunks),
            'namespace': namespace
        }
        
        return chunks
        
    except Exception as e:
        st.error(f"Error retrieving chunks: {str(e)}")
        return []