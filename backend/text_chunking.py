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
    if not text or not isinstance(text, str): #checking if text is empty from pdf or text is not string
        st.warning("No valid text to process")
        return [], [], []
    
    try:
        text_splitter = CharacterTextSplitter(
            separator="\n", #split text on newline character
            chunk_size=1000,  #one text chunk contains 1000 characters
            chunk_overlap=200, #200 character overlaps between adjacent chunks
            length_function=len #to calculate length of text
        )
        
        chunks = text_splitter.split_text(text) #creating chunks based on text_splitter parameters
        
        if not chunks: #checking if chunks are empty
            st.warning("No chunks created")
            return [], [], []
        
        model_path = ensure_model_exists("all-MiniLM-L6-v2") #ensure_model_exists function from model_utils.py where this embedding model gets downloaded
        if not model_path: #if this model is not there then error is shown
            st.error("Failed to load embedding model")
            return chunks, [], []
        
        embeddings_model = HuggingFaceEmbeddings( #embedding model
            model_name=model_path, #model
            model_kwargs={'device': 'cpu'} #runs on cpu for me
        )
        
        chunk_metadata = []
        for i, chunk in enumerate(chunks): #iterating through chunks using i,chunk pairs in chunks
            chunk_meta = {                  #chunk_meta dictionary which stores chunk with index
                "text": chunk,
                "chunk_index": i
            }
            
            if metadata and isinstance(metadata, dict): #checking if metadata is there and its dictionary
                for key, value in metadata.items(): #iterating through key,value pairs of metadata
                    if key != "text" and key != "chunk_index": #checking if key is not text and not chunk_index
                        chunk_meta[key] = value                 #adding key,value pair to chunk_meta
            
            chunk_metadata.append(chunk_meta) #adding chunk_meta to chunk_metadata list
        
        raw_embeddings = embeddings_model.embed_documents(chunks) #creating embddings for each chunk using embed_documents function of embeddings_model
        
        return chunks, raw_embeddings, chunk_metadata
        
    except Exception as e:
        st.error(f"Error chunking and embedding text: {str(e)}")
        return [], [], []