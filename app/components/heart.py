import streamlit as st
import os
import tempfile
import uuid
import numpy as np
from pathlib import Path

# Import custom modules
from backend.pdf_ingestion import extract_text_from_pdf
from backend.text_chunking import chunk_and_embed
from backend.pinecone_storage import initialize_pinecone, store_embeddings
from backend.query_processing import process_query
from backend.retrieval import retrieve_chunks
from backend.response_generation import generate_response
from landing_page.components.navbar import render_navbar
from landing_page.components.footer import render_footer

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "namespace" not in st.session_state:
    st.session_state.namespace = f"session_{uuid.uuid4().hex[:8]}"

# Load CSS styling
def load_css():
    # Load the existing CSS from the landing page
    with open("landing_page/styles/styles.css") as f:
        main_css = f.read()
    
    # Add additional CSS specific to the heart.py functionality
    additional_css = """
    /* Query interface styling */
    .chat-container {
        background-color: #2a2a2a;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 10px 30px rgba(0, 255, 0, 0.2), 
                    0 0 15px rgba(0, 255, 0, 0.15) inset;
        border: 1px solid rgba(0, 255, 0, 0.1);
    }
    
    .user-message {
        background-color: #3a3a3a;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        border-left: 3px solid #00ff00;
    }
    
    .assistant-message {
        background-color: #222222;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        border-left: 3px solid #00ff00;
    }
    
    /* Upload section styling */
    .upload-section {
        background-color: #2a2a2a;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 10px 30px rgba(0, 255, 0, 0.2), 
                    0 0 15px rgba(0, 255, 0, 0.15) inset;
        border: 1px solid rgba(0, 255, 0, 0.1);
    }
    
    .file-list {
        margin-top: 20px;
        padding: 10px;
        background-color: #333333;
        border-radius: 8px;
    }
    
    .file-item {
        display: flex;
        align-items: center;
        padding: 8px;
        border-bottom: 1px solid rgba(0, 255, 0, 0.1);
    }
    
    .file-item:last-child {
        border-bottom: none;
    }
    
    .file-icon {
        color: #00ff00;
        margin-right: 10px;
    }
    
    /* Custom button styling */
    .custom-button {
        background-color: #00ff00;
        color: #1a1a1a;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .custom-button:hover {
        background-color: #00cc00;
        transform: translateY(-2px);
    }
    
    /* Chat input styling */
    .stTextInput>div>div>input {
        background-color: #333333;
        border: 1px solid rgba(0, 255, 0, 0.3);
        color: white;
    }
    
    /* Background styling */
    body {
        background-color: #1a1a1a !important;
    }
    
    .stApp {
        background-color: #1a1a1a;
        background: radial-gradient(circle, rgba(0, 255, 0, 0.06) 0%, transparent 70%);
    }
    
    html {
        scroll-behavior: smooth;
    }
    """
    
    # Combine CSS and apply
    st.markdown(f"<style>{main_css}\n{additional_css}</style>", unsafe_allow_html=True)

def show_main_app():
    # Initialize session state variables - moved inside the function
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []
    if "namespace" not in st.session_state:
        st.session_state.namespace = f"session_{uuid.uuid4().hex[:8]}"
    if "query_input" not in st.session_state:
        st.session_state.query_input = ""


    # Apply styling
    load_css()
    
    # Render navbar
    render_navbar()
    
    # Add a container with proper styling for the main content
    st.markdown(
        """
        <div style="padding: 40px; max-width: 1200px; margin: 0 auto; color: white;">
        <h1 style="color: #00ff00; font-size: 42px; margin-bottom: 30px;">QueryQuack - PDF Query Engine</h1>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Function to process uploaded files
    def process_uploaded_files(uploaded_files):
        with st.spinner("Processing files..."):
            # Initialize Pinecone
            index = initialize_pinecone()
            if not index:
                st.error("Failed to connect to Pinecone. Check your API key.")
                return False

            # Process each file
            for file in uploaded_files:
                if file.name in st.session_state.processed_files:
                    continue

                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                    tmp.write(file.getvalue())
                    pdf_path = tmp.name

                try:
                    # Extract text and metadata from PDF
                    text_result = extract_text_from_pdf(pdf_path)
                    if isinstance(text_result, tuple) and len(text_result) == 2:
                        text, pdf_metadata = text_result
                    else:
                        text = text_result
                        pdf_metadata = {}

                    if not text or not isinstance(text, str):
                        st.warning(f"No valid text extracted from {file.name}")
                        continue

                    # Create metadata for the document
                    metadata = {
                        "filename": file.name,
                        "source": "uploaded_pdf"
                    }
                    # Merge in any additional metadata
                    if isinstance(pdf_metadata, dict):
                        for key, value in pdf_metadata.items():
                            if key != "filename":
                                metadata[key] = value

                    # Chunk and embed text
                    chunks, embeddings, chunk_metadata = chunk_and_embed(text, metadata)

                    if not chunks:
                        st.warning(f"No chunks created for {file.name}")
                        continue

                    # Store embeddings in Pinecone
                    store_success = store_embeddings(
                        embeddings,
                        chunk_metadata,
                        namespace=st.session_state.namespace
                    )

                    if store_success:
                        st.session_state.processed_files.append(file.name)

                finally:
                    # Remove temporary file
                    if os.path.exists(pdf_path):
                        os.remove(pdf_path)

            return len(st.session_state.processed_files) > 0
    
    # Function to display chat messages
    def display_chat_history():
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message"><strong>You:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="assistant-message"><strong>QueryQuack 🦆:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)
    
    # Process user query
    def handle_user_query():
        if not st.session_state.query_input:
            return
        
        query = st.session_state.query_input
        
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        # Clear the input
        st.session_state.query_input = ""
        
        try:
            # Process the query
            query_embedding, processed_query = process_query(query)
            
            if query_embedding is None:
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": "Sorry, I had trouble processing your query. Please try again."
                })
                return
            
            # Retrieve relevant chunks
            chunks = retrieve_chunks(
                query_embedding, 
                namespace=st.session_state.namespace,
                top_k=5
            )
            
            if not chunks:
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": "I couldn't find relevant information in the uploaded documents. Please try a different question or upload more files."
                })
                return
            
            # Generate response
            answer = generate_response(processed_query, chunks)
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": answer
            })
            
        except Exception as e:
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": f"An error occurred: {str(e)}"
            })
    
    # Create two-column layout
    col1, col2 = st.columns([1, 1])
    
    # Left column - File upload section
    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.header("Upload PDF Files")
        
        uploaded_files = st.file_uploader(
            "Select PDFs to upload", 
            type=["pdf"], 
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.write(f"Selected {len(uploaded_files)} files:")
            for file in uploaded_files:
                st.markdown(f"- {file.name}")
            
            process_button = st.button("Process Files", use_container_width=True, key="process_button")
            if process_button:
                success = process_uploaded_files(uploaded_files)
                if success:
                    st.success("✅ Files processed successfully!")
                else:
                    st.error("❌ Failed to process files.")
        
        # Show processed files
        if st.session_state.processed_files:
            st.markdown("### Processed Files")
            st.markdown('<div class="file-list">', unsafe_allow_html=True)
            for file in st.session_state.processed_files:
                st.markdown(f'<div class="file-item"><span class="file-icon">📄</span> {file}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Right column - Query interface
    with col2:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        st.header("Ask Questions About Your Documents")
        
        # Display chat history
        display_chat_history()
        
        # Only show query input if files have been processed
        if st.session_state.processed_files:
            # Initialize query input in session state if not exists
            if "query_input" not in st.session_state:
                st.session_state.query_input = ""
            
            # Create text input for query
            st.text_input(
                "Type your question here",
                key="query_input",
                on_change=handle_user_query,
                placeholder="What would you like to know about your documents?"
            )
            
            st.markdown(
                """
                <div style="font-size: 12px; color: #cccccc; margin-top: 5px;">
                Press Enter to submit your question
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.info("Please upload and process PDF files to start asking questions.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Add spacing before the footer
    st.markdown("<div style='margin-top: 100px;'></div>", unsafe_allow_html=True)
    
    # Render the footer
    st.markdown('<div id="footer"></div>', unsafe_allow_html=True)
    footer_html = render_footer()
    st.markdown(footer_html, unsafe_allow_html=True)

if __name__ == "__main__":
    show_main_app()