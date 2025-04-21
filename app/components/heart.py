import streamlit as st
import os
import tempfile
import uuid
import numpy as np
from pathlib import Path

from backend.pdf_ingestion import extract_text_from_pdf
from backend.text_chunking import chunk_and_embed
from backend.pinecone_storage import initialize_pinecone, store_embeddings
from backend.query_processing import process_query
from backend.retrieval import retrieve_chunks
from backend.response_generation import generate_response
from landing_page.components.navbar import render_navbar
from landing_page.components.footer import render_footer

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "namespace" not in st.session_state:
    st.session_state.namespace = f"session_{uuid.uuid4().hex[:8]}"

def load_css():
    with open("landing_page/styles/styles.css") as f:
        main_css = f.read()
    
    with open("landing_page/styles/styles.css") as f:
        additional_css = f.read()
    
    st.markdown(f"<style>{main_css}\n{additional_css}</style>", unsafe_allow_html=True)

def show_main_app():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []
    if "namespace" not in st.session_state:
        st.session_state.namespace = f"session_{uuid.uuid4().hex[:8]}"
    if "query_input" not in st.session_state:
        st.session_state.query_input = ""

    load_css()
    
    render_navbar()
    
    st.markdown(
        """
        <div style="padding: 40px; max-width: 1200px; margin: 0 auto; text-align: center; color: white;">
        <h1 style="color: white; font-size: 42px; margin-bottom: 30px;">QueryQuack - PDF Query Engine</h1>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    def process_uploaded_files(uploaded_files):
        with st.spinner("Processing files..."):
            index = initialize_pinecone()
            if not index:
                st.error("Failed to connect to Pinecone. Check your API key.")
                return False

            for file in uploaded_files:
                if file.name in st.session_state.processed_files:
                    continue

                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                    tmp.write(file.getvalue())
                    pdf_path = tmp.name

                try:
                    text_result = extract_text_from_pdf(pdf_path)
                    if isinstance(text_result, tuple) and len(text_result) == 2:
                        text, pdf_metadata = text_result
                    else:
                        text = text_result
                        pdf_metadata = {}

                    if not text or not isinstance(text, str):
                        st.warning(f"No valid text extracted from {file.name}")
                        continue

                    metadata = {
                        "filename": file.name,
                        "source": "uploaded_pdf"
                    }
                    if isinstance(pdf_metadata, dict):
                        for key, value in pdf_metadata.items():
                            if key != "filename":
                                metadata[key] = value

                    chunks, embeddings, chunk_metadata = chunk_and_embed(text, metadata)

                    if not chunks:
                        st.warning(f"No chunks created for {file.name}")
                        continue

                    store_success = store_embeddings(
                        embeddings,
                        chunk_metadata,
                        namespace=st.session_state.namespace
                    )

                    if store_success:
                        st.session_state.processed_files.append(file.name)

                finally:
                    if os.path.exists(pdf_path):
                        os.remove(pdf_path)

            return len(st.session_state.processed_files) > 0
    
    def display_chat_history():
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message"><strong>You:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="assistant-message"><strong>QueryQuack 🦆:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)
    
    def handle_user_query():
        if not st.session_state.query_input:
            return
        
        query = st.session_state.query_input
        
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        st.session_state.query_input = ""
        
        try:
            if 'sources_used' not in st.session_state:
                st.session_state['sources_used'] = []
                
            if 'debug_info' not in st.session_state:
                st.session_state['debug_info'] = {}
            
            query_embedding, processed_query, original_query = process_query(query)
            
            if query_embedding is None:
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": "Sorry, I had trouble processing your query. Please try again."
                })
                return
            
            chunks = retrieve_chunks(
                query_embedding,
                query_text=processed_query,
                namespace=st.session_state.namespace,
                top_k=8
            )
            
            if not chunks:
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": "I couldn't find relevant information in the uploaded documents. Please try a different question or upload more files."
                })
                return
            
            answer = generate_response(original_query, chunks)
            
            response_text = answer
            
            if st.session_state['sources_used']:
                sources_text = "\n\n**Sources:**\n"
                for filename, chunk_index in st.session_state['sources_used']:
                    sources_text += f"- {filename} (Chunk {chunk_index})\n"
                response_text += sources_text
            
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": response_text
            })
            
            if st.session_state.get('debug_info') and len(st.session_state['debug_info']) > 0:
                with st.expander("Debug Info", expanded=False):
                    st.write(st.session_state['debug_info'])
            
        except Exception as e:
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": f"An error occurred: {str(e)}"
            })
    
    col1, col2 = st.columns([1, 1])
    
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
                    st.success("Files processed successfully!")
                else:
                    st.error("Failed to process files.")
        
        if st.session_state.processed_files:
            st.markdown("### Processed Files")
            st.markdown('<div class="file-list">', unsafe_allow_html=True)
            for file in st.session_state.processed_files:
                st.markdown(f'<div class="file-item"><span class="file-icon">📄</span> {file}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        st.header("Ask Questions About Your Documents")
        
        display_chat_history()
        
        if st.session_state.processed_files:
            if "query_input" not in st.session_state:
                st.session_state.query_input = ""
            
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
    
    st.markdown("<div style='margin-top: 100px;'></div>", unsafe_allow_html=True)
    
    st.markdown('<div id="footer"></div>', unsafe_allow_html=True)
    footer_html = render_footer()
    st.markdown(footer_html, unsafe_allow_html=True)

if __name__ == "__main__":
    show_main_app()