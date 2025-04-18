import streamlit as st
import os
import sys
import pdfplumber  # Import pdfplumber for PDF processing

st.set_page_config(
    page_title="QueryQuack",
    page_icon="landing_page/styles/images/logo.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from landing_page.app import render_landing_page
from landing_page.components.navbar import render_navbar
from landing_page.components.footer import render_footer

def show_main_app():
    """Display the main application with PDF upload and query functionality"""
    # Add full background color and styling
    st.markdown(
        """
        <style>
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
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Render navbar
    render_navbar()
    
    # Apply the same CSS styling
    with open("landing_page/styles/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    # Add a container with proper styling for the main content
    st.markdown(
        """
        <div style="padding: 40px; max-width: 1200px; margin: 0 auto; color: white;">
        <h1 style="color: #00ff00; font-size: 42px; margin-bottom: 30px;">QueryQuack - PDF Query Engine</h1>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # File upload section
    st.header("Upload Your PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Display success message
        st.success(f"Successfully uploaded: {uploaded_file.name}")
        
        try:
            # Extract text from PDF
            with pdfplumber.open(uploaded_file) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() + "\n\n"
                
            # Show a preview of the PDF content
            st.subheader("PDF Content Preview")
            st.text_area("Extracted Text", text[:1000] + "..." if len(text) > 1000 else text, height=200)
            
            # Query section
            st.header("Ask Questions About Your PDF")
            query = st.text_input("Enter your question:")
            
            if query:
                st.info("Processing your query...")
                
                # Simulate response processing (replace with actual implementation)
                response = "This would be the answer to your query based on the PDF content."
                
                st.subheader("Answer")
                st.write(response)
                
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
    
    # Add spacing before the footer
    st.markdown("<div style='margin-top: 100px;'></div>", unsafe_allow_html=True)
    
    # Render the footer
    st.markdown('<div id="footer"></div>', unsafe_allow_html=True)
    footer_html = render_footer()
    st.markdown(footer_html, unsafe_allow_html=True)

def main():
    """Main application entry point."""
    # Check if we need to show the main app or landing page
    params = st.query_params
    
    if "page" in params and params["page"] == "main":
        show_main_app()
    else:
        render_landing_page()

if __name__ == "__main__":
    main()