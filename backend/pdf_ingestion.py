import streamlit as st
from PyPDF2 import PdfReader
import tempfile
import os

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        text: Extracted text
        metadata: PDF metadata
    """
    try:
        text = ""
        pdf_reader = PdfReader(pdf_path)
        
        metadata = {}
        if pdf_reader.metadata:
            for key, value in pdf_reader.metadata.items():
                if key and value and isinstance(key, str) and isinstance(value, str):
                    clean_key = key.replace('/', '') if key.startswith('/') else key
                    metadata[clean_key] = value
        
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
        
        if not text:
            st.warning("No text extracted from PDF. The file might be scanned or image-based.")
        
        return text, metadata
    
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None, {}