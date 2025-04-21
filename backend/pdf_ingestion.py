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
        text = "" #string to store all the extracted text from the inputed pdf
        pdf_reader = PdfReader(pdf_path) #object to access the content and metadata of the pdf accessing from pdf path
        
        metadata = {}               #dictionary to store metadata of the pdf
        if pdf_reader.metadata:     #if there is metadata in the pdf
            for key, value in pdf_reader.metadata.items(): #iterating key,value pairs of th metadata
                if key and value and isinstance(key, str) and isinstance(value, str): #whether key and value exists? and whether they are strings
                    clean_key = key.replace('/', '') if key.startswith('/') else key #if key starts with / then remove it else same key. ex: /Author to Author 
                    metadata[clean_key] = value #adding key,value pair to metadata dictionary
        
        for page in pdf_reader.pages: #iterating through pages of the pdf
            page_text = page.extract_text() #if page=1 extracting text from page 1 and storing it in page_text object
            if page_text: #if page_text is not empty or not None
                text += page_text + "\n\n" #adding page_text to main text string and adding 1 new line after each page_text
        
        if not text: #if text is empty-
            st.warning("No text extracted from PDF. The file might be scanned or image-based.")
        
        return text, metadata
    
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None, {}