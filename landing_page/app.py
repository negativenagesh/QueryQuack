import streamlit as st
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from landing_page.components.navbar import render_navbar
from landing_page.components.footer import render_footer
from landing_page.components.hero_section import render_hero_section
from landing_page.components.pdf_preview import render_pdf_preview

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

def render_landing_page():
    params = st.query_params
        
    if "page" in params and params["page"] == "landing":
        st.query_params.clear()  
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
        /* Smooth scrolling for anchor links */
        html {
            scroll-behavior: smooth;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    render_navbar()
    with open("landing_page/styles/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    st.markdown('<div id="hero-section"></div>', unsafe_allow_html=True)
    hero_html = render_hero_section()
    pdf_html = render_pdf_preview()

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(hero_html, unsafe_allow_html=True)
    with col2:
        st.markdown(pdf_html, unsafe_allow_html=True)

    st.markdown(
        """
        <div style="text-align: center; margin-top: 10px;">
        <a href="#why_resumai_exists" id="scroll-arrow" style="text-decoration: none;">
            <span style="font-size: 70px; color: #00ff00; transition: transform 0.3s ease; display: inline-block;">↓</span>
        </a>
        </div>
        <style>
        #scroll-arrow span:hover {
            transform: scale(1.2);
        }
        #scroll-arrow span:active {
            transform: scale(0.9);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div id="footer"></div>', unsafe_allow_html=True)
    footer_html = render_footer()
    st.markdown(footer_html, unsafe_allow_html=True)

if __name__ == "__main__":
    render_landing_page()