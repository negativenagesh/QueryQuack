import streamlit as st
import os
import sys

st.set_page_config(
    page_title="QueryQuack",
    page_icon="landing_page/styles/images/logo.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from landing_page.app import render_landing_page
from components.heart import show_main_app

def main():
    """Main application entry point."""
    params = st.query_params
    
    if "page" in params and params["page"] == "main":
        show_main_app()
    else:
        render_landing_page()

if __name__ == "__main__":
    main()