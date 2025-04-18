import streamlit as st
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

def render_hero_section():
    hero_html = """
    <div class="hero-text">
        <h1>QueryQuack</h1>
        <h2>Quack the Query, Crack the PDF!</h2>
        <p>Your AI-powered platform to unlock insights from PDFs with lightning-fast retrieval and smart answers.</p>
        <div class="hero-buttons">
            <a href="#powerful_features" class="hero-button explore-btn" onclick="parent.location='#powerful_features'">Discover Features</a>
            <a href="#how_queryquack_works" class="hero-button how-it-works-btn" onclick="parent.location='#how_queryquack_works'">How It Works</a>
            <a href="/?page=main" class="hero-button explore-btn" style="margin-top: 10px;">Start Quacking</a>
        </div>
        <div class="hero-tags">
            <span class="hero-tag">PDF-Powered</span>
            <span class="hero-tag">AI-Driven</span>
            <span class="hero-tag">Instant Insights</span>
        </div>
    </div>
    """
    
    return hero_html