import streamlit as st
import os

def get_image_as_base64(path):
    """Convert an image file to base64 string"""
    import base64
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def render_footer():
    logo_path = "landing_page/styles/images/logo.png"

    footer_html = f"""
    <div class="footer-container">
        <div class="footer-content">
            <div class="footer-column">
                <img src="data:image/png;base64,{get_image_as_base64(logo_path)}" height="130" alt="Resumai", style="margin-bottom: 20px;">
                <p>Elevate your career journey with our AI-powered resume optimization platform designed to help you break through ATS barriers and land your dream job.</p>
                <div class="social-icons">
                    <a href="https://github.com/negativenagesh/QueryQuack" target="_blank">
                    <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub" style="width: 25px; height: 25px; margin-left: 10px;">
                </a>
                </div>
            </div>
            <div class="footer-column">
                <h3>Quick Links</h3>
                <ul>
                    <li><a href="#hero-section">Home</a></li>
                    <li><a href="#why_resumai_exists">About</a></li>
                    <li><a href="#powerful_features">Features</a></li>
                    <li><a href="#how_resumai_works">How It Works</a></li>
                    <li><a href="#resumai_benefits">Benefits</a></li>
                    <li><a href="#success_stories">Testimonials</a></li>
                    <li><a href="#faq">FAQ</a></li>
                    <li><a href="#get_in_touch">Contact</a></li>
                </ul>
            </div>
            <div class="footer-column">
                <h3>Resources</h3>
            <ul>
                <li><a href="https://github.com/negativenagesh/QueryQuack" target="_blank">GitHub Repository</a></li>
                <li><a href="https://github.com/negativenagesh/QueryQuack/blob/main/README.md" target="_blank">Documentation</a></li>
                <li><a href="https://github.com/negativenagesh/QueryQuack?tab=readme-ov-file#setup-instructions" target="_blank">Setup</a></li>
                <li><a href="https://github.com/negativenagesh/QueryQuack?tab=readme-ov-file#preview-click-on-image-to-watch-video" target="_blank">Demo</a></li>
                <li><a href="https://github.com/negativenagesh/QueryQuack/blob/main/README.md#-contributing" target="_blank">Contributing Guide</a></li>
                <li><a href="https://github.com/negativenagesh/QueryQuack/blob/main/LICENSE" target="_blank">License</a></li>
            </ul>
            </div>
            <div class="footer-column">
                <h3>Stay Updated</h3>
                <p>Subscribe to our newsletter for the latest updates, tips, and resources.</p>
                <div class="newsletter-form">
                    <input type="email" placeholder="Your email">
                    <button>Subscribe</button>
                </div>
                <p class="newsletter-note">By subscribing, you agree to our Privacy Policy and consent to receive updates from our company.</p>
            </div>
        </div>
        <div class="footer-bottom">
            <span>© 2025 QueryQuack. All rights reserved.</span>
            <div>
                <a href="#">Privacy Policy</a>
                <a href="#">Terms of Service</a>
                <a href="#">Cookie Policy</a>
            </div>
        </div>
    </div>
    """
    return footer_html