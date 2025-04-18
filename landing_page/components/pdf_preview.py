import streamlit as st

def render_pdf_preview():
    pdf_html = """
    <div class="pdf-preview">
        <div class="pdf-header">
            <div class="pdf-icon">PDF</div>
            <div class="pdf-pages">1/3</div>
        </div>
        <div class="pdf-content">
            <div class="pdf-bar" style="width: 90%;"></div>
            <div class="pdf-placeholder"></div>
            <div class="pdf-placeholder"></div>
            <div class="pdf-placeholder" style="height: 50px;"></div>
            <div class="pdf-text-lines">
                <div></div>
                <div></div>
                <div></div>
            </div>
        </div>
    </div>
    """
    return pdf_html