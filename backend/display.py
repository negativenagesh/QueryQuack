import streamlit as st

def render_hero_section():
    """Render the hero section for QueryQuack."""
    hero_html = """
    <div class="hero-text">
        <h1>QueryQuack</h1>
        <h2>Quack the Query, Crack the PDF!</h2>
        <p>Your AI-powered platform to unlock insights from PDFs with lightning-fast retrieval and smart answers.</p>
        <div class="hero-buttons">
            <a href="#powerful_features" class="hero-button explore-btn" onclick="parent.location='#powerful_features'">Discover Features</a>
            <a href="#how_queryquack_works" class="hero-button how-it-works-btn" onclick="parent.location='#how_queryquack_works'">How It Works</a>
            <a href="?page=upload" class="hero-button explore-btn" style="margin-top: 10px;" onclick="window.location.href='?page=upload'; return false;">Start Quacking</a>
        </div>
        <div class="hero-tags">
            <span class="hero-tag">PDF-Powered</span>
            <span class="hero-tag">AI-Driven</span>
            <span class="hero-tag">Instant Insights</span>
        </div>
    </div>
    """
    st.markdown(hero_html, unsafe_allow_html=True)

def display_results(answer, chunks, confidence_scores=None):
    """Display the answer and source information."""
    st.subheader("Answer:")
    st.write(answer)
    
    if chunks:
        st.subheader("Sources:")
        for i, (pdf, chunk, text) in enumerate(chunks):
            confidence = confidence_scores[i] if confidence_scores else "N/A"
            with st.expander(f"{pdf} (Chunk {chunk}, Confidence: {confidence})"):
                st.write(text)
                # Placeholder for PDF viewer (requires additional libraries like streamlit-pdf-viewer)
                st.write(f"View {pdf} in external viewer (not implemented).")
    else:
        st.warning("No sources found.")