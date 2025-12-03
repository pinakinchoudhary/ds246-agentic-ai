"""
Streamlit interface for autocomplete system.
Beautiful, interactive UI for testing the retrieval engine.
"""
import streamlit as st
import time
from typing import List
import pandas as pd

from config import *
from retrieval_engine import RetrievalEngine

# Page config
st.set_page_config(
    page_title=STREAMLIT_TITLE,
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .result-card {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        border-left: 4px solid #1f77b4;
    }
    .result-number {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .result-query {
        font-size: 1.2rem;
        font-weight: 500;
        margin-left: 1rem;
    }
    .result-score {
        font-size: 0.9rem;
        color: #666;
        margin-left: 1rem;
    }
    .stats-box {
        padding: 1rem;
        background-color: #e7f3ff;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_engine():
    """Load retrieval engine (cached)."""
    with st.spinner("🚀 Loading retrieval engine... This may take a minute..."):
        engine = RetrievalEngine()
    return engine


def format_result(rank: int, query: str, score: float = None):
    """Format a single result nicely."""
    score_text = f"<span class='result-score'>Score: {score:.4f}</span>" if score is not None else ""
    
    html = f"""
    <div class='result-card'>
        <span class='result-number'>{rank}.</span>
        <span class='result-query'>{query}</span>
        {score_text}
    </div>
    """
    return html


def main():
    # Header
    st.markdown("<h1 class='main-header'>🔍 Smart Autocomplete System</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='sub-header'>Powered by Semantic Search + BM25 + LightGBM Reranking</p>",
        unsafe_allow_html=True
    )
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        show_scores = st.checkbox(
            "Show scores",
            value=STREAMLIT_SHOW_SCORES,
            help="Display ranking scores for each result"
        )
        
        show_stats = st.checkbox(
            "Show retrieval stats",
            value=True,
            help="Display timing and performance statistics"
        )
    
    # Load engine
    try:
        engine = load_engine()
        st.success("✅ Retrieval engine loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading engine: {e}")
        st.stop()
    
    # Main input area
    st.markdown("---")
    st.header("🔎 Enter Your Search Prefix")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        prefix = st.text_input(
            "Type your search query:",
            placeholder="e.g., ifon 16, blaek blaijer, running sho...",
            label_visibility="collapsed",
            key="search_input"
        )
    
    with col2:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    # Perform search
    if prefix and (search_button or prefix):
        st.markdown("---")
        st.header(f"📋 Results for: *\"{prefix}\"*")
        
        # Retrieve results
        start_time = time.time()
        
        try:
            if show_scores:
                results, scores = engine.retrieve(prefix, return_scores=True)
            else:
                results = engine.retrieve(prefix)
                scores = [None] * len(results)
            
            elapsed_ms = (time.time() - start_time) * 1000
            
        except Exception as e:
            st.error(f"Error during retrieval: {e}")
            st.stop()
        
        # Display stats
        if show_stats:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("⏱️ Latency", f"{elapsed_ms:.2f} ms")
            with col2:
                st.metric("📊 Results", len(results))
            with col3:
                st.metric("📝 Prefix Length", len(prefix))
            with col4:
                cache_status = "HIT" if prefix in engine.cache else "MISS"
                st.metric("💾 Cache", cache_status)
        
        st.markdown("---")
        
        # Display results
        if results:
            st.subheader(f"🎯 Top {len(results)} Completions")
            
            for i, (query, score) in enumerate(zip(results, scores), 1):
                if show_scores and score is not None:
                    st.markdown(format_result(i, query, score), unsafe_allow_html=True)
                else:
                    st.markdown(format_result(i, query), unsafe_allow_html=True)
        else:
            st.warning("⚠️ No results found. Try a different prefix!")
        
        # Additional analysis
        if results and show_stats:
            st.markdown("---")
            st.subheader("📈 Result Analysis")
            
            # Create dataframe for analysis
            analysis_data = []
            for i, query in enumerate(results):
                from data_loader import normalize_text
                norm_prefix = normalize_text(prefix)
                norm_query = normalize_text(query)
                
                analysis_data.append({
                    "Rank": i + 1,
                    "Query": query,
                    "Length": len(query),
                    "Prefix Match": norm_query.startswith(norm_prefix),
                    "Score": scores[i] if scores[i] is not None else 0
                })
            
            df = pd.DataFrame(analysis_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(df, hide_index=True, use_container_width=True)
            
            with col2:
                # Simple statistics
                st.write("**Statistics:**")
                st.write(f"- Avg query length: {df['Length'].mean():.1f} chars")
                st.write(f"- Prefix matches: {df['Prefix Match'].sum()}/{len(df)}")
                if show_scores:
                    st.write(f"- Avg score: {df['Score'].mean():.4f}")
                    st.write(f"- Score range: {df['Score'].min():.4f} - {df['Score'].max():.4f}")


if __name__ == "__main__":
    main()