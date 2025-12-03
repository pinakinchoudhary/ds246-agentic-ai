"""
Streamlit interface for autocomplete system.
Beautiful, interactive UI for testing the retrieval engine.
"""
import streamlit as st
import time
from typing import List
import pandas as pd

from config import *
from inference import predict

# Adapter to present a RetrievalEngine-like interface backed by inference.predict

# Page config
st.set_page_config(
    page_title="Autocomplete Retrieval Tester",
    page_icon="🔎",
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
        font-weight: bold;
        margin-right: 0.5rem;
    }
    .result-score {
        float: right;
        color: #888;
        font-size: 0.9rem;
    }
    .result-query {
        font-size: 1.1rem;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_engine():
    """Load a lightweight adapter that wraps the inferencing.predict function.
    The adapter exposes a `retrieve(prefix, return_scores=False)` method and a `cache` dict,
    to avoid changing any UI code that expects a RetrievalEngine-like object.
    """
    from inference import predict

    class InferenceAdapter:
        def __init__(self):
            # simple in-memory cache mapping prefix -> (results, scores)
            self.cache = {}

        def _score_results(self, prefix, results):
            # Produce simple, deterministic scores so the UI can display statistics.
            # Score in [0,1]. Prioritize exact prefix startswith matches.
            scores = []
            lp = len(prefix) if prefix else 0
            for q in results:
                ql = len(q) if q else 1
                q_low = q.lower()
                prefix_low = prefix.lower()
                # exact startswith
                if lp > 0 and q_low.startswith(prefix_low):
                    # score closer to 1 for shorter gap between prefix and full query
                    gap = max(ql - lp, 0)
                    score = 1.0 - (gap / max(ql, lp, 1)) * 0.3
                else:
                    # fallback: proportion of characters matched from start
                    common = 0
                    for a,b in zip(q_low, prefix_low):
                        if a==b:
                            common += 1
                        else:
                            break
                    frac = common / max(lp,1)
                    score = 0.4 + 0.6 * frac
                # clamp
                score = max(0.0, min(1.0, score))
                scores.append(float(score))
            return scores

        def retrieve(self, prefix, return_scores=False):
            # Return cached if present
            key = prefix or ""
            if key in self.cache:
                results, scores = self.cache[key]
            else:
                # use predict from inference.py; let predict use its defaults
                try:
                    results = predict(prefix)
                except Exception as e:
                    # If predict fails, propagate error to caller
                    raise
                scores = self._score_results(prefix or "", results)
                self.cache[key] = (results, scores)

            if return_scores:
                return results, scores
            else:
                return results

    return InferenceAdapter()


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


def get_ui_settings():
    """Return settings controlled from the sidebar."""
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
        
        max_results = st.slider(
            "Max results to display",
            min_value=5,
            max_value=50,
            value=STREAMLIT_MAX_RESULTS,
            step=1
        )
        
        prefix_filter = st.checkbox(
            "Enable 3-char prefix filter (faster)",
            value=True,
            help="Use prefix filtering in backend if supported"
        )
        
        custom_top_k = st.number_input(
            "Top-k (override)",
            min_value=1,
            max_value=100,
            value=0,
            help="If >0, override the backend's default top-k"
        )
        
    return {
        "show_scores": show_scores,
        "show_stats": show_stats,
        "max_results": max_results,
        "prefix_filter": prefix_filter,
        "custom_top_k": custom_top_k
    }


def search_ui(engine):
    """Main UI controls for entering a prefix and viewing results."""
    st.markdown("<div class='main-header'>Autocomplete Retrieval Tester</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Type prefixes to see predicted completions and ranking behavior</div>", unsafe_allow_html=True)
    
    settings = get_ui_settings()
    show_scores = settings["show_scores"]
    show_stats = settings["show_stats"]
    max_results = settings["max_results"]
    prefix_filter = settings["prefix_filter"]
    custom_top_k = settings["custom_top_k"]
    
    col1, col2 = st.columns([3,1])
    with col1:
        prefix = st.text_input("Prefix", value="", max_chars=200, placeholder="Type a search prefix (e.g. 'nike sho')")

    with col2:
        st.write("### Controls")
        if st.button("Clear cache"):
            engine.cache.clear()
            st.success("Cache cleared")

        st.write(f"Cache entries: {len(engine.cache)}")
        st.write(" ")
        st.write("Tip: toggle 'Show scores' to see numeric diagnostics")
    
    if prefix is None or prefix.strip() == "":
        st.info("Enter a prefix to display predictions.")
        return
    
    # Retrieve results
    st.header(f"📋 Results for: *\"{prefix}\"*")
    
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
    
    # Trim results to max_results or custom_top_k
    top_k = int(custom_top_k) if int(custom_top_k) > 0 else max_results
    results = results[:top_k]
    scores = scores[:top_k]
    
    # Display results
    for idx, q in enumerate(results, start=1):
        sc = scores[idx-1] if scores and idx-1 < len(scores) else None
        st.markdown(format_result(idx, q, sc), unsafe_allow_html=True)
    
    # Show timing and stats
    if show_stats:
        st.write(f"Retrieval time: {elapsed_ms:.1f} ms")
        st.write(f"Results returned: {len(results)}")
    
    # Detailed table view
    show_table = st.checkbox("Show table of results", value=False)
    if show_table:
        df = pd.DataFrame({
            "Rank": list(range(1, len(results) + 1)),
            "Query": results,
            "Score": [f"{s:.4f}" if s is not None else "" for s in scores],
            "Length": [len(q) for q in results],
            "Prefix Match": [1 if q.lower().startswith(prefix.lower()) else 0 for q in results]
        })
        col1, col2 = st.columns([2,1])
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


def main():
    # Sidebar info
    with st.sidebar:
        st.markdown("## About")
        st.markdown("This app tests the autocomplete retrieval system. The backend is the inferencing pipeline (inference.py).")
    
    # Load engine
    try:
        engine = load_engine()
        st.success("✅ Retrieval engine loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading engine: {e}")
        raise
    
    # Run UI
    search_ui(engine)


if __name__ == "__main__":
    main()
