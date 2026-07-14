"""
🔍 Semantic Search Engine — Premium Streamlit Application
AI-powered vector similarity search over your documents.
"""

import streamlit as st
import time

from rag_pipeline import RAGPipeline
from config import APP_TITLE, TOP_K, CHUNK_SIZE

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Semantic Search Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Root Variables ── */
:root {
    --bg-primary: #0E1117;
    --bg-secondary: #161B22;
    --bg-card: #1C2333;
    --accent-start: #667EEA;
    --accent-end: #764BA2;
    --teal: #38B2AC;
    --text-primary: #E6EDF3;
    --text-secondary: #8B949E;
    --border-color: #30363D;
    --success: #2DD4BF;
    --warning: #FBBF24;
    --danger: #F87171;
    --shadow-glow: 0 0 30px rgba(102,126,234,0.15);
}

/* ── Global ── */
html, body, .stApp, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: #30363D; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-start); }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1117 0%, #161B22 100%) !important;
    border-right: 1px solid var(--border-color) !important;
}
[data-testid="stSidebar"] .stMarkdown h2 {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--text-secondary);
    margin-top: 1.2rem;
}

/* ── Gradient Text ── */
.gradient-text {
    background: linear-gradient(135deg, var(--accent-start), var(--accent-end), var(--teal));
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientShift 6s ease infinite;
    font-weight: 800;
}
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ── Hero Section ── */
.hero-container {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    animation: fadeInDown 0.8s ease-out;
}
.hero-container h1 { font-size: 2.8rem; margin-bottom: 0.3rem; }
.hero-subtitle {
    color: var(--text-secondary);
    font-size: 1.05rem;
    font-weight: 400;
    margin-top: 0;
}
@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-18px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Search Bar ── */
.search-wrapper { max-width: 720px; margin: 0 auto 1.8rem; }
.search-wrapper input[type="text"],
div[data-testid="stTextInput"] input {
    background: var(--bg-card) !important;
    border: 1.5px solid var(--border-color) !important;
    border-radius: 14px !important;
    padding: 0.85rem 1.2rem !important;
    font-size: 1.05rem !important;
    color: var(--text-primary) !important;
    transition: border-color 0.3s, box-shadow 0.3s !important;
    font-family: 'Inter', sans-serif !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: var(--accent-start) !important;
    box-shadow: 0 0 0 3px rgba(102,126,234,0.25) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-start), var(--accent-end)) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.6rem 2rem !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 0.3px;
    transition: transform 0.2s, box-shadow 0.25s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(102,126,234,0.35) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Result Card ── */
.result-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    animation: fadeInUp 0.5s ease-out both;
}
.result-card:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow-glow);
    border-color: var(--accent-start);
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* stagger children */
.result-card:nth-child(1) { animation-delay: 0.05s; }
.result-card:nth-child(2) { animation-delay: 0.12s; }
.result-card:nth-child(3) { animation-delay: 0.19s; }
.result-card:nth-child(4) { animation-delay: 0.26s; }
.result-card:nth-child(5) { animation-delay: 0.33s; }

.card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.7rem;
}
.card-source {
    font-size: 0.8rem;
    color: var(--text-secondary);
    display: flex;
    align-items: center;
    gap: 0.45rem;
}
.card-source span { opacity: 0.7; }
.card-snippet {
    color: var(--text-primary);
    font-size: 0.93rem;
    line-height: 1.65;
    margin-top: 0.4rem;
}

/* ── Score Badge ── */
.score-badge {
    display: inline-block;
    padding: 0.22rem 0.7rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.5px;
}
.score-high   { background: rgba(45,212,191,0.15); color: var(--success); border: 1px solid rgba(45,212,191,0.3); }
.score-medium { background: rgba(251,191,36,0.12); color: var(--warning); border: 1px solid rgba(251,191,36,0.25); }
.score-low    { background: rgba(248,113,113,0.12); color: var(--danger); border: 1px solid rgba(248,113,113,0.25); }

/* ── Metric Cards ── */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 1rem 1.1rem;
    text-align: center;
    transition: border-color 0.3s;
}
.metric-card:hover { border-color: var(--accent-start); }
.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent-start), var(--teal));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: var(--text-secondary);
    margin-top: 0.25rem;
}

/* ── Status Pill ── */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem 0.85rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
}
.status-active { background: rgba(45,212,191,0.12); color: var(--success); }
.status-empty  { background: rgba(139,148,158,0.12); color: var(--text-secondary); }

/* ── Divider ── */
.styled-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border-color), transparent);
    border: none;
    margin: 1.4rem 0;
}

/* ── Empty State ── */
.empty-state {
    text-align: center;
    padding: 3.5rem 1rem;
    animation: fadeInUp 0.6s ease-out;
}
.empty-state .icon { font-size: 3.2rem; margin-bottom: 0.8rem; }
.empty-state h3 { color: var(--text-primary); font-weight: 600; margin-bottom: 0.4rem; }
.empty-state p  { color: var(--text-secondary); font-size: 0.92rem; max-width: 440px; margin: 0 auto; }

/* ── Sidebar brand ── */
.sidebar-brand {
    text-align: center;
    padding: 0.8rem 0 0.2rem;
}
.sidebar-brand h1 {
    font-size: 1.35rem;
    margin: 0;
}
.sidebar-brand p {
    font-size: 0.72rem;
    color: var(--text-secondary);
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-top: 0.2rem;
}

/* ── Expander Tweaks ── */
details[data-testid="stExpander"] {
    border-color: var(--border-color) !important;
    border-radius: 12px !important;
    background: var(--bg-card) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] section {
    border: 1.5px dashed var(--border-color) !important;
    border-radius: 12px !important;
    transition: border-color 0.3s;
}
[data-testid="stFileUploader"] section:hover {
    border-color: var(--accent-start) !important;
}

/* ── Hide default header / footer ── */
header[data-testid="stHeader"] { background: transparent !important; }
footer { display: none !important; }

/* ── Toast / Alerts ── */
div[data-testid="stAlert"] {
    border-radius: 12px !important;
    font-family: 'Inter', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Session State Initialisation
# ──────────────────────────────────────────────
if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline()
if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

pipeline: RAGPipeline = st.session_state.pipeline


# ──────────────────────────────────────────────
# Helper – score badge
# ──────────────────────────────────────────────
def _score_badge(score: float) -> str:
    """Return an HTML score badge with colour coding."""
    pct = f"{score * 100:.1f}%"
    if score >= 0.7:
        cls = "score-high"
    elif score >= 0.5:
        cls = "score-medium"
    else:
        cls = "score-low"
    return f'<span class="score-badge {cls}">{pct} match</span>'


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    # Brand
    st.markdown(
        '<div class="sidebar-brand">'
        '<h1><span class="gradient-text">🔍 Semantic Search</span></h1>'
        '<p>embedding-powered RAG</p>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    # ── Upload Documents ──
    st.markdown("## 📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Drop your files here",
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button("🚀 Process & Index", use_container_width=True):
            st.session_state.is_processing = True
            try:
                with st.spinner("Chunking & embedding documents…"):
                    result = pipeline.ingest_uploaded_files(uploaded_files)
                st.success(
                    f"✅ Indexed **{result['num_documents']}** doc(s) → **{result['num_chunks']}** chunks"
                )
            except Exception as exc:
                st.error(f"Ingestion failed: {exc}")
            finally:
                st.session_state.is_processing = False

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    # ── Settings ──
    with st.expander("⚙️ Settings", expanded=False):
        top_k = st.slider(
            "Number of results",
            min_value=1,
            max_value=20,
            value=TOP_K,
            help="How many matching chunks to return.",
        )
        chunk_size = st.slider(
            "Chunk size (tokens)",
            min_value=200,
            max_value=1000,
            value=CHUNK_SIZE,
            step=50,
            help="Target token count per chunk during ingestion.",
        )

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    # ── Collection Stats ──
    st.markdown("## 📊 Collection Stats")
    try:
        stats = pipeline.get_stats()
    except Exception:
        stats = {"total_chunks": 0, "total_documents": 0, "index_loaded": False}

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{stats["total_documents"]}</div>'
            f'<div class="metric-label">Documents</div></div>',
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{stats["total_chunks"]}</div>'
            f'<div class="metric-label">Chunks</div></div>',
            unsafe_allow_html=True,
        )

    idx_status = stats.get("index_loaded", False)
    pill_cls = "status-active" if idx_status else "status-empty"
    pill_icon = "🟢" if idx_status else "⚪"
    pill_label = "Index Loaded" if idx_status else "No Index"
    st.markdown(
        f'<div style="text-align:center;margin-top:0.7rem;">'
        f'<span class="status-pill {pill_cls}">{pill_icon} {pill_label}</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    # ── Clear Index ──
    if st.button("🗑️ Clear Index", use_container_width=True):
        try:
            pipeline.clear_index()
            st.session_state.search_results = []
            st.success("Index cleared.")
        except Exception as exc:
            st.error(f"Could not clear index: {exc}")


# ──────────────────────────────────────────────
# Main Area – Hero
# ──────────────────────────────────────────────
st.markdown(
    '<div class="hero-container">'
    '<h1 class="gradient-text">🔍 Semantic Search Engine</h1>'
    '<p class="hero-subtitle">'
    "Find answers using AI-powered vector similarity — not just keywords"
    "</p></div>",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Search Bar
# ──────────────────────────────────────────────
search_col_l, search_col_m, search_col_r = st.columns([1, 6, 1])
with search_col_m:
    query = st.text_input(
        "Search your documents…",
        placeholder="e.g.  What is retrieval-augmented generation?",
        label_visibility="collapsed",
        key="search_input",
    )
    btn_col1, btn_col2, btn_col3 = st.columns([2, 1, 2])
    with btn_col2:
        search_clicked = st.button("Search", use_container_width=True)

# ──────────────────────────────────────────────
# Execute Search
# ──────────────────────────────────────────────
if search_clicked and query.strip():
    if not stats.get("index_loaded", False):
        st.warning("⚠️ No index loaded. Upload and process documents first.")
    else:
        try:
            with st.spinner("Searching across your knowledge base…"):
                results = pipeline.search(query.strip(), k=top_k)
            st.session_state.search_results = results
        except Exception as exc:
            st.error(f"Search failed: {exc}")
            st.session_state.search_results = []

# ──────────────────────────────────────────────
# Results
# ──────────────────────────────────────────────
results = st.session_state.search_results

if results:
    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        f"<p style='text-align:center;color:var(--text-secondary);font-size:0.88rem;'>"
        f"Showing <strong>{len(results)}</strong> result{'s' if len(results) != 1 else ''}</p>",
        unsafe_allow_html=True,
    )

    for idx, res in enumerate(results):
        score = res.get("score", 0.0)
        source = res.get("source", "unknown")
        page = res.get("page", "–")
        content = res.get("content", "")
        snippet = content[:300] + ("…" if len(content) > 300 else "")

        badge_html = _score_badge(score)

        card_html = f"""
        <div class="result-card" style="animation-delay:{idx * 0.07}s">
            <div class="card-header">
                <div class="card-source">
                    📄 <strong>{source}</strong> <span>·</span> Page {page}
                </div>
                {badge_html}
            </div>
            <div class="card-snippet">{snippet}</div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

        # Full content expander
        if len(content) > 300:
            with st.expander(f"📖 Read full chunk #{idx + 1}"):
                st.markdown(content)

elif search_clicked and query.strip():
    # Search was executed but returned nothing
    st.markdown(
        '<div class="empty-state">'
        '<div class="icon">🔎</div>'
        "<h3>No matching results</h3>"
        "<p>Try rephrasing your query or uploading more documents to expand the knowledge base.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
else:
    # Initial / idle state
    st.markdown(
        '<div class="empty-state">'
        '<div class="icon">📚</div>'
        "<h3>Ready to search</h3>"
        "<p>Upload your documents in the sidebar, then type a question above to find semantically relevant passages.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center;font-size:0.72rem;color:var(--text-secondary);letter-spacing:1px;">'
    "BUILT WITH STREAMLIT  ·  POWERED BY VECTOR EMBEDDINGS  ·  RAG PIPELINE"
    "</p>",
    unsafe_allow_html=True,
)
