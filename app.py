"""
app.py — Streamlit UI for Semantic Keyword Extraction
======================================================
Run with:
    streamlit run app.py

Requires:
    pip install streamlit pypdf scikit-learn numpy matplotlib pandas
"""

import io
import re
import math
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import streamlit as st
from pypdf import PdfReader

# ── import pipeline ──────────────────────────────────────────────────────────
from keyword_extractor import KeywordExtractionPipeline


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Semantic Keyword Extractor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS  — dark research-tool aesthetic
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Main background */
.stApp {
    background-color: #0d1117;
    color: #e6edf3;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 16px;
}
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace;
    color: #58a6ff !important;
    font-size: 2rem !important;
}
[data-testid="stMetricLabel"] {
    color: #8b949e !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Headers */
h1 { color: #e6edf3 !important; font-family: 'IBM Plex Sans', sans-serif !important; font-weight: 600 !important; }
h2 { color: #c9d1d9 !important; font-family: 'IBM Plex Sans', sans-serif !important; font-weight: 600 !important; }
h3 { color: #8b949e !important; font-family: 'IBM Plex Mono', monospace !important; }

/* Keyword badge rows */
.kw-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    font-weight: 600;
    margin: 3px;
}
.kw-semantic { background: #1f3a5f; color: #58a6ff; border: 1px solid #1f6feb; }
.kw-tfidf   { background: #2d1f3a; color: #c084fc; border: 1px solid #7c3aed; }
.kw-shared  { background: #1f3a2d; color: #3fb950; border: 1px solid #238636; }

/* Dataframe */
[data-testid="stDataFrame"] { border: 1px solid #30363d; border-radius: 8px; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: #161b22; border-bottom: 1px solid #30363d; }
.stTabs [data-baseweb="tab"] { color: #8b949e; font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; }
.stTabs [aria-selected="true"] { color: #58a6ff !important; border-bottom: 2px solid #58a6ff !important; }

/* File uploader */
[data-testid="stFileUploader"] {
    border: 2px dashed #30363d;
    border-radius: 12px;
    padding: 8px;
    background: #161b22;
}

/* Buttons */
.stButton > button {
    background: #1f6feb;
    color: white;
    border: none;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    letter-spacing: 0.03em;
    padding: 10px 24px;
}
.stButton > button:hover { background: #388bfd; }

/* Code blocks */
code { font-family: 'IBM Plex Mono', monospace !important; color: #79c0ff; }

/* Divider */
hr { border-color: #30363d; }

/* Info / warning boxes */
.stInfo    { background: #1f3a5f; border-left: 4px solid #58a6ff; }
.stWarning { background: #3a2d1f; border-left: 4px solid #d29922; }
.stSuccess { background: #1f3a2d; border-left: 4px solid #3fb950; }

/* Section label */
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.70rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 8px;
}

/* Score bar container */
.score-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 6px 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
}
.score-bar-bg {
    flex: 1;
    background: #21262d;
    border-radius: 4px;
    height: 8px;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.6s ease;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

DARK_BG   = "#0d1117"
PANEL_BG  = "#161b22"
BORDER    = "#30363d"
BLUE      = "#58a6ff"
PURPLE    = "#c084fc"
GREEN     = "#3fb950"
ORANGE    = "#e3b341"
RED       = "#f85149"
TEXT_DIM  = "#8b949e"
TEXT_MAIN = "#e6edf3"


def extract_text_from_pdf(uploaded_file) -> tuple[str, int, dict]:
    """Return (full_text, page_count, metadata) from uploaded PDF."""
    reader = PdfReader(uploaded_file)
    pages  = reader.pages
    texts  = []
    for page in pages:
        t = page.extract_text() or ""
        texts.append(t)
    full_text = "\n\n".join(texts)
    meta = reader.metadata or {}
    return full_text, len(pages), {
        "title":   getattr(meta, "title",   None) or "—",
        "author":  getattr(meta, "author",  None) or "—",
        "subject": getattr(meta, "subject", None) or "—",
    }


def word_count(text: str) -> int:
    return len(text.split())


def sentence_count(text: str) -> int:
    return len(re.split(r"[.!?]+", text))


def avg_word_length(text: str) -> float:
    words = [w for w in text.split() if w.isalpha()]
    if not words:
        return 0.0
    return round(sum(len(w) for w in words) / len(words), 2)


def lexical_diversity(text: str) -> float:
    words = [w.lower() for w in text.split() if w.isalpha()]
    if not words:
        return 0.0
    return round(len(set(words)) / len(words), 4)


# ─────────────────────────────────────────────────────────────────────────────
# CHART BUILDERS  (matplotlib, dark-theme)
# ─────────────────────────────────────────────────────────────────────────────

def _dark_fig(w=10, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(PANEL_BG)
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.tick_params(colors=TEXT_DIM, labelsize=9)
    ax.xaxis.label.set_color(TEXT_DIM)
    ax.yaxis.label.set_color(TEXT_DIM)
    ax.title.set_color(TEXT_MAIN)
    return fig, ax


def chart_similarity_bars(semantic_kws, tfidf_kws):
    """Side-by-side horizontal bar chart: semantic vs TF-IDF scores."""
    sem_labels  = [k for k, _ in semantic_kws]
    sem_scores  = [s for _, s in semantic_kws]
    tfi_labels  = [k for k, _ in tfidf_kws]
    tfi_scores  = [s for _, s in tfidf_kws]
    n = max(len(sem_labels), len(tfi_labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(4, n * 0.55 + 1.5)))
    fig.patch.set_facecolor(PANEL_BG)
    fig.suptitle("Keyword Scores Comparison", color=TEXT_MAIN,
                 fontsize=13, fontweight="bold", y=1.01)

    for ax, labels, scores, color, title in [
        (ax1, sem_labels, sem_scores, BLUE,   "🧠 Semantic (LSA)"),
        (ax2, tfi_labels, tfi_scores, PURPLE, "📊 TF-IDF Baseline"),
    ]:
        ax.set_facecolor(PANEL_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        bars = ax.barh(labels[::-1], scores[::-1], color=color, alpha=0.85,
                       height=0.6, edgecolor="none")
        # Value labels
        for bar, score in zip(bars, scores[::-1]):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{score:.4f}", va="center", ha="left",
                    color=TEXT_MAIN, fontsize=8,
                    fontfamily="monospace")
        ax.set_title(title, color=TEXT_MAIN, fontsize=11, pad=10)
        ax.tick_params(colors=TEXT_DIM, labelsize=8.5)
        ax.set_xlabel("Score", color=TEXT_DIM, fontsize=9)
        ax.set_xlim(0, max(scores) * 1.25 if scores else 1)
        ax.axvline(0, color=BORDER, linewidth=0.8)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)

    plt.tight_layout()
    return fig


def chart_overlap_venn_like(sem_set, tfi_set):
    """Stacked bar showing shared vs unique keywords."""
    only_sem = len(sem_set - tfi_set)
    shared   = len(sem_set & tfi_set)
    only_tfi = len(tfi_set - sem_set)
    total    = only_sem + shared + only_tfi or 1

    fig, ax = plt.subplots(figsize=(9, 2.2))
    fig.patch.set_facecolor(PANEL_BG)
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    segments = [
        (only_sem / total, BLUE,   f"Semantic only  ({only_sem})"),
        (shared   / total, GREEN,  f"Shared  ({shared})"),
        (only_tfi / total, PURPLE, f"TF-IDF only  ({only_tfi})"),
    ]
    x = 0
    for frac, color, label in segments:
        ax.barh(0, frac, left=x, color=color, alpha=0.85, height=0.55, edgecolor=PANEL_BG, linewidth=1.5)
        if frac > 0.05:
            ax.text(x + frac / 2, 0, label, ha="center", va="center",
                    color="white", fontsize=8.5, fontweight="bold",
                    fontfamily="monospace")
        x += frac

    ax.set_title("Keyword Overlap — Semantic vs TF-IDF", color=TEXT_MAIN,
                 fontsize=11, pad=12)
    ax.set_xlim(0, 1)
    plt.tight_layout()
    return fig


def chart_candidate_ngram_dist(candidates):
    """Bar chart: distribution of 1-gram, 2-gram, 3-gram candidates."""
    dist = {1: 0, 2: 0, 3: 0}
    for c in candidates:
        n = len(c.split())
        if n in dist:
            dist[n] += 1
        else:
            dist[3] += 1

    fig, ax = _dark_fig(6, 3.5)
    labels = ["Unigram\n(1 word)", "Bigram\n(2 words)", "Trigram\n(3 words)"]
    values = [dist[1], dist[2], dist[3]]
    colors = [BLUE, ORANGE, GREEN]
    bars = ax.bar(labels, values, color=colors, alpha=0.85, width=0.5, edgecolor="none")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom",
                color=TEXT_MAIN, fontsize=10, fontfamily="monospace")
    ax.set_title("Candidate Phrase Distribution", color=TEXT_MAIN, fontsize=11, pad=10)
    ax.set_ylabel("Count", color=TEXT_DIM, fontsize=9)
    ax.set_ylim(0, max(values) * 1.2 + 2)
    ax.axhline(0, color=BORDER, linewidth=0.5)
    plt.tight_layout()
    return fig


def chart_score_scatter(semantic_kws):
    """Dot plot: semantic score per keyword with rank annotation."""
    if not semantic_kws:
        return None
    labels = [k for k, _ in semantic_kws]
    scores = [s for _, s in semantic_kws]
    ranks  = list(range(1, len(labels) + 1))

    fig, ax = _dark_fig(10, 4)
    scatter = ax.scatter(ranks, scores, c=scores, cmap="Blues",
                         s=120, zorder=3, edgecolors=BLUE, linewidths=0.8)
    ax.plot(ranks, scores, color=BORDER, linewidth=1, zorder=2, linestyle="--")
    for r, s, label in zip(ranks, scores, labels):
        ax.annotate(label, (r, s), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=7.5,
                    color=TEXT_MAIN, fontfamily="monospace",
                    rotation=30)
    ax.set_xlabel("Rank", color=TEXT_DIM)
    ax.set_ylabel("Cosine Similarity Score", color=TEXT_DIM)
    ax.set_title("Semantic Score by Keyword Rank", color=TEXT_MAIN, fontsize=11, pad=14)
    ax.set_xticks(ranks)
    ax.set_xticklabels([f"#{r}" for r in ranks], color=TEXT_DIM, fontsize=8)
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.ax.tick_params(colors=TEXT_DIM, labelsize=8)
    cbar.set_label("Score", color=TEXT_DIM, fontsize=8)
    cbar.ax.yaxis.set_tick_params(color=TEXT_DIM)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_DIM)
    plt.tight_layout()
    return fig


def chart_score_distribution(semantic_kws, tfidf_kws):
    """Histogram of score distributions for both methods."""
    fig, ax = _dark_fig(8, 3.5)
    sem_scores = [s for _, s in semantic_kws]
    tfi_scores = [s for _, s in tfidf_kws]
    bins = 8
    ax.hist(sem_scores, bins=bins, alpha=0.7, color=BLUE,   label="Semantic", edgecolor=PANEL_BG)
    ax.hist(tfi_scores, bins=bins, alpha=0.7, color=PURPLE, label="TF-IDF",   edgecolor=PANEL_BG)
    ax.set_xlabel("Score", color=TEXT_DIM)
    ax.set_ylabel("Frequency", color=TEXT_DIM)
    ax.set_title("Score Distribution", color=TEXT_MAIN, fontsize=11, pad=10)
    ax.legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT_MAIN, fontsize=9)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='font-family: IBM Plex Mono, monospace; font-size: 1.15rem; 
                font-weight: 600; color: #58a6ff; margin-bottom: 4px;'>
        🔍 Keyword Extractor
    </div>
    <div style='font-size: 0.75rem; color: #8b949e; margin-bottom: 24px;'>
        Semantic · LSA · TF-IDF
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### ⚙️ Pipeline Settings")

    top_k = st.slider("Top-K keywords", min_value=3, max_value=20, value=10, step=1,
                       help="Number of keywords to extract from the document.")

    ngram_min, ngram_max = st.select_slider(
        "N-gram range",
        options=[1, 2, 3],
        value=(1, 3),
        help="Min and max phrase length in words."
    )

    embedding_dims = st.select_slider(
        "LSA embedding dimensions",
        options=[10, 20, 30, 50, 75, 100],
        value=50,
        help="Higher = richer semantic space, but needs larger vocab."
    )

    st.markdown("---")
    st.markdown("#### 📄 Text Options")

    max_chars = st.number_input(
        "Max characters to process",
        min_value=500, max_value=100_000, value=20_000, step=500,
        help="Truncates very long PDFs to keep processing fast."
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size: 0.72rem; color: #8b949e; line-height: 1.6;'>
        <b style='color:#c9d1d9;'>How it works</b><br>
        1. PDF text is extracted via <code>pypdf</code><br>
        2. N-gram candidates are generated<br>
        3. LSA (TF-IDF + SVD) embeds the document and all candidates<br>
        4. Cosine similarity ranks candidates by semantic relevance<br>
        5. TF-IDF baseline is computed for comparison
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<h1 style='font-family: IBM Plex Sans, sans-serif; font-size: 2rem; 
           font-weight: 600; margin-bottom: 0;'>
    Semantic Keyword Extraction
</h1>
<p style='color: #8b949e; font-size: 0.95rem; margin-top: 4px;'>
    Upload a PDF · Extract meaningful keywords · Compare semantic vs TF-IDF methods
</p>
<hr>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "Drop a PDF here or click to browse",
    type=["pdf"],
    help="Multi-page PDFs are supported. Text is extracted from all pages.",
)

if not uploaded_file:
    st.markdown("""
    <div style='text-align: center; padding: 60px 20px; color: #8b949e;'>
        <div style='font-size: 3rem; margin-bottom: 12px;'>📄</div>
        <div style='font-size: 1.1rem; font-weight: 600; color: #c9d1d9;'>No file uploaded yet</div>
        <div style='font-size: 0.88rem; margin-top: 8px;'>Upload a PDF using the panel above to begin extraction</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACT TEXT
# ─────────────────────────────────────────────────────────────────────────────

with st.spinner("📖 Reading PDF…"):
    try:
        raw_text, page_count, pdf_meta = extract_text_from_pdf(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read PDF: {e}")
        st.stop()

if not raw_text.strip():
    st.warning("⚠️ No extractable text found. The PDF may be scanned/image-only.")
    st.stop()

# Truncate if needed
if len(raw_text) > max_chars:
    raw_text = raw_text[:max_chars]
    st.info(f"ℹ️ Document truncated to {max_chars:,} characters. Adjust in sidebar if needed.")


# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENT STATS STRIP
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("### 📋 Document Overview")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Pages",            page_count)
m2.metric("Words",            f"{word_count(raw_text):,}")
m3.metric("Sentences",        f"{sentence_count(raw_text):,}")
m4.metric("Avg Word Length",  f"{avg_word_length(raw_text)} chars")
m5.metric("Lexical Diversity",f"{lexical_diversity(raw_text):.3f}")

with st.expander("📄 PDF Metadata", expanded=False):
    col_a, col_b, col_c = st.columns(3)
    col_a.markdown(f"**Title:** {pdf_meta['title']}")
    col_b.markdown(f"**Author:** {pdf_meta['author']}")
    col_c.markdown(f"**Subject:** {pdf_meta['subject']}")

with st.expander("📝 Extracted Text Preview", expanded=False):
    preview = raw_text[:3000] + ("…" if len(raw_text) > 3000 else "")
    st.text_area("", value=preview, height=220, disabled=True)

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# RUN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("### 🚀 Running Extraction Pipeline")

with st.spinner("⚙️ Building embeddings and ranking keywords…"):
    try:
        pipeline = KeywordExtractionPipeline(
            ngram_range=(ngram_min, ngram_max),
            embedding_dims=embedding_dims,
            top_k=top_k,
        )
        results = pipeline.run(raw_text)
    except ValueError as e:
        st.error(f"Pipeline error: {e}")
        st.stop()

semantic_kws = results["semantic"]
tfidf_kws    = results["tfidf"]
candidates   = results["candidates"]
expl_var     = results["explained_var"]

sem_set = {k.lower() for k, _ in semantic_kws}
tfi_set = {k.lower() for k, _ in tfidf_kws}
shared  = sem_set & tfi_set


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE STATS STRIP
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("### 📊 Pipeline Statistics")

p1, p2, p3, p4, p5 = st.columns(5)
p1.metric("Candidates Generated",  len(candidates))
p2.metric("Semantic Keywords",      len(semantic_kws))
p3.metric("TF-IDF Keywords",        len(tfidf_kws))
p4.metric("Shared Keywords",        len(shared))
p5.metric("LSA Explained Variance", f"{expl_var}%",
           delta="semantic richness",
           delta_color="normal")

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧠 Semantic Keywords",
    "📊 TF-IDF Baseline",
    "📈 Visualisations",
    "🔍 Comparison",
    "📋 Raw Data",
])


# ── TAB 1 : SEMANTIC ─────────────────────────────────────────────────────────

with tab1:
    st.markdown("#### Top Semantic Keywords  _(LSA embeddings + cosine similarity)_")
    st.caption("Keywords are ranked by their cosine similarity to the full document embedding.")

    max_score = max((s for _, s in semantic_kws), default=1)

    for rank, (kw, score) in enumerate(semantic_kws, 1):
        pct  = score / max_score if max_score else 0
        col_rank, col_kw, col_bar, col_score = st.columns([0.6, 3, 5, 1.4])
        with col_rank:
            st.markdown(f"<span style='color:{TEXT_DIM}; font-family: IBM Plex Mono, monospace; font-size:0.9rem;'>#{rank}</span>", unsafe_allow_html=True)
        with col_kw:
            st.markdown(f"<span class='kw-badge kw-semantic'>{kw}</span>", unsafe_allow_html=True)
        with col_bar:
            st.markdown(f"""
            <div style='margin-top:10px;'>
              <div class='score-bar-bg'>
                <div class='score-bar-fill' style='width:{pct*100:.1f}%; background:{BLUE};'></div>
              </div>
            </div>""", unsafe_allow_html=True)
        with col_score:
            st.markdown(f"<span style='font-family:IBM Plex Mono,monospace; color:{BLUE}; font-size:0.85rem;'>{score:.4f}</span>", unsafe_allow_html=True)

    st.markdown("---")

    # Summary stats for semantic scores
    sem_scores_arr = np.array([s for _, s in semantic_kws])
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Max score",  f"{sem_scores_arr.max():.4f}")
    s2.metric("Min score",  f"{sem_scores_arr.min():.4f}")
    s3.metric("Mean score", f"{sem_scores_arr.mean():.4f}")
    s4.metric("Std dev",    f"{sem_scores_arr.std():.4f}")


# ── TAB 2 : TF-IDF ───────────────────────────────────────────────────────────

with tab2:
    st.markdown("#### TF-IDF Baseline Keywords")
    st.caption("Classic term-frequency × inverse document-frequency scoring across sentences.")

    max_tfi = max((s for _, s in tfidf_kws), default=1)

    for rank, (kw, score) in enumerate(tfidf_kws, 1):
        pct = score / max_tfi if max_tfi else 0
        col_rank, col_kw, col_bar, col_score = st.columns([0.6, 3, 5, 1.4])
        with col_rank:
            st.markdown(f"<span style='color:{TEXT_DIM}; font-family: IBM Plex Mono, monospace; font-size:0.9rem;'>#{rank}</span>", unsafe_allow_html=True)
        with col_kw:
            css_class = "kw-shared" if kw.lower() in shared else "kw-tfidf"
            st.markdown(f"<span class='kw-badge {css_class}'>{kw}</span>", unsafe_allow_html=True)
        with col_bar:
            st.markdown(f"""
            <div style='margin-top:10px;'>
              <div class='score-bar-bg'>
                <div class='score-bar-fill' style='width:{pct*100:.1f}%; background:{PURPLE};'></div>
              </div>
            </div>""", unsafe_allow_html=True)
        with col_score:
            st.markdown(f"<span style='font-family:IBM Plex Mono,monospace; color:{PURPLE}; font-size:0.85rem;'>{score:.4f}</span>", unsafe_allow_html=True)

    st.markdown("---")

    tfi_scores_arr = np.array([s for _, s in tfidf_kws])
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Max score",  f"{tfi_scores_arr.max():.4f}")
    t2.metric("Min score",  f"{tfi_scores_arr.min():.4f}")
    t3.metric("Mean score", f"{tfi_scores_arr.mean():.4f}")
    t4.metric("Std dev",    f"{tfi_scores_arr.std():.4f}")


# ── TAB 3 : VISUALISATIONS ────────────────────────────────────────────────────

with tab3:
    st.markdown("#### Keyword Score Charts")

    st.pyplot(chart_similarity_bars(semantic_kws, tfidf_kws))
    st.markdown("---")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**N-gram Candidate Distribution**")
        st.pyplot(chart_candidate_ngram_dist(candidates))
    with col_r:
        st.markdown("**Score Distribution Histogram**")
        st.pyplot(chart_score_distribution(semantic_kws, tfidf_kws))

    st.markdown("---")
    st.markdown("**Semantic Score by Rank** _(dot plot)_")
    fig_scatter = chart_score_scatter(semantic_kws)
    if fig_scatter:
        st.pyplot(fig_scatter)

    st.markdown("---")
    st.markdown("**Keyword Overlap**")
    st.pyplot(chart_overlap_venn_like(sem_set, tfi_set))


# ── TAB 4 : COMPARISON ───────────────────────────────────────────────────────

with tab4:
    st.markdown("#### Side-by-Side Comparison")

    only_sem = sem_set - tfi_set
    only_tfi = tfi_set - sem_set

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"**🧠 Semantic Only** `({len(only_sem)})`")
        for kw in sorted(only_sem):
            st.markdown(f"<span class='kw-badge kw-semantic'>{kw}</span>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"**✅ Shared** `({len(shared)})`")
        for kw in sorted(shared):
            st.markdown(f"<span class='kw-badge kw-shared'>{kw}</span>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"**📊 TF-IDF Only** `({len(only_tfi)})`")
        for kw in sorted(only_tfi):
            st.markdown(f"<span class='kw-badge kw-tfidf'>{kw}</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Quantitative Comparison Summary")

    sem_arr = np.array([s for _, s in semantic_kws])
    tfi_arr = np.array([s for _, s in tfidf_kws])

    comparison_df = pd.DataFrame({
        "Metric": [
            "Keywords extracted", "Mean score", "Max score",
            "Min score", "Score std dev", "Score range",
            "Unique to method", "Shared with other method",
        ],
        "Semantic (LSA)": [
            len(semantic_kws),
            f"{sem_arr.mean():.4f}",
            f"{sem_arr.max():.4f}",
            f"{sem_arr.min():.4f}",
            f"{sem_arr.std():.4f}",
            f"{sem_arr.max() - sem_arr.min():.4f}",
            len(only_sem),
            len(shared),
        ],
        "TF-IDF Baseline": [
            len(tfidf_kws),
            f"{tfi_arr.mean():.4f}",
            f"{tfi_arr.max():.4f}",
            f"{tfi_arr.min():.4f}",
            f"{tfi_arr.std():.4f}",
            f"{tfi_arr.max() - tfi_arr.min():.4f}",
            len(only_tfi),
            len(shared),
        ],
    })
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### LSA Model Interpretability")
    i1, i2, i3 = st.columns(3)
    i1.metric("Explained Variance", f"{expl_var}%",
               help="% of variance in the TF-IDF matrix captured by the LSA embedding.")
    i2.metric("Embedding Dimensions", embedding_dims,
               help="Number of latent semantic dimensions used.")
    i3.metric("Vocabulary Size (candidates)", len(candidates),
               help="Total candidate phrases fed to the embedding model.")


# ── TAB 5 : RAW DATA ─────────────────────────────────────────────────────────

with tab5:
    st.markdown("#### All Generated Candidates")
    cand_df = pd.DataFrame({
        "Phrase": candidates,
        "Words":  [len(c.split()) for c in candidates],
        "Chars":  [len(c) for c in candidates],
    })
    st.dataframe(cand_df, use_container_width=True, height=300)

    st.markdown("---")
    st.markdown("#### Semantic Keywords — Full Table")
    sem_df = pd.DataFrame(semantic_kws, columns=["Keyword", "Cosine Similarity"])
    sem_df.index += 1
    sem_df["Normalised Score (%)"] = (sem_df["Cosine Similarity"] / sem_df["Cosine Similarity"].max() * 100).round(1)
    st.dataframe(sem_df, use_container_width=True)

    st.markdown("---")
    st.markdown("#### TF-IDF Keywords — Full Table")
    tfi_df = pd.DataFrame(tfidf_kws, columns=["Keyword", "TF-IDF Score"])
    tfi_df.index += 1
    tfi_df["Normalised Score (%)"] = (tfi_df["TF-IDF Score"] / tfi_df["TF-IDF Score"].max() * 100).round(1)
    st.dataframe(tfi_df, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Export Results")
    export_df = pd.DataFrame({
        "Semantic Keyword": [k for k, _ in semantic_kws] + [""] * max(0, len(tfidf_kws) - len(semantic_kws)),
        "Semantic Score":   [s for _, s in semantic_kws]  + [""] * max(0, len(tfidf_kws) - len(semantic_kws)),
        "TF-IDF Keyword":  [k for k, _ in tfidf_kws]    + [""] * max(0, len(semantic_kws) - len(tfidf_kws)),
        "TF-IDF Score":    [s for _, s in tfidf_kws]    + [""] * max(0, len(semantic_kws) - len(tfidf_kws)),
    })
    csv = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️  Download Results as CSV",
        data=csv,
        file_name="keyword_extraction_results.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("""
<div style='text-align:center; color: #8b949e; font-size: 0.78rem; 
            font-family: IBM Plex Mono, monospace; padding: 8px;'>
    Semantic Keyword Extraction Pipeline · LSA Embeddings · TF-IDF Baseline · Built with Streamlit
</div>
""", unsafe_allow_html=True)
