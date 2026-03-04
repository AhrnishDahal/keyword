# Semantic Keyword Extraction — Streamlit App

Upload a PDF and extract semantically meaningful keywords using LSA embeddings,
also tried and compared side-by-side with a TF-IDF baseline.

---

##  Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the app
streamlit run app.py
```

Opens automatically at `http://localhost:8501`.

---

## File Structure

```
keyword_extraction/
├── app.py                  ← Streamlit UI (run this)
├── keyword_extractor.py    ← Core pipeline (TextPreprocessor, LSAEmbedder, etc.)
├── run_extraction.py       ← CLI runner (no UI, for scripting)
├── tests.py                ← Unit tests (37 tests)
├── requirements.txt        ← Python dependencies
└── README.md
```

---

## Features

| Section | What you get |
|---|---|
| Document Overview | Page count, word count, sentences, avg word length, lexical diversity |
| Pipeline Statistics | Candidates generated, keywords extracted, LSA explained variance |
| Semantic Keywords tab | Ranked keywords with cosine similarity score bars |
| TF-IDF Baseline tab | Classic TF-IDF scores for comparison |
| Visualisations tab | Bar charts, scatter plot, histogram, overlap chart |
| Comparison tab | Side-by-side stats table, shared vs unique keywords |
| Raw Data tab | Full candidate list, exportable CSV |

---

## Sidebar Controls

| Setting | Default | Description |
|---|---|---|
| Top-K keywords | 10 | How many keywords to return |
| N-gram range | 1–3 | Min/max phrase length |
| LSA dimensions | 50 | Embedding space size |
| Max characters | 20,000 | Truncation limit for large PDFs |

---

##   Pipeline 

```
PDF Upload
    │
    ▼
Text Extraction (pypdf)
    │
    ▼
Candidate Generation  ← n-grams (1–3 words), stop-word edge filtering
    │
    ▼
LSA Embedding Model   ← TF-IDF matrix → TruncatedSVD → L2-normalised vectors
    │
  ┌─┴──────────────┐
  ▼                ▼
Document        Candidate
Embedding       Embeddings
  │                │
  └────────┬───────┘
           ▼
  Cosine Similarity
           │
           ▼
   Keyword Ranking → Top-K Semantic Keywords
           │
  TF-IDF Baseline  ← parallel comparison track
           │
           ▼
  Results + Charts + Export
```

---

## Running Tests

```bash
python tests.py
# 37 tests · should all pass
```

---



