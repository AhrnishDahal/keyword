"""
Semantic Keyword Extraction Pipeline
=====================================
Extracts keywords from text using:
  - Semantic embeddings via TF-IDF + LSA (Latent Semantic Analysis)
  - Cosine similarity ranking
  - TF-IDF baseline for comparison

Architecture mirrors a KeyBERT-style pipeline but runs fully offline
using scikit-learn's TruncatedSVD as the embedding model.
"""

import re
import math
import numpy as np
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from typing import List, Tuple, Optional


# ─────────────────────────────────────────────
# 1. TEXT PREPROCESSING
# ─────────────────────────────────────────────

class TextPreprocessor:
    """
    Cleans and normalises input text before embedding.
    """

    STOP_WORDS = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "shall",
        "this", "that", "these", "those", "it", "its", "as", "so", "if",
        "not", "no", "nor", "than", "then", "also", "such", "both",
        "each", "more", "most", "other", "some", "any", "all", "into",
        "through", "during", "about", "against", "between", "very",
        "can", "just", "how", "what", "which", "who", "when", "where",
        "their", "they", "them", "there", "our", "we", "us", "you",
        "your", "he", "she", "his", "her", "i", "me", "my",
    }

    def clean(self, text: str) -> str:
        """Remove special characters, extra whitespace."""
        text = re.sub(r"[^\w\s\-]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def is_valid_token(self, token: str) -> bool:
        """Reject stop words, short tokens, pure numbers."""
        token_lower = token.lower()
        if token_lower in self.STOP_WORDS:
            return False
        if len(token) < 2:
            return False
        if token.isdigit():
            return False
        return True

    def tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenizer."""
        return [t for t in self.clean(text).split() if self.is_valid_token(t)]


# ─────────────────────────────────────────────
# 2. CANDIDATE KEYWORD GENERATION
# ─────────────────────────────────────────────

class CandidateGenerator:
    """
    Generates n-gram candidate phrases from the document.
    Supports unigrams, bigrams, and trigrams.
    """

    def __init__(self, ngram_range: Tuple[int, int] = (1, 3)):
        self.ngram_range = ngram_range
        self.preprocessor = TextPreprocessor()

    def _extract_ngrams(self, tokens: List[str], n: int) -> List[str]:
        """Slide a window of size n across the token list."""
        return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    def generate(self, text: str) -> List[str]:
        """
        Returns deduplicated list of candidate keyword phrases.
        Filters out n-grams that start or end with a stop word.
        """
        tokens = self.preprocessor.clean(text).split()
        candidates = []
        stop = self.preprocessor.STOP_WORDS
        min_n, max_n = self.ngram_range

        for n in range(min_n, max_n + 1):
            for i in range(len(tokens) - n + 1):
                phrase_tokens = tokens[i:i + n]
                # Skip if first or last token is a stop word / short
                if phrase_tokens[0].lower() in stop or phrase_tokens[-1].lower() in stop:
                    continue
                if len(phrase_tokens[0]) < 2 or len(phrase_tokens[-1]) < 2:
                    continue
                phrase = " ".join(phrase_tokens)
                candidates.append(phrase)

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for c in candidates:
            key = c.lower()
            if key not in seen:
                seen.add(key)
                unique.append(c)

        return unique


# ─────────────────────────────────────────────
# 3. EMBEDDING MODEL  (LSA / Semantic)
# ─────────────────────────────────────────────

class LSAEmbedder:
    """
    Builds document embeddings using TF-IDF + Truncated SVD (LSA).

    This mirrors the role of a sentence-transformer model:
      - fit() trains on a corpus
      - embed() encodes any text into a dense vector
    The resulting vectors capture latent semantic structure,
    enabling cosine-similarity-based keyword ranking.
    """

    def __init__(self, n_components: int = 50):
        self.n_components = n_components
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,        # apply log(1+tf) scaling
        )
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._fitted = False

    def fit(self, texts: List[str]) -> "LSAEmbedder":
        """Train the TF-IDF vocab and SVD projection on a corpus."""
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        # Guard: SVD components can't exceed matrix rank
        n_comp = min(self.n_components, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
        self.svd.n_components = max(1, n_comp)
        self.svd.fit(tfidf_matrix)
        self._fitted = True
        return self

    def embed(self, texts: List[str]) -> np.ndarray:
        """Transform texts to semantic embedding vectors (L2-normalised)."""
        if not self._fitted:
            raise RuntimeError("Call fit() before embed().")
        tfidf = self.vectorizer.transform(texts)
        vectors = self.svd.transform(tfidf)
        # L2 normalise so cosine similarity == dot product
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms


# ─────────────────────────────────────────────
# 4. SEMANTIC KEYWORD RANKER
# ─────────────────────────────────────────────

class SemanticKeywordRanker:
    """
    Ranks candidate phrases by their cosine similarity to the
    full document embedding. Higher score → more semantically central.
    """

    def rank(
        self,
        doc_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        candidates: List[str],
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Returns top_k (keyword, score) tuples sorted by descending similarity.
        doc_embedding  : shape (1, d)
        candidate_embeddings : shape (N, d)
        """
        sims = cosine_similarity(doc_embedding, candidate_embeddings)[0]  # (N,)
        ranked_idx = np.argsort(sims)[::-1]
        results = []
        seen_unigrams: set = set()

        for idx in ranked_idx:
            phrase = candidates[idx]
            score = float(sims[idx])

            # Soft redundancy filter: skip if all words already covered
            words = set(phrase.lower().split())
            if words.issubset(seen_unigrams) and len(phrase.split()) == 1:
                continue

            results.append((phrase, round(score, 4)))
            seen_unigrams.update(words)

            if len(results) >= top_k:
                break

        return results


# ─────────────────────────────────────────────
# 5. TF-IDF BASELINE
# ─────────────────────────────────────────────

class TFIDFBaseline:
    """
    Classic TF-IDF keyword extraction for comparison.
    Treats the document as a single-document corpus; uses
    IDF computed over candidate phrases as pseudo-documents.
    """

    def extract(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Fit TF-IDF on sentences, then score unigrams/bigrams by
        their maximum TF-IDF weight across all sentences.
        """
        # Split into sentence-level pseudo-documents for IDF
        sentences = re.split(r"[.!?;]\s*", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        if not sentences:
            sentences = [text]

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )
        try:
            matrix = vectorizer.fit_transform(sentences)
        except ValueError:
            return []

        feature_names = vectorizer.get_feature_names_out()
        # Max TF-IDF score across all sentence rows
        scores = np.asarray(matrix.max(axis=0).todense()).flatten()
        ranked = np.argsort(scores)[::-1]

        results = []
        for idx in ranked[:top_k]:
            results.append((feature_names[idx], round(float(scores[idx]), 4)))
        return results


# ─────────────────────────────────────────────
# 6. MAIN PIPELINE
# ─────────────────────────────────────────────

class KeywordExtractionPipeline:
    """
    Orchestrates the full semantic keyword extraction workflow:

        Text → Preprocessing → Candidate Generation
             → LSA Embeddings → Cosine Ranking → Top-K Keywords

    Also runs TF-IDF baseline for side-by-side comparison.
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 3),
        embedding_dims: int = 50,
        top_k: int = 10,
    ):
        self.ngram_range = ngram_range
        self.top_k = top_k
        self.preprocessor = TextPreprocessor()
        self.candidate_gen = CandidateGenerator(ngram_range=ngram_range)
        self.embedder = LSAEmbedder(n_components=embedding_dims)
        self.ranker = SemanticKeywordRanker()
        self.tfidf_baseline = TFIDFBaseline()

    def _build_corpus(self, document: str, candidates: List[str]) -> List[str]:
        """
        Combine document + candidates into one corpus for LSA fitting.
        This gives the SVD a richer vocabulary to learn from.
        """
        return [document] + candidates

    def run(self, document: str) -> dict:
        """
        Full pipeline execution.

        Returns
        -------
        dict with keys:
          candidates      : all generated candidate phrases
          semantic        : [(phrase, score), ...] top-K semantic keywords
          tfidf           : [(phrase, score), ...] top-K TF-IDF keywords
          explained_var   : variance explained by the LSA model (%)
        """
        if not document.strip():
            raise ValueError("Input document is empty.")

        # Step 1 – Generate candidates
        candidates = self.candidate_gen.generate(document)
        if not candidates:
            raise ValueError("No valid candidate phrases found in document.")

        # Step 2 – Build corpus & fit embedder
        corpus = self._build_corpus(document, candidates)
        self.embedder.fit(corpus)

        # Step 3 – Embed document and candidates
        doc_embedding = self.embedder.embed([document])          # (1, d)
        cand_embeddings = self.embedder.embed(candidates)        # (N, d)

        # Step 4 – Rank candidates by semantic similarity
        semantic_keywords = self.ranker.rank(
            doc_embedding, cand_embeddings, candidates, top_k=self.top_k
        )

        # Step 5 – TF-IDF baseline
        tfidf_keywords = self.tfidf_baseline.extract(document, top_k=self.top_k)

        # Step 6 – Explained variance (interpretability metric)
        explained_var = round(
            float(np.sum(self.embedder.svd.explained_variance_ratio_)) * 100, 2
        )

        return {
            "candidates": candidates,
            "semantic": semantic_keywords,
            "tfidf": tfidf_keywords,
            "explained_var": explained_var,
        }
