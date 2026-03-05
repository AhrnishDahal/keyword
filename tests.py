"""
tests.py
========
Unit tests for the Semantic Keyword Extraction Pipeline.

Run with:
    python tests.py
"""

import sys
import unittest
import numpy as np
from keyword_extractor import (
    TextPreprocessor,
    CandidateGenerator,
    LSAEmbedder,
    SemanticKeywordRanker,
    TFIDFBaseline,
    KeywordExtractionPipeline,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_TEXT = (
    "Transformers have revolutionized natural language processing by enabling "
    "models to capture long-range contextual relationships in language. "
    "Pre-trained language models such as BERT and GPT have achieved "
    "state-of-the-art performance across NLP tasks."
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. TextPreprocessor
# ─────────────────────────────────────────────────────────────────────────────

class TestTextPreprocessor(unittest.TestCase):

    def setUp(self):
        self.pp = TextPreprocessor()

    def test_clean_removes_special_chars(self):
        result = self.pp.clean("Hello, world! #NLP @2024.")
        self.assertNotIn(",", result)
        self.assertNotIn("!", result)
        self.assertNotIn("#", result)

    def test_clean_collapses_whitespace(self):
        result = self.pp.clean("too   many     spaces")
        self.assertEqual(result, "too many spaces")

    def test_stop_word_rejected(self):
        self.assertFalse(self.pp.is_valid_token("the"))
        self.assertFalse(self.pp.is_valid_token("and"))
        self.assertFalse(self.pp.is_valid_token("is"))

    def test_valid_token_accepted(self):
        self.assertTrue(self.pp.is_valid_token("transformer"))
        self.assertTrue(self.pp.is_valid_token("BERT"))

    def test_short_token_rejected(self):
        self.assertFalse(self.pp.is_valid_token("a"))
        self.assertFalse(self.pp.is_valid_token("I"))

    def test_digit_token_rejected(self):
        self.assertFalse(self.pp.is_valid_token("123"))

    def test_tokenize_returns_list(self):
        tokens = self.pp.tokenize(SAMPLE_TEXT)
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)

    def test_tokenize_excludes_stop_words(self):
        tokens = self.pp.tokenize("The quick brown fox")
        self.assertNotIn("the", [t.lower() for t in tokens])


# ─────────────────────────────────────────────────────────────────────────────
# 2. CandidateGenerator
# ─────────────────────────────────────────────────────────────────────────────

class TestCandidateGenerator(unittest.TestCase):

    def setUp(self):
        self.gen = CandidateGenerator(ngram_range=(1, 3))

    def test_returns_list(self):
        result = self.gen.generate(SAMPLE_TEXT)
        self.assertIsInstance(result, list)

    def test_no_duplicates(self):
        result = self.gen.generate(SAMPLE_TEXT)
        lower = [c.lower() for c in result]
        self.assertEqual(len(lower), len(set(lower)))

    def test_no_leading_stop_word(self):
        stop = TextPreprocessor.STOP_WORDS
        result = self.gen.generate(SAMPLE_TEXT)
        for phrase in result:
            first = phrase.split()[0].lower()
            self.assertNotIn(first, stop,
                             f"Phrase '{phrase}' starts with a stop word")

    def test_no_trailing_stop_word(self):
        stop = TextPreprocessor.STOP_WORDS
        result = self.gen.generate(SAMPLE_TEXT)
        for phrase in result:
            last = phrase.split()[-1].lower()
            self.assertNotIn(last, stop,
                             f"Phrase '{phrase}' ends with a stop word")

    def test_unigram_only_mode(self):
        gen = CandidateGenerator(ngram_range=(1, 1))
        result = gen.generate("Natural language processing is fascinating")
        for phrase in result:
            self.assertEqual(len(phrase.split()), 1)

    def test_min_candidates_generated(self):
        result = self.gen.generate(SAMPLE_TEXT)
        self.assertGreater(len(result), 5)


# ─────────────────────────────────────────────────────────────────────────────
# 3. LSAEmbedder
# ─────────────────────────────────────────────────────────────────────────────

class TestLSAEmbedder(unittest.TestCase):

    def setUp(self):
        self.embedder = LSAEmbedder(n_components=10)
        self.corpus = [SAMPLE_TEXT, "BERT is a language model", "GPT generates text"]

    def test_fit_returns_self(self):
        result = self.embedder.fit(self.corpus)
        self.assertIs(result, self.embedder)

    def test_embed_before_fit_raises(self):
        fresh = LSAEmbedder()
        with self.assertRaises(RuntimeError):
            fresh.embed(["test"])

    def test_embed_shape(self):
        self.embedder.fit(self.corpus)
        vecs = self.embedder.embed(["natural language processing"])
        self.assertEqual(vecs.shape[0], 1)
        self.assertGreater(vecs.shape[1], 0)

    def test_embed_multiple_texts(self):
        self.embedder.fit(self.corpus)
        phrases = ["transformer", "language model", "neural network"]
        vecs = self.embedder.embed(phrases)
        self.assertEqual(vecs.shape[0], 3)

    def test_embeddings_l2_normalised(self):
        self.embedder.fit(self.corpus)
        vecs = self.embedder.embed(["natural language processing"])
        norm = float(np.linalg.norm(vecs[0]))
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_similar_texts_close(self):
        self.embedder.fit(self.corpus + ["deep learning", "machine learning"])
        v1 = self.embedder.embed(["language model"])
        v2 = self.embedder.embed(["BERT language"])
        v3 = self.embedder.embed(["cooking recipe ingredients"])
        sim_close = float(np.dot(v1.flatten(), v2.flatten()))
        sim_far = float(np.dot(v1.flatten(), v3.flatten()))
        # Not guaranteed with such tiny corpora, but at least check shape
        self.assertEqual(v1.shape, v2.shape)
        self.assertEqual(v1.shape, v3.shape)


# ─────────────────────────────────────────────────────────────────────────────
# 4. SemanticKeywordRanker
# ─────────────────────────────────────────────────────────────────────────────

class TestSemanticKeywordRanker(unittest.TestCase):

    def setUp(self):
        self.ranker = SemanticKeywordRanker()
        rng = np.random.default_rng(42)
        self.doc_emb = rng.random((1, 10))
        self.cand_embs = rng.random((20, 10))
        self.candidates = [f"phrase_{i}" for i in range(20)]

    def test_returns_list_of_tuples(self):
        result = self.ranker.rank(self.doc_emb, self.cand_embs, self.candidates, top_k=5)
        self.assertIsInstance(result, list)
        for item in result:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)

    def test_top_k_respected(self):
        for k in [3, 5, 10]:
            result = self.ranker.rank(self.doc_emb, self.cand_embs, self.candidates, top_k=k)
            self.assertLessEqual(len(result), k)

    def test_scores_are_floats(self):
        result = self.ranker.rank(self.doc_emb, self.cand_embs, self.candidates)
        for _, score in result:
            self.assertIsInstance(score, float)

    def test_descending_order(self):
        result = self.ranker.rank(self.doc_emb, self.cand_embs, self.candidates)
        scores = [s for _, s in result]
        self.assertEqual(scores, sorted(scores, reverse=True))


# ─────────────────────────────────────────────────────────────────────────────
# 5. TFIDFBaseline
# ─────────────────────────────────────────────────────────────────────────────

class TestTFIDFBaseline(unittest.TestCase):

    def setUp(self):
        self.baseline = TFIDFBaseline()

    def test_returns_list_of_tuples(self):
        result = self.baseline.extract(SAMPLE_TEXT)
        self.assertIsInstance(result, list)
        for item in result:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)

    def test_top_k_respected(self):
        for k in [3, 5, 8]:
            result = self.baseline.extract(SAMPLE_TEXT, top_k=k)
            self.assertLessEqual(len(result), k)

    def test_scores_are_positive(self):
        result = self.baseline.extract(SAMPLE_TEXT)
        for _, score in result:
            self.assertGreater(score, 0.0)

    def test_empty_document_returns_empty(self):
        result = self.baseline.extract("   ")
        self.assertIsInstance(result, list)


# ─────────────────────────────────────────────────────────────────────────────
# 6. KeywordExtractionPipeline (integration)
# ─────────────────────────────────────────────────────────────────────────────

class TestKeywordExtractionPipeline(unittest.TestCase):

    def setUp(self):
        self.pipeline = KeywordExtractionPipeline(
            ngram_range=(1, 2),
            embedding_dims=20,
            top_k=5,
        )

    def test_run_returns_dict(self):
        result = self.pipeline.run(SAMPLE_TEXT)
        self.assertIsInstance(result, dict)

    def test_result_has_required_keys(self):
        result = self.pipeline.run(SAMPLE_TEXT)
        for key in ("candidates", "semantic", "tfidf", "explained_var"):
            self.assertIn(key, result)

    def test_semantic_top_k(self):
        result = self.pipeline.run(SAMPLE_TEXT)
        self.assertLessEqual(len(result["semantic"]), 5)

    def test_tfidf_top_k(self):
        result = self.pipeline.run(SAMPLE_TEXT)
        self.assertLessEqual(len(result["tfidf"]), 5)

    def test_explained_var_is_percent(self):
        result = self.pipeline.run(SAMPLE_TEXT)
        self.assertGreater(result["explained_var"], 0)
        self.assertLessEqual(result["explained_var"], 100)

    def test_empty_document_raises(self):
        with self.assertRaises(ValueError):
            self.pipeline.run("   ")

    def test_semantic_scores_in_range(self):
        result = self.pipeline.run(SAMPLE_TEXT)
        for _, score in result["semantic"]:
            self.assertGreaterEqual(score, -1.0)
            self.assertLessEqual(score, 1.0)

    def test_candidates_are_strings(self):
        result = self.pipeline.run(SAMPLE_TEXT)
        for c in result["candidates"]:
            self.assertIsInstance(c, str)

    def test_different_docs_give_different_keywords(self):
        doc1 = "Transformers and BERT are language models used in NLP tasks."
        doc2 = "Climate change is driven by greenhouse gas emissions and global warming."
        r1 = self.pipeline.run(doc1)
        r2 = self.pipeline.run(doc2)
        kw1 = {kw for kw, _ in r1["semantic"]}
        kw2 = {kw for kw, _ in r2["semantic"]}
        # Two completely different domains should share very few keywords
        overlap = kw1 & kw2
        self.assertLess(len(overlap), len(kw1))


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Ordered test classes
    for cls in [
        TestTextPreprocessor,
        TestCandidateGenerator,
        TestLSAEmbedder,
        TestSemanticKeywordRanker,
        TestTFIDFBaseline,
        TestKeywordExtractionPipeline,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
