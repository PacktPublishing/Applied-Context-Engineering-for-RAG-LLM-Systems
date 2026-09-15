"""
SEMANTIC SEARCH vs KEYWORD SEARCH vs HYBRID
Lesson 4.4 - Same queries, different results from each method.

Semantic search finds MEANING (embeddings + cosine similarity).
Keyword search finds TERMS (BM25).
Hybrid combines both -- and beats either one alone.

Run: python hybrid_pipeline.py
Requires: pip install sentence-transformers scikit-learn rank-bm25
"""

import re
from typing import List, Tuple

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

# One embedding model, loaded once and reused by every method below.
model = SentenceTransformer("all-MiniLM-L6-v2")


def tokenize(text: str) -> List[str]:
    """
    Split into lowercase word tokens.

    GOTCHA: a naive `text.lower().split()` keeps punctuation attached, so
    "subscription," never matches "subscription" and BM25 silently scores 0.
    Splitting on word characters fixes that.
    """
    return re.findall(r"\w+", text.lower())


# =============================================================================
# 1. SEMANTIC SEARCH  --  finds MEANING
# Wins when the query uses different words than the document
# ("cancel" -> "terminate"). Fails on exact terms (IDs, error codes).
# =============================================================================


def semantic_search(
    query: str, chunks: List[str], top_k: int = 5
) -> List[Tuple[str, float]]:
    query_embedding = model.encode(query)
    chunk_embeddings = model.encode(chunks)

    similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]

    top_indices = similarities.argsort()[-top_k:][::-1]
    return [(chunks[i], float(similarities[i])) for i in top_indices]


# =============================================================================
# 2. KEYWORD SEARCH (BM25)  --  finds TERMS
# Wins when exact matches matter (error codes, SKUs, product names).
# Fails on synonyms and natural-language queries.
# =============================================================================


def keyword_search(
    query: str, chunks: List[str], top_k: int = 5
) -> List[Tuple[str, float]]:
    tokenized_chunks = [tokenize(chunk) for chunk in chunks]
    tokenized_query = tokenize(query)

    bm25 = BM25Okapi(tokenized_chunks)
    scores = bm25.get_scores(tokenized_query)

    top_indices = scores.argsort()[-top_k:][::-1]
    return [(chunks[i], float(scores[i])) for i in top_indices]


# =============================================================================
# 3. HYBRID SEARCH  --  the best of both
# alpha balances the two: 1.0 = pure semantic, 0.0 = pure keyword.
# We normalize each score set to 0-1 first so they're comparable.
# =============================================================================


def hybrid_search(
    query: str,
    chunks: List[str],
    top_k: int = 5,
    alpha: float = 0.5,  # semantic weight (1) vs keyword weight (0)
) -> List[Tuple[str, float]]:
    # Pull a wider net from each method, then re-rank the union.
    semantic_results = semantic_search(query, chunks, top_k=top_k * 2)
    keyword_results = keyword_search(query, chunks, top_k=top_k * 2)

    def normalize(results):
        if not results:
            return {}
        scores = [r[1] for r in results]
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s:
            return {r[0]: 1.0 for r in results}
        return {r[0]: (r[1] - min_s) / (max_s - min_s) for r in results}

    semantic_norm = normalize(semantic_results)
    keyword_norm = normalize(keyword_results)

    all_chunks = set(semantic_norm) | set(keyword_norm)
    combined = {}
    for chunk in all_chunks:
        sem_score = semantic_norm.get(chunk, 0)
        key_score = keyword_norm.get(chunk, 0)
        combined[chunk] = alpha * sem_score + (1 - alpha) * key_score

    sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]


# =============================================================================
# 4. RECIPROCAL RANK FUSION (RRF)  --  combine WITHOUT normalizing scores
# Uses only the RANK of each item, so it's robust to outliers and works
# with any number of retrievers. k=60 is the standard empirical constant.
# =============================================================================


def reciprocal_rank_fusion(
    result_lists: List[List[str]], k: int = 60
) -> List[Tuple[str, float]]:
    scores = {}
    for result_list in result_lists:
        for rank, item in enumerate(result_list):
            scores[item] = scores.get(item, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# =============================================================================
# 5. THE COMPLETE HYBRID PIPELINE
# Wraps it all in a reusable class that pre-builds the index once.
# =============================================================================


class HybridRetriever:
    def __init__(self, chunks: List[str], alpha: float = 0.5):
        self.chunks = chunks
        self.alpha = alpha

        # Semantic index
        self.model = model
        self.embeddings = self.model.encode(chunks)

        # Keyword index
        self.tokenized = [tokenize(c) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized)

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        query_emb = self.model.encode(query)
        sem_scores = cosine_similarity([query_emb], self.embeddings)[0]
        key_scores = self.bm25.get_scores(tokenize(query))

        sem_norm = (sem_scores - sem_scores.min()) / (
            sem_scores.max() - sem_scores.min() + 1e-6
        )
        key_norm = (key_scores - key_scores.min()) / (
            key_scores.max() - key_scores.min() + 1e-6
        )

        combined = self.alpha * sem_norm + (1 - self.alpha) * key_norm
        top_indices = combined.argsort()[-top_k:][::-1]

        return [
            {
                "chunk": self.chunks[i],
                "score": float(combined[i]),
                "semantic_score": float(sem_scores[i]),
                "keyword_score": float(key_scores[i]),
            }
            for i in top_indices
        ]


# =============================================================================
# COMPARISON  --  watch the three methods disagree on the same queries
# =============================================================================

if __name__ == "__main__":
    chunks = [
        "To cancel your subscription, open account settings and click Terminate.",
        "Working out regularly improves your overall health and energy.",
        "Error code E-4042 means the payment gateway timed out.",
        "Our refund policy allows returns within 30 days of purchase.",
        "Exercise has many benefits, including better sleep and mood.",
        "Contact billing support to dispute a charge on your invoice.",
    ]

    # Query A: pure vocabulary mismatch -> KEYWORD fails (0 shared words),
    #          semantic rescues it ("stop paying monthly" ~ cancel subscription).
    # Query B: exact token -> KEYWORD wins big ("E-4042"), semantic is weak.
    # Hybrid is the only method that nails BOTH.
    queries = ["I want to stop paying every month", "E-4042"]

    for query in queries:
        print("=" * 60)
        print(f"QUERY: {query!r}")
        print("=" * 60)

        print("\n-- SEMANTIC (meaning) --")
        for chunk, score in semantic_search(query, chunks, top_k=2):
            print(f"  [{score:.3f}] {chunk}")

        print("\n-- KEYWORD / BM25 (exact terms) --")
        for chunk, score in keyword_search(query, chunks, top_k=2):
            print(f"  [{score:.3f}] {chunk}")

        print("\n-- HYBRID (alpha=0.5) --")
        for chunk, score in hybrid_search(query, chunks, top_k=2, alpha=0.5):
            print(f"  [{score:.3f}] {chunk}")

        print("\n-- RRF (rank fusion, no score tuning) --")
        sem_ranked = [c for c, _ in semantic_search(query, chunks, top_k=5)]
        key_ranked = [c for c, _ in keyword_search(query, chunks, top_k=5)]
        for chunk, score in reciprocal_rank_fusion([sem_ranked, key_ranked])[:2]:
            print(f"  [{score:.4f}] {chunk}")
        print()
