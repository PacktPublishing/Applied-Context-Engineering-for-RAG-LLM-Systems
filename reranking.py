"""
RE-RANKING -- the secret weapon for RAG accuracy.
Lesson 4.5 - Retrieve many (fast), re-rank (accurate), return few.

Stage 1: hybrid search casts a WIDE net -> lots of candidates (fast, approximate).
Stage 2: a cross-encoder re-reads each candidate WITH the query -> precise order.

We reuse the exact HybridRetriever from Lesson 4.4 so you can watch re-ranking
fix an order that hybrid search alone got slightly wrong.

Run: python reranking.py
Requires: pip install sentence-transformers scikit-learn rank-bm25
          (cohere is OPTIONAL -- only for the production API example)
"""

from typing import List

from sentence_transformers import CrossEncoder

# Reuse Lesson 4.4's retriever verbatim -- same chunks, same search, no surprises.
from hybrid_pipeline import HybridRetriever


# =============================================================================
# 1. BI-ENCODER vs CROSS-ENCODER  --  why re-ranking is more accurate
#
# Bi-encoder (what HybridRetriever uses):
#     query    -> encoder -> vector  ┐
#                                     ├─ cosine similarity
#     document -> encoder -> vector  ┘
#   Fast: document vectors are pre-computed once. But the query and the
#   document are encoded SEPARATELY -- they never "see" each other.
#
# Cross-encoder (the re-ranker):
#     [query, document] -> encoder -> single relevance score
#   The pair is read TOGETHER, so the model can judge fine-grained relevance.
#   Much more accurate, but slower -- you can't pre-compute, you score live.
#
# The trick: only run the slow-but-smart model on the handful of candidates
# the fast-but-rough model already shortlisted.
# =============================================================================


# =============================================================================
# 2. THE RE-RANKER  --  local cross-encoder
# =============================================================================


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        # First run downloads ~80MB and caches it. After that it's instant.
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 3,
    ) -> List[dict]:
        """
        Re-rank candidate chunks by true relevance to the query.

        Args:
            query: the search query
            candidates: candidate chunks (plain strings) from initial retrieval
            top_k: how many to return after re-ranking

        Returns:
            Top-k chunks sorted by cross-encoder relevance score (highest first).
        """
        if not candidates:
            return []

        # Pair the query with every candidate -- this is what the cross-encoder
        # reads "together". GOTCHA: candidates must be STRINGS here. If you feed
        # in (chunk, score) tuples or {"chunk": ...} dicts straight from a
        # retriever, the model silently scores garbage. Unpack them first.
        pairs = [[query, candidate] for candidate in candidates]

        # One score per pair. Higher = more relevant.
        scores = self.model.predict(pairs)

        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            {"chunk": chunk, "score": float(score)}
            for chunk, score in scored[:top_k]
        ]


# =============================================================================
# 3. THE FULL TWO-STAGE PIPELINE
# Stage 1 hands strings to Stage 2. The unpack (c["chunk"]) is the bridge --
# get it wrong and re-ranking quietly breaks. This is THE line to watch.
# =============================================================================


class RAGWithReranking:
    def __init__(self, chunks: List[str]):
        self.retriever = HybridRetriever(chunks)
        self.reranker = Reranker()

    def retrieve(
        self,
        query: str,
        initial_k: int = 10,
        final_k: int = 3,
    ) -> List[dict]:
        """
        Two-stage retrieval:
          1. Fast hybrid search casts a wide net -> initial_k candidates.
          2. Accurate cross-encoder re-ranks them -> final_k results.
        """
        # Stage 1: candidates. HybridRetriever.search returns dicts, so we pull
        # the raw text out before handing it to the re-ranker.
        candidates = self.retriever.search(query, top_k=initial_k)
        candidate_texts = [c["chunk"] for c in candidates]

        # Stage 2: re-rank.
        return self.reranker.rerank(query, candidate_texts, top_k=final_k)


# =============================================================================
# 4. PRODUCTION OPTION  --  Cohere Rerank API (paid, fast, no local model)
# Lazy import so this file still runs without `cohere` installed.
# Note the CURRENT API: cohere.ClientV2 + model "rerank-v3.5".
# (The old cohere.Client + "rerank-english-v2.0" still works but is legacy.)
# =============================================================================


class CohereReranker:
    def __init__(self, api_key: str):
        import cohere  # pip install cohere

        self.client = cohere.ClientV2(api_key)

    def rerank(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 3,
    ) -> List[dict]:
        response = self.client.rerank(
            model="rerank-v3.5",
            query=query,
            documents=candidates,
            top_n=top_k,
        )
        return [
            {"chunk": candidates[r.index], "score": r.relevance_score}
            for r in response.results
        ]


# =============================================================================
# DEMO  --  watch re-ranking PROMOTE the precise answer
#
# A help-center knowledge base with deliberate lexical decoys: lots of chunks
# say "refund" or "cancel", but only ONE actually answers the question. Hybrid
# search shortlists the right neighborhood; the cross-encoder picks the winner.
# =============================================================================

if __name__ == "__main__":
    chunks = [
        # THE CORRECT ANSWER to "which plan can I get money back on".
        # Note it says "refundable", not "money back" -- a vocab gap the
        # bi-encoder bridges semantically, but it still lands at #2, not #1.
        "Annual plans are refundable on a prorated basis for the months you "
        "didn't use.",
        # The relevant CONTRAST -- the other plan, negated. Bi-encoders are
        # notoriously blind to the word "not"; the cross-encoder is not.
        "Monthly plans are not refundable once the billing cycle has started.",
        # KEYWORD MAGNET / wrong answer: stuffed with "refund", so BM25 shoves
        # it to hybrid's #1 -- but it's about PROCESSING TIME, not which plan.
        "Approved refunds are processed within 5-7 business days back to your "
        "original card.",
        "Refunds are available within 30 days of your original purchase.",
        # Decoys / noise -- should not survive re-ranking.
        "To cancel your subscription, open Account Settings and click Terminate.",
        "Our billing cycle runs monthly; you're charged on the same date each "
        "month.",
        "Free trials convert to paid plans automatically unless you cancel first.",
        "Contact billing support to dispute a charge on your invoice.",
        "Error code E-4042 means the payment gateway timed out -- retry or "
        "contact support.",
        "Working out regularly improves your overall health and energy.",
    ]

    query = "Which plan can I get my money back on?"

    print("=" * 70)
    print(f"QUERY: {query!r}")
    print("=" * 70)

    # ---- Stage 1 only: what hybrid search alone returns ----
    retriever = HybridRetriever(chunks)
    stage1 = retriever.search(query, top_k=5)

    print("\n-- STAGE 1: HYBRID SEARCH top 5 (fast, approximate) --")
    for rank, c in enumerate(stage1, 1):
        print(f"  {rank}. [{c['score']:.3f}] {c['chunk'][:60]}...")

    # ---- Stage 1 + Stage 2: re-rank a slightly wider candidate pool ----
    # We cast for 6 candidates, then return the best 3 -- retrieve many, rank
    # accurately, return few.
    pipeline = RAGWithReranking(chunks)
    reranked = pipeline.retrieve(query, initial_k=6, final_k=3)

    print("\n-- STAGE 2: RE-RANKED top 3 (cross-encoder, accurate) --")
    for rank, r in enumerate(reranked, 1):
        print(f"  {rank}. [{r['score']:+.3f}] {r['chunk'][:60]}...")

    print(
        "\nNotice the flip: hybrid's #1 ('refunds processed in 5-7 days') is a "
        "\nkeyword magnet -- it's stuffed with 'refund' but answers the WRONG "
        "\nquestion. The cross-encoder reads each candidate WITH the query, "
        "\ndemotes that decoy, and promotes 'Annual plans are refundable' to #1."
        "\n\nThe scores are raw relevance logits, not 0-1 -- don't read the "
        "\nabsolute number, read the ORDER. Higher = more relevant."
    )
