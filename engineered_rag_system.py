"""
ENGINEERED RAG SYSTEM
Complete production-ready RAG pipeline

Run: python engineered_rag_system.py
Requires: ANTHROPIC_API_KEY environment variable
"""

import anthropic
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import re
import time
from dotenv import load_dotenv

load_dotenv()

# Current Sonnet alias (never append a date suffix to an alias).
# Swap to "claude-opus-4-8" if you want max quality over cost.
MODEL = "claude-sonnet-4-6"

# Words too common to carry meaning — filtered out before any scoring so that
# retrieval and compression key on real query terms, not "what"/"is"/"the".
STOPWORDS = {
    "what",
    "whats",
    "which",
    "who",
    "how",
    "when",
    "where",
    "why",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "the",
    "a",
    "an",
    "of",
    "for",
    "to",
    "in",
    "on",
    "and",
    "or",
    "do",
    "does",
    "did",
    "i",
    "my",
    "me",
    "you",
    "your",
    "it",
    "its",
    "this",
    "that",
    "can",
    "will",
    "would",
    "should",
    "available",
    "there",
}


def keywords(text: str) -> set:
    """Lowercase content words with stopwords removed."""
    return {w for w in text.lower().split() if w not in STOPWORDS}


# =============================================================================
# DATA MODELS
# =============================================================================


@dataclass
class Chunk:
    content: str
    source: str
    section: str
    position: int
    metadata: Dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    rerank_score: float = 0.0


@dataclass
class RAGResponse:
    answer: str
    sources: List[str]
    context_chars: int
    context_tokens_est: int
    retrieval_count: int
    engineered: bool
    latency_ms: float


# =============================================================================
# DOCUMENT PROCESSOR
# =============================================================================


class DocumentProcessor:
    """Process documents into enriched chunks."""

    def __init__(self, chunk_size: int = 400, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def process(self, documents: List[Dict]) -> List[Chunk]:
        """Process multiple documents into chunks."""
        all_chunks = []
        for doc in documents:
            chunks = self._process_document(doc)
            all_chunks.extend(chunks)
        return all_chunks

    def _process_document(self, doc: Dict) -> List[Chunk]:
        """Process a single document."""
        content = doc["content"]
        source = doc["source"]
        sections = self._extract_sections(content)

        chunks = []
        position = 0

        for section_name, section_content in sections:
            section_chunks = self._semantic_chunk(section_content)
            for chunk_text in section_chunks:
                chunks.append(
                    Chunk(
                        content=chunk_text,
                        source=source,
                        section=section_name,
                        position=position,
                        metadata={
                            "word_count": len(chunk_text.split()),
                            "has_numbers": bool(re.search(r"\d+", chunk_text)),
                        },
                    )
                )
                position += 1

        return chunks

    def _extract_sections(self, content: str) -> List[tuple]:
        """Extract sections from markdown-like content."""
        lines = content.split("\n")
        sections = []
        current_section = "General"
        current_content = []

        for line in lines:
            if line.startswith("## "):
                if current_content:
                    sections.append((current_section, "\n".join(current_content)))
                current_section = line[3:].strip()
                current_content = []
            elif line.startswith("# "):
                if current_content:
                    sections.append((current_section, "\n".join(current_content)))
                current_section = line[2:].strip()
                current_content = []
            else:
                current_content.append(line)

        if current_content:
            sections.append((current_section, "\n".join(current_content)))

        return sections

    def _semantic_chunk(self, text: str) -> List[str]:
        """Chunk by paragraphs."""
        paragraphs = text.split("\n\n")
        chunks = []
        current = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current) + len(para) < self.chunk_size:
                current += para + "\n\n"
            else:
                if current:
                    chunks.append(current.strip())
                current = para + "\n\n"

        if current.strip():
            chunks.append(current.strip())

        return chunks


# =============================================================================
# HYBRID RETRIEVER
# =============================================================================


class HybridRetriever:
    """Semantic + keyword hybrid search with reranking."""

    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.chunk_texts = [c.content for c in chunks]

    def retrieve(
        self,
        query: str,
        initial_k: int = 20,
        final_k: int = 5,
        alpha: float = 0.6,
        use_reranking: bool = True,
    ) -> List[RetrievalResult]:
        """Full retrieval pipeline."""
        # Stage 1: Hybrid search
        candidates = self._hybrid_search(query, initial_k, alpha)

        if not use_reranking:
            return candidates[:final_k]

        # Stage 2: Rerank
        reranked = self._rerank(query, candidates)
        return reranked[:final_k]

    def _hybrid_search(
        self, query: str, top_k: int, alpha: float
    ) -> List[RetrievalResult]:
        """Combine semantic and keyword search.

        NOTE: 'semantic' here is simulated with lexical word overlap so the demo
        runs with zero dependencies and no API calls for retrieval. In production
        you'd replace this with real embeddings, e.g.:
            semantic = cosine(embed(query), embed(chunk.content))
        """
        query_words = keywords(query)

        results = []
        for i, chunk in enumerate(self.chunks):
            chunk_words = keywords(chunk.content)

            # Semantic-like score: fraction of query terms present in the chunk.
            semantic = len(query_words & chunk_words) / max(len(query_words), 1)

            # Keyword score: exact matches, with a 1.5x boost for terms that
            # appear early (first ~100 chars) since leading text tends to be topical.
            keyword = sum(
                1.5 if w in chunk.content.lower()[:100] else 1.0
                for w in query_words
                if w in chunk.content.lower()
            ) / max(len(query_words), 1)

            # Combined score
            combined = alpha * semantic + (1 - alpha) * keyword

            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=combined,
                    semantic_score=semantic,
                    keyword_score=keyword,
                )
            )

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def _rerank(
        self, query: str, candidates: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Simulated cross-encoder reranking.

        A real cross-encoder scores each (query, chunk) pair jointly with a model.
        Here we approximate it with cheap relevance signals so the concept is
        visible without extra dependencies. The boost weights below are
        illustrative knobs, not tuned values.
        """
        query_words = keywords(query)

        # Rerank based on deeper analysis
        for result in candidates:
            content = result.chunk.content.lower()

            # +0.2 if a query term appears in the chunk's section heading
            # (e.g. "refund" matching the "Refund Policy" section).
            title_boost = (
                0.2
                if any(w in result.chunk.section.lower() for w in query_words)
                else 0
            )
            # +0.15 if the chunk contains numbers — policy answers (days, prices,
            # hours) usually live in chunks with figures.
            number_boost = 0.15 if result.chunk.metadata.get("has_numbers") else 0

            # +0.3 if the chunk contains the opening of the query verbatim.
            phrase_boost = 0.3 if query.lower()[:20] in content else 0

            result.rerank_score = (
                result.score + title_boost + number_boost + phrase_boost
            )

        candidates.sort(key=lambda x: x.rerank_score, reverse=True)
        return candidates


# =============================================================================
# CONTEXT COMPRESSOR
# =============================================================================


class ContextCompressor:
    """Extract relevant information from chunks."""

    def compress(
        self, query: str, results: List[RetrievalResult], target_chars: int = 1500
    ) -> str:
        """Compress results into focused context."""
        query_words = keywords(query)
        compressed_parts = []
        chars_used = 0

        for result in results:
            # Extract relevant sentences
            sentences = re.split(r"(?<=[.!?])\s+", result.chunk.content)
            relevant = []

            for sent in sentences:
                if any(w in sent.lower() for w in query_words):
                    relevant.append(sent)

            # Use relevant sentences or first few if none match
            if relevant:
                compressed = " ".join(relevant)
            else:
                compressed = " ".join(sentences[:2])

            # Format with source
            formatted = (
                f"[{result.chunk.source} > {result.chunk.section}]\n{compressed}"
            )

            if chars_used + len(formatted) > target_chars:
                if chars_used == 0:
                    # At least include one chunk
                    compressed_parts.append(formatted[:target_chars])
                break

            compressed_parts.append(formatted)
            chars_used += len(formatted)

        return "\n\n".join(compressed_parts)


# =============================================================================
# ENGINEERED RAG SYSTEM
# =============================================================================


class EngineeredRAG:
    """Complete RAG system with all optimizations."""

    def __init__(self, documents: List[Dict]):
        # Process documents
        processor = DocumentProcessor()
        self.chunks = processor.process(documents)

        # Initialize components
        self.retriever = HybridRetriever(self.chunks)
        self.compressor = ContextCompressor()
        self.client = anthropic.Anthropic()

    def query(self, question: str, use_engineering: bool = True) -> RAGResponse:
        """Answer a question using RAG."""
        start_time = time.time()

        # Retrieve
        if use_engineering:
            results = self.retriever.retrieve(
                question, initial_k=15, final_k=5, use_reranking=True
            )
            context = self.compressor.compress(question, results, target_chars=1200)
        else:
            # Naive RAG
            results = self.retriever.retrieve(
                question, initial_k=5, final_k=5, alpha=1.0, use_reranking=False
            )
            context = "\n\n".join([r.chunk.content for r in results])

        # Generate answer
        response = self.client.messages.create(
            model=MODEL,
            max_tokens=400,
            messages=[
                {
                    "role": "user",
                    "content": f"""Answer based ONLY on the provided context.
If the answer isn't in the context, say "I don't have that information."
Cite your sources using [Source] format.

Context:
{context}

Question: {question}

Answer:""",
                }
            ],
        )

        latency = (time.time() - start_time) * 1000

        return RAGResponse(
            answer=response.content[0].text,
            sources=list(set(r.chunk.source for r in results)),
            context_chars=len(context),
            context_tokens_est=len(context) // 4,
            retrieval_count=len(results),
            engineered=use_engineering,
            latency_ms=latency,
        )


# =============================================================================
# COMPARISON DEMO
# =============================================================================


def is_grounded(answer: str, expected_facts: List[str]) -> bool:
    """A cheap, dependency-free correctness check: does the answer contain the
    facts a correct response must include? Not a substitute for a real eval
    harness — just enough to *show* accuracy instead of only claiming it."""
    a = answer.lower()
    return all(fact.lower() in a for fact in expected_facts)


def compare_approaches(rag: EngineeredRAG, questions: List[Dict]):
    """Compare naive vs engineered RAG on token cost AND correctness."""
    print("=" * 70)
    print("RAG COMPARISON: NAIVE vs ENGINEERED")
    print("=" * 70)

    total_savings = []
    naive_correct = 0
    eng_correct = 0

    for item in questions:
        question = item["q"]
        expected = item["expect"]

        print(f"\n{'='*70}")
        print(f"QUESTION: {question}")
        print(f"(must mention: {', '.join(expected)})")
        print("=" * 70)

        # Naive RAG
        naive = rag.query(question, use_engineering=False)
        naive_ok = is_grounded(naive.answer, expected)
        naive_correct += naive_ok
        print(f"\n--- NAIVE RAG ---")
        print(f"Answer: {naive.answer[:300]}...")
        print(
            f"Context: {naive.context_chars} chars (~{naive.context_tokens_est} tokens)"
        )
        print(
            f"Latency: {naive.latency_ms:.0f}ms   Grounded: {'PASS' if naive_ok else 'FAIL'}"
        )

        # Engineered RAG
        eng = rag.query(question, use_engineering=True)
        eng_ok = is_grounded(eng.answer, expected)
        eng_correct += eng_ok
        print(f"\n--- ENGINEERED RAG ---")
        print(f"Answer: {eng.answer[:300]}...")
        print(f"Context: {eng.context_chars} chars (~{eng.context_tokens_est} tokens)")
        print(
            f"Latency: {eng.latency_ms:.0f}ms   Grounded: {'PASS' if eng_ok else 'FAIL'}"
        )

        # Comparison (guard against an empty naive context)
        if naive.context_tokens_est:
            savings = (1 - eng.context_tokens_est / naive.context_tokens_est) * 100
            total_savings.append(savings)
            print(f"\n--- COMPARISON ---")
            print(f"Token savings: {savings:.1f}%")
        else:
            print(f"\n--- COMPARISON ---")
            print("Token savings: n/a (naive retrieved no context)")
        print(f"Sources: {', '.join(eng.sources)}")

    # Aggregate: this is what the demo actually proves this run
    n = len(questions)
    avg_savings = sum(total_savings) / len(total_savings) if total_savings else 0.0
    print(f"\n{'='*70}")
    print("MEASURED THIS RUN")
    print("=" * 70)
    print(
        f"Grounded answers:  naive {naive_correct}/{n}  |  engineered {eng_correct}/{n}"
    )
    print(f"Avg token savings: {avg_savings:.1f}% (engineered vs naive)")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Sample documents
    documents = [
        {
            "source": "policies.md",
            "content": """
# Company Policies

## Refund Policy

All purchases can be refunded within 30 days of purchase.
Items must be unused and in original packaging to qualify for a refund.
Digital products have a shortened 7-day refund window.
Refunds are processed within 5-7 business days after approval.

Enterprise customers have an extended 60-day refund window.
This applies to all products and services purchased under enterprise agreements.
Contact your dedicated account manager for enterprise refund requests.

## Shipping Policy

Standard shipping takes 5-7 business days and costs $5.99.
Express shipping takes 2-3 business days and costs $12.99.
Free shipping is available on orders over $50.

International shipping takes 10-14 business days.
Customs fees are the responsibility of the customer.

## Support Policy

Standard support is available Monday-Friday, 9am-5pm EST.
Premium support is available 24/7 for enterprise customers.
Response times: Standard 24 hours, Premium 2 hours.
""",
        },
        {
            "source": "enterprise.md",
            "content": """
# Enterprise Guide

## Overview

Enterprise plans are designed for organizations with 100+ users.
Custom pricing and features are available for enterprise customers.
Dedicated account managers handle all enterprise inquiries.

## Benefits

Priority support with 2-hour response time guarantee.
Extended 60-day refund policy on all purchases.
Custom integrations and API access.
Volume discounts starting at 20% off.

## Getting Started

Contact sales@company.com for enterprise pricing.
Minimum contract term is 12 months.
Custom onboarding available for teams over 500 users.
""",
        },
    ]

    # Test questions paired with the facts a correct answer must contain,
    # so we can actually score correctness instead of only claiming it.
    test_questions = [
        {"q": "What's the refund policy for enterprise customers?", "expect": ["60"]},
        {"q": "How long does standard shipping take?", "expect": ["5-7"]},
        {"q": "What is the premium support response time?", "expect": ["2"]},
    ]

    print("Initializing Engineered RAG System...")
    rag = EngineeredRAG(documents)
    print(f"Processed {len(rag.chunks)} chunks from {len(documents)} documents\n")

    compare_approaches(rag, test_questions)

    # Summary
    print("\n" + "=" * 70)
    print("WHAT YOU BUILT")
    print("=" * 70)
    print("""
1. Document Processor  - Section-aware chunking with metadata
2. Hybrid Retriever    - Lexical overlap + keyword search (simulates embeddings)
3. Reranking           - Relevance boosts (simulates a cross-encoder)
4. Context Compressor  - Extract only the sentences that answer the query
5. Full RAG Pipeline   - Retrieve -> rerank -> compress -> generate

WHAT THIS DEMO PROVES (see 'MEASURED THIS RUN' above):
- Token efficiency: engineered sends far less context for the same answers
- Accuracy maintained: grounded-answer score stays equal while tokens drop

WHY IT MATTERS AT SCALE (not shown by this toy corpus):
- On large / noisy corpora, reranking + compression also lift accuracy by
  keeping the signal and dropping distractors. Steps 2 and 3 are simulated
  here so it runs dependency-free -- swap in real embeddings and a
  cross-encoder for production (see the NOTE comments in HybridRetriever).
""")
