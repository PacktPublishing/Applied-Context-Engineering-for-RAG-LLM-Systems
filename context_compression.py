"""
CONTEXT COMPRESSION
Extract signal, remove noise

Run: python context_compression.py
"""

from typing import List, Dict
from dataclasses import dataclass
import re

# Common words that add no signal for relevance scoring.
# In production you'd use embeddings (semantic meaning) instead of word matching.
STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "for",
    "of",
    "to",
    "in",
    "on",
    "and",
    "or",
    "but",
    "what",
    "whats",
    "how",
    "why",
    "do",
    "does",
    "with",
    "at",
    "by",
    "from",
    "this",
    "that",
    "these",
    "those",
    "it",
}


def keywords(text: str) -> set:
    """Lowercase words with punctuation stripped, stopwords removed."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if len(w) > 1 and w not in STOPWORDS}


# =============================================================================
# EXTRACTIVE COMPRESSION
# =============================================================================


class ExtractiveCompressor:
    """
    Keep only sentences most relevant to the query.
    Fast and cheap - no LLM calls.
    """

    def __init__(self):
        # NOTE: This demo scores sentences by keyword overlap so it runs
        # anywhere with zero dependencies. In production you'd swap this for
        # semantic similarity:
        #   from sentence_transformers import SentenceTransformer, util
        #   self.model = SentenceTransformer('all-MiniLM-L6-v2')
        #   scores = util.cos_sim(model.encode(query), model.encode(sentences))
        # Embeddings understand meaning ("reimbursement" ~ "refund");
        # keyword overlap only catches exact word matches.
        pass

    def compress(self, query: str, text: str, keep_ratio: float = 0.5) -> str:
        """
        Extract most relevant sentences.

        Args:
            query: The search query
            text: Text to compress
            keep_ratio: Fraction of sentences to keep (0.5 = 50%)
        """
        # Split into sentences
        sentences = self._split_sentences(text)
        if len(sentences) <= 2:
            return text

        # Score each sentence by keyword overlap with the query.
        # (Stopwords/punctuation stripped so ranking reflects real signal.)
        query_words = keywords(query)
        scores = []
        for sent in sentences:
            overlap = len(query_words & keywords(sent))
            scores.append(overlap)

        # Keep top sentences (maintain original order)
        num_keep = max(1, int(len(sentences) * keep_ratio))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :num_keep
        ]
        top_indices.sort()  # Restore original order

        return " ".join([sentences[i] for i in top_indices])

    def _split_sentences(self, text: str) -> List[str]:
        """Simple sentence splitting."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]


# =============================================================================
# LLM-BASED COMPRESSION
# =============================================================================


class LLMCompressor:
    """
    Use LLM to extract relevant information.
    More accurate, but costs tokens.
    """

    def compress(self, query: str, text: str) -> str:
        """
        Simulated LLM compression.
        In production, use Claude Haiku for speed/cost.
        """
        # This simulates what an LLM would extract
        # In production:
        # response = client.messages.create(
        #     model="claude-3-haiku-20240307",
        #     max_tokens=200,
        #     messages=[{
        #         "role": "user",
        #         "content": f"Extract ONLY information relevant to: {query}\n\nText:\n{text}"
        #     }]
        # )

        # For demo, we'll do keyword-based extraction
        query_words = keywords(query)
        sentences = re.split(r"(?<=[.!?])\s+", text)

        relevant = []
        for sent in sentences:
            if query_words & keywords(sent):
                relevant.append(sent.strip())

        return " ".join(relevant) if relevant else text[:200]


# =============================================================================
# HYBRID COMPRESSION PIPELINE
# =============================================================================


@dataclass
class CompressionResult:
    original_text: str
    compressed_text: str
    original_chars: int
    compressed_chars: int
    compression_ratio: float
    method: str


class HybridCompressor:
    """
    Two-stage compression:
    1. Extractive (fast, cheap)
    2. LLM refinement if needed (accurate)
    """

    def __init__(self, target_tokens: int = 500):
        self.target_tokens = target_tokens
        self.extractive = ExtractiveCompressor()
        self.llm = LLMCompressor()

    def compress(self, query: str, chunks: List[str]) -> CompressionResult:
        """
        Compress multiple chunks into focused context.
        """
        original = "\n\n".join(chunks)
        original_chars = len(original)

        # Target character count (rough: 4 chars per token)
        target_chars = self.target_tokens * 4

        # Stage 1: Extractive compression on each chunk
        extracted = []
        for chunk in chunks:
            compressed = self.extractive.compress(query, chunk, keep_ratio=0.5)
            extracted.append(compressed)

        combined = "\n\n".join(extracted)

        # Stage 2: If still too long, use LLM
        method = "extractive"
        if len(combined) > target_chars * 1.2:
            combined = self.llm.compress(query, combined)
            method = "extractive + llm"

        return CompressionResult(
            original_text=original,
            compressed_text=combined,
            original_chars=original_chars,
            compressed_chars=len(combined),
            compression_ratio=len(combined) / original_chars,
            method=method,
        )


# =============================================================================
# COMPRESSION PIPELINE FOR RAG
# =============================================================================


class RAGCompressionPipeline:
    """
    Complete compression pipeline for RAG results.
    """

    def __init__(self, target_tokens: int = 800):
        self.target_tokens = target_tokens
        self.compressor = ExtractiveCompressor()

    def process(self, query: str, results: List[Dict]) -> str:
        """
        Process retrieved chunks into compressed context.

        Args:
            query: The search query
            results: List of {"chunk": str, "source": str, ...}
        """
        compressed_parts = []
        chars_used = 0
        target_chars = self.target_tokens * 4

        for result in results:
            chunk = result.get("chunk", result.get("content", ""))
            source = result.get("source", "Unknown")

            # Compress the chunk
            compressed = self.compressor.compress(query, chunk, keep_ratio=0.6)

            # Add source attribution
            formatted = f"[{source}]\n{compressed}"

            # Check budget
            if chars_used + len(formatted) > target_chars:
                # Compress more aggressively to fit
                remaining = target_chars - chars_used
                if remaining > 100:
                    ratio = remaining / len(formatted)
                    compressed = self.compressor.compress(
                        query, chunk, keep_ratio=ratio
                    )
                    formatted = f"[{source}]\n{compressed}"
                    compressed_parts.append(formatted)
                break

            compressed_parts.append(formatted)
            chars_used += len(formatted)

        return "\n\n---\n\n".join(compressed_parts)


# =============================================================================
# MEASURING COMPRESSION QUALITY
# =============================================================================


def evaluate_compression(query: str, original: str, compressed: str) -> Dict:
    """
    Compression is only useful if the answer SURVIVES.
    Measure how much we saved AND whether the signal is still there.

    In production you'd send `compressed` to an LLM and check the answer
    matches the one from `original`. Here we check (dependency-free) that
    the key facts the query asks about are still present after compression.
    """
    original_tokens = len(original) // 4  # rough: ~4 chars per token
    compressed_tokens = len(compressed) // 4

    # Did the compressed text keep the words the query is asking about?
    query_words = keywords(query)
    kept = query_words & keywords(compressed)
    signal_retained = len(kept) / len(query_words) if query_words else 0.0

    return {
        "original_tokens": original_tokens,
        "compressed_tokens": compressed_tokens,
        "tokens_saved": original_tokens - compressed_tokens,
        "compression_ratio": len(compressed) / len(original) if original else 0.0,
        "signal_retained": signal_retained,
        "kept_terms": sorted(kept),
    }


# =============================================================================
# DEMO
# =============================================================================

SAMPLE_CHUNKS = [
    """Our refund policy is designed to ensure customer satisfaction.
    All purchases can be refunded within 30 days of the original purchase date.
    Items must be unused and in their original packaging.
    Digital products have a 7-day refund window due to their nature.
    Enterprise customers have an extended 60-day refund window for all products.
    Refunds are typically processed within 5-7 business days after approval.
    You will receive an email confirmation when the refund is complete.
    Please note that shipping costs are not refundable.""",
    """Enterprise customers receive special benefits and policies.
    Priority support is available 24/7 for enterprise accounts.
    Dedicated account managers handle all enterprise inquiries.
    Custom integrations can be developed for enterprise needs.
    Volume discounts are available for large orders.
    Enterprise refund policies extend to 60 days from purchase.
    Annual contracts may have different terms and conditions.
    Contact your account manager for enterprise-specific questions.""",
]


if __name__ == "__main__":
    print("=" * 70)
    print("CONTEXT COMPRESSION")
    print("=" * 70)

    query = "What's the refund policy for enterprise customers?"

    # Show original content
    print(f"\nQuery: '{query}'")
    print(f"\nOriginal chunks ({len(SAMPLE_CHUNKS)}):")
    total_chars = sum(len(c) for c in SAMPLE_CHUNKS)
    print(f"  Total: {total_chars} characters (~{total_chars//4} tokens)")

    # Extractive compression
    print("\n" + "-" * 50)
    print("EXTRACTIVE COMPRESSION (50% keep ratio)")
    print("-" * 50)

    compressor = ExtractiveCompressor()
    for i, chunk in enumerate(SAMPLE_CHUNKS, 1):
        compressed = compressor.compress(query, chunk, keep_ratio=0.5)
        print(f"\nChunk {i}:")
        print(f"  Before: {len(chunk)} chars")
        print(f"  After:  {len(compressed)} chars ({len(compressed)/len(chunk):.0%})")
        print(f"  Content: {compressed[:100]}...")

    # LLM-based compression (Strategy 2)
    print("\n" + "-" * 50)
    print("LLM-BASED COMPRESSION (simulated Claude Haiku)")
    print("-" * 50)

    llm = LLMCompressor()
    llm_out = llm.compress(query, SAMPLE_CHUNKS[0])
    print(f"\nChunk 1:")
    print(f"  Before: {len(SAMPLE_CHUNKS[0])} chars")
    print(
        f"  After:  {len(llm_out)} chars " f"({len(llm_out)/len(SAMPLE_CHUNKS[0]):.0%})"
    )
    print(f"  Content: {llm_out}")

    # Hybrid compression - low target forces the LLM refinement stage to fire
    print("\n" + "-" * 50)
    print("HYBRID COMPRESSION PIPELINE")
    print("-" * 50)

    hybrid = HybridCompressor(target_tokens=80)
    result = hybrid.compress(query, SAMPLE_CHUNKS)

    print(f"\nOriginal: {result.original_chars} chars")
    print(f"Compressed: {result.compressed_chars} chars")
    print(f"Ratio: {result.compression_ratio:.0%}")
    print(f"Method: {result.method}")
    print(f"\nCompressed output:")
    print(f"  {result.compressed_text[:200]}...")

    # RAG pipeline compression
    print("\n" + "-" * 50)
    print("RAG COMPRESSION PIPELINE")
    print("-" * 50)

    rag_results = [
        {"chunk": SAMPLE_CHUNKS[0], "source": "refund-policy.md"},
        {"chunk": SAMPLE_CHUNKS[1], "source": "enterprise-guide.md"},
    ]

    pipeline = RAGCompressionPipeline(target_tokens=150)
    context = pipeline.process(query, rag_results)

    print(f"\nTarget: 150 tokens (~600 chars)")
    print(f"Result: {len(context)} chars")
    print(f"\nFormatted context:")
    print(context)

    # Measuring quality - did we save tokens AND keep the answer?
    print("\n" + "-" * 50)
    print("MEASURING COMPRESSION QUALITY")
    print("-" * 50)

    original = "\n\n".join(SAMPLE_CHUNKS)
    metrics = evaluate_compression(query, original, context)
    print(f"\nOriginal:    {metrics['original_tokens']} tokens")
    print(f"Compressed:  {metrics['compressed_tokens']} tokens")
    print(
        f"Tokens saved: {metrics['tokens_saved']} "
        f"({1 - metrics['compression_ratio']:.0%} smaller)"
    )
    print(
        f"Signal retained: {metrics['signal_retained']:.0%} "
        f"(query terms kept: {', '.join(metrics['kept_terms'])})"
    )
    print("\n=> Fewer tokens, and the enterprise refund answer is still there.")

    # Summary
    print("\n" + "=" * 70)
    print("COMPRESSION STRATEGIES")
    print("=" * 70)
    print("""
| Strategy    | Speed | Cost   | Quality | Use Case             |
|-------------|-------|--------|---------|----------------------|
| Extractive  | Fast  | Free   | Good    | Most situations      |
| LLM-based   | Slow  | Tokens | Better  | Complex content      |
| Hybrid      | Med   | Some   | Best    | Production systems   |

BEST PRACTICES:
1. Start with extractive (fast, free)
2. Add LLM only if extractive quality isn't enough
3. Always respect token budget
4. Include source attribution in compressed output
5. Target 40-60% compression while maintaining answer quality
""")
