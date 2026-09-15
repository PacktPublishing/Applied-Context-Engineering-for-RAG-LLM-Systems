"""
Lesson 4.3: Chunk Size, Overlap, and Metadata — The Math

A single, zero-setup, runnable demo of every concept from the lesson:
  1. The chunk-size tradeoff (small = precise, large = context-rich)
  2. The math behind overlap (storage cost vs. boundary coverage)
  3. Metadata as the secret weapon (the EnrichedChunk)
  4. Why metadata matters (filter-then-search, hybrid retrieval, attribution)
  5. The chunk-parameter formula (pick params from query type + context budget)

Run it:  python chunk_math.py
No external libraries — just the standard library.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List


# ---------------------------------------------------------------------------
# 1. THE CHUNK-SIZE TRADEOFF
# ---------------------------------------------------------------------------
# There's no universally "correct" chunk size. The right size depends on what
# you're retrieving for. This table is the same one from the lesson, but here
# we print it so students can see the reasoning side-by-side.

CHUNK_SIZE_GUIDE = [
    # (use_case,        optimal_tokens, why)
    ("Q&A",             "200-400",      "Precise answers"),
    ("Summarization",   "500-1000",     "Need full context"),
    ("Code",            "300-500",      "Function-sized"),
    ("Legal docs",      "400-800",      "Clause-sized"),
]


def show_tradeoff():
    print("=" * 70)
    print("1. THE CHUNK-SIZE TRADEOFF")
    print("=" * 70)
    print("""
Small chunks (100-300 tokens): precise retrieval, higher recall,
    BUT lose context and fragment ideas.
Large chunks (500-1500 tokens): preserve context, less storage,
    BUT lower precision and more noise per chunk.

The "sweet spot" depends entirely on your use case:
""")
    print(f"  {'Use Case':<16}{'Optimal Size':<16}{'Why'}")
    print(f"  {'-' * 60}")
    for use_case, size, why in CHUNK_SIZE_GUIDE:
        print(f"  {use_case:<16}{size + ' tokens':<16}{why}")
    print()


# ---------------------------------------------------------------------------
# 2. THE MATH BEHIND OVERLAP
# ---------------------------------------------------------------------------
# Overlap repeats a few tokens at the boundary of each chunk so an idea that
# straddles a boundary isn't split in half. The cost is storage: every
# overlapped token gets stored twice (or more). This function quantifies it.

def calculate_overlap(chunk_size: int, doc_size: int, overlap_pct: float = 0.1):
    """Show the storage cost of overlap for a given chunk size.

    Trade-off:
      - More overlap = better boundary coverage, more storage
      - Less overlap = less storage, potential gaps at boundaries
    """
    overlap = int(chunk_size * overlap_pct)

    # Without overlap, chunks tile the document end-to-end.
    chunks_no_overlap = doc_size / chunk_size

    # With overlap, each new chunk only advances by (chunk_size - overlap).
    effective_step = chunk_size - overlap
    chunks_with_overlap = (doc_size - overlap) / effective_step

    storage_increase = chunks_with_overlap / chunks_no_overlap

    return {
        "overlap_tokens": overlap,
        "chunks_no_overlap": round(chunks_no_overlap, 2),
        "chunks_with_overlap": round(chunks_with_overlap, 2),
        "storage_increase": f"{(storage_increase - 1) * 100:.1f}%",
    }


# Recommended overlap by chunk size (smaller chunks need a larger % overlap
# because a fixed idea spans a larger fraction of a small chunk).
OVERLAP_GUIDE = [
    # (chunk_size, overlap_tokens, overlap_pct, reasoning)
    (200,  "20-40",   "10-20%", "Small chunks need more overlap"),
    (500,  "50-100",  "10-20%", "Standard recommendation"),
    (1000, "100-150", "10-15%", "Large chunks need less"),
]
# storage_increase ≈ rule_of_thumb × (1 − overlap/doc_size)

def show_overlap_math():
    print("=" * 70)
    print("2. THE MATH BEHIND OVERLAP")
    print("=" * 70)
    print("\nFor a 1000-token document chunked at 200 tokens:\n")

    for pct in (0.10, 0.20, 0.50):
        r = calculate_overlap(chunk_size=200, doc_size=1000, overlap_pct=pct)
        print(f"  {int(pct * 100):>3}% overlap "
              f"({r['overlap_tokens']:>3} tokens)  ->  "
              f"{r['chunks_no_overlap']} chunks become "
              f"{r['chunks_with_overlap']} chunks  =  "
              f"+{r['storage_increase']} storage")

    # Why don't these match the famous "10% -> 11%, 20% -> 25%, 50% -> 2x"
    # rule of thumb? Because that rule is the LARGE-document limit. The exact
    # formula is:  rule_of_thumb x (1 - overlap/doc_size). On a tiny 1000-token
    # doc the correction term bites; as the doc grows it vanishes and the
    # formula converges right back to the rule of thumb. Watch it converge:
    print("\n  The rule of thumb (10%->11%, 20%->25%, 50%->2x) is the")
    print("  LARGE-document limit. Watch a 50% overlap converge as docs grow:\n")
    for doc in (1_000, 10_000, 100_000, 1_000_000):
        r = calculate_overlap(chunk_size=200, doc_size=doc, overlap_pct=0.50)
        print(f"    doc = {doc:>9,} tokens  ->  +{r['storage_increase']:>6} storage")
    print("    (limit ............................  +100.0% storage)")

    print("\n  Recommended overlap by chunk size:\n")
    print(f"  {'Chunk Size':<14}{'Overlap':<18}{'%':<10}{'Reasoning'}")
    print(f"  {'-' * 64}")
    for size, ov_tokens, ov_pct, reason in OVERLAP_GUIDE:
        print(f"  {str(size) + ' tok':<14}{ov_tokens + ' tok':<18}{ov_pct:<10}{reason}")
    print()


# ---------------------------------------------------------------------------
# 3. METADATA: THE SECRET WEAPON
# ---------------------------------------------------------------------------
# The chunk text is only half the story. Wrapping it with metadata is what
# turns "search every chunk" into "filter, then search a tiny slice" — and
# lets the LLM cite its source.

@dataclass
class EnrichedChunk:
    content: str                                  # The actual text

    # --- Structural metadata: where this chunk lives in the document ---
    source: str = ""                              # Document name / path
    section: str = ""                             # e.g. "Refund Policy"
    subsection: str = ""                          # e.g. "Enterprise Customers"
    position: int = 0                             # Chunk N of M

    # --- Semantic metadata: what this chunk is about ---
    summary: str = ""                             # LLM-generated summary
    keywords: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)

    # --- Retrieval metadata: how we find and stamp it ---
    embedding: List[float] = field(default_factory=list)  # Vector embedding
    created_at: datetime = field(default_factory=datetime.now)

    def to_context(self) -> str:
        """Format for the LLM context window — note the source breadcrumb,
        which is what enables accurate citation."""
        return f"[Source: {self.source} > {self.section}]\n{self.content}"


# A small in-memory "index" of enriched chunks so we can demo filtering and
# hybrid retrieval without a real vector database.
def build_demo_index() -> List[EnrichedChunk]:
    return [
        EnrichedChunk(
            content="All purchases can be refunded within 30 days.",
            source="policies.md", section="Refund Policy",
            subsection="Standard Customers", position=1,
            keywords=["refund", "30 days", "standard"],
            entities=["30 days"],
        ),
        EnrichedChunk(
            content="Enterprise customers have a 60-day refund window.",
            source="policies.md", section="Refund Policy",
            subsection="Enterprise Customers", position=2,
            keywords=["refund", "60 days", "enterprise"],
            entities=["enterprise", "60 days"],
        ),
        EnrichedChunk(
            content="Standard shipping takes 5-7 business days.",
            source="policies.md", section="Shipping Policy",
            subsection="Standard", position=3,
            keywords=["shipping", "standard", "5-7 days"],
            entities=["5-7 business days"],
        ),
        EnrichedChunk(
            content="Free shipping on enterprise orders over $50.",
            source="policies.md", section="Shipping Policy",
            subsection="Enterprise", position=4,
            keywords=["shipping", "free", "enterprise", "$50"],
            entities=["enterprise", "$50"],
        ),
    ]


# ---------------------------------------------------------------------------
# 4. WHY METADATA MATTERS
# ---------------------------------------------------------------------------
# These are toy implementations. A real system swaps the keyword matching for
# vector similarity, but the *shape* of the win is identical: metadata lets you
# shrink the search space BEFORE you do the expensive semantic comparison.

def naive_search(index: List[EnrichedChunk], query: str) -> List[EnrichedChunk]:
    """Search ALL chunks — every chunk is a candidate."""
    q = query.lower()
    return [c for c in index if any(k in q or k in c.content.lower()
                                    for k in c.keywords)]


def filtered_search(index: List[EnrichedChunk], query: str,
                    section: str) -> List[EnrichedChunk]:
    """Filter by metadata FIRST, then search the survivors only."""
    candidates = [c for c in index if c.section == section]
    q = query.lower()
    return [c for c in candidates if any(k in q or k in c.content.lower()
                                         for k in c.keywords)]


def hybrid_search(index: List[EnrichedChunk], query: str,
                  keyword_filter: str = "", section_filter: str = "") -> List[EnrichedChunk]:
    """Combine a semantic-ish match with keyword + section filters."""
    results = index
    if section_filter:
        results = [c for c in results if c.section == section_filter]
    if keyword_filter:
        results = [c for c in results if keyword_filter.lower() in
                   [k.lower() for k in c.keywords]]
    q = query.lower()
    return [c for c in results if any(k in q or k in c.content.lower()
                                      for k in c.keywords)]


def show_metadata_power():
    print("=" * 70)
    print("3 & 4. METADATA: THE SECRET WEAPON")
    print("=" * 70)
    index = build_demo_index()
    query = "what is the refund window for enterprise?"

    print(f"\n  Index has {len(index)} enriched chunks.")
    print(f'  Query: "{query}"\n')

    naive = naive_search(index, query)
    print(f"  (a) NAIVE search — scans all {len(index)} chunks, "
          f"matches {len(naive)}:")
    for c in naive:
        print(f"        - [{c.section} > {c.subsection}] {c.content}")

    filtered = filtered_search(index, query, section="Refund Policy")
    print(f"\n  (b) FILTER-then-search — pre-filter to section='Refund Policy', "
          f"then match {len(filtered)}:")
    for c in filtered:
        print(f"        - [{c.section} > {c.subsection}] {c.content}")

    hybrid = hybrid_search(index, query, keyword_filter="enterprise",
                           section_filter="Refund Policy")
    print(f"\n  (c) HYBRID — section='Refund Policy' + keyword='enterprise', "
          f"matches {len(hybrid)}:")
    for c in hybrid:
        print(f"        - [{c.section} > {c.subsection}] {c.content}")

    print("\n  (d) SOURCE ATTRIBUTION — what the LLM actually receives:\n")
    for c in hybrid:
        for line in c.to_context().splitlines():
            print(f"        {line}")
    print()


# ---------------------------------------------------------------------------
# 5. THE CHUNK-PARAMETER FORMULA
# ---------------------------------------------------------------------------
# Don't guess. Derive your chunk size from (a) how specific your queries are
# and (b) how much of the context window you can afford to spend on retrieval.

def calculate_chunk_params(
    avg_doc_length: int,        # Average document length in tokens
    query_specificity: str,     # "high", "medium", "low"
    context_window: int,        # LLM context window in tokens
    num_chunks_to_retrieve: int = 5,
):
    # Base chunk size driven by how specific queries are.
    if query_specificity == "high":
        chunk_size = 200        # Precise answers
    elif query_specificity == "medium":
        chunk_size = 400        # Balanced
    else:
        chunk_size = 800        # Broad context

    # Overlap as a fixed fraction of chunk size.
    overlap = int(chunk_size * 0.15)  # 15% overlap

    # Make sure everything we retrieve fits in the context budget, leaving
    # ~40% of the window free for the prompt + the model's response.
    total_retrieval_tokens = num_chunks_to_retrieve * chunk_size
    max_retrieval = context_window * 0.6

    if total_retrieval_tokens > max_retrieval:
        # Too big — shrink the chunk size so the retrieval set fits.
        chunk_size = int(max_retrieval / num_chunks_to_retrieve)
        overlap = int(chunk_size * 0.15)
        total_retrieval_tokens = num_chunks_to_retrieve * chunk_size

    return {
        "chunk_size": chunk_size,
        "overlap": overlap,
        "chunks_per_doc": avg_doc_length // (chunk_size - overlap),
        "retrieval_budget": total_retrieval_tokens,
    }


def show_formula():
    print("=" * 70)
    print("5. THE CHUNK-PARAMETER FORMULA")
    print("=" * 70)

    scenarios = [
        ("High-specificity Q&A, small context",
         dict(avg_doc_length=2000, query_specificity="high",
              context_window=8000, num_chunks_to_retrieve=5)),
        ("Balanced, mid context",
         dict(avg_doc_length=5000, query_specificity="medium",
              context_window=16000, num_chunks_to_retrieve=5)),
        ("Broad summarization, large context",
         dict(avg_doc_length=20000, query_specificity="low",
              context_window=32000, num_chunks_to_retrieve=8)),
        ("Budget-constrained (formula auto-shrinks chunks)",
         dict(avg_doc_length=2000, query_specificity="low",
              context_window=4000, num_chunks_to_retrieve=10)),
    ]

    for label, kwargs in scenarios:
        params = calculate_chunk_params(**kwargs)
        print(f"\n  {label}")
        print(f"    inputs:  {kwargs}")
        print(f"    output:  {params}")
    print()


# ---------------------------------------------------------------------------
# KEY TAKEAWAYS
# ---------------------------------------------------------------------------
def show_takeaways():
    print("=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    takeaways = [
        "Small chunks = precise; large chunks = context-rich.",
        "10-20% overlap is the sweet spot (cost ~= overlap % on large docs).",
        "Metadata enables FILTERING, not just searching — that's the real win.",
        "Calculate params from query type AND your context budget.",
        "These are starting points — always test with YOUR actual data.",
    ]
    for i, t in enumerate(takeaways, 1):
        print(f"  {i}. {t}")
    print()


if __name__ == "__main__":
    show_tradeoff()
    show_overlap_math()
    show_metadata_power()
    show_formula()
    show_takeaways()
