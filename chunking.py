def fixed_size_chunk(text: str, chunk_size: int = 500, overlap: int = 50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def semantic_chunk(text: str, max_size: int = 1000):
    """Structure-based chunking: split on paragraph boundaries, then pack up to max_size.

    Caveat: this is the *simple* version of "semantic" chunking. True semantic
    chunking embeds each sentence and splits where embedding similarity drops
    (i.e. where the topic actually shifts). In production you'd wire up an
    embedding model + a vector database (e.g. Pinecone) with LangChain to do
    that for real. Here we approximate it with paragraph structure so the demo
    runs with zero setup.
    """
    # Split by paragraphs first
    paragraphs = text.split("\n\n")

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) < max_size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def recursive_chunk(text: str, separators: list, max_size: int = 1000):
    if not separators or len(text) <= max_size:
        return [text] if text else []

    sep = separators[0]
    parts = text.split(sep)

    chunks = []
    current = ""

    def finalize(buffer):
        # `buffer` always ends with the separator we appended below;
        # drop that trailing separator before trimming whitespace.
        return buffer[: -len(sep)].strip()

    for part in parts:
        if len(current) + len(part) < max_size:
            current += part + sep
        else:
            if current:
                chunks.append(finalize(current))
                current = ""
            # Recursively chunk parts that are still too large
            if len(part) > max_size:
                chunks.extend(recursive_chunk(part, separators[1:], max_size))
            else:
                current = part + sep

    if current:
        chunks.append(finalize(current))

    return chunks


def sentence_window_chunk(text: str, window_size: int = 3):
    import nltk

    sentences = nltk.sent_tokenize(text)

    chunks = []
    for i, sentence in enumerate(sentences):
        # Store the sentence as the chunk
        # But keep surrounding context as metadata
        start = max(0, i - window_size)
        end = min(len(sentences), i + window_size + 1)

        chunks.append(
            {
                "chunk": sentence,  # What we embed and search
                "context": " ".join(sentences[start:end]),  # What we send to LLM
                "position": i,
            }
        )

    return chunks


text = """
# Company Policies

## Refund Policy

All purchases can be refunded within 30 days.
Enterprise customers have a 60-day window.
Refunds are processed within 5 business days.

## Shipping Policy

Standard shipping takes 5-7 business days.
Express shipping takes 2-3 business days.
Free shipping on orders over $50.
""".strip()

def show(title, chunks):
    print(f"\n=== {title} ===")
    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- chunk {i} [{len(chunk)} chars] ---")
        print(chunk)


show("Fixed Size (200 chars)", fixed_size_chunk(text, 200, 20))
show("Semantic", semantic_chunk(text, 300))
show("Recursive", recursive_chunk(text, ["\n## ", "\n\n", ". "], 200))
