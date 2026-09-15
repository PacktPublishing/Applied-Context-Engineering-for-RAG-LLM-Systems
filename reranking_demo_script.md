# Lesson 4.5 Demo Script — Re-ranking, the Secret Weapon

Companion narration for `reranking.py`. Same honest style as 4.4 — we run it,
we read the *real* output, and we don't pretend the scores are prettier than
they are.

Run the code with:

```bash
python reranking.py
```

> **Numbers below match the actual output.** If you edit the chunks, the query,
> or `initial_k`/`final_k`, re-run and update the figures so your narration
> stays accurate.

Stage directions are in _italics_.

---

## Opening Hook

> "Here's a pattern that quietly fixes a lot of RAG mistakes: re-ranking. You
> retrieve a wide net of candidates fast, then use a smarter, slower model to
> pick the best few. One extra step, and your #1 result stops being whatever
> had the most keyword matches and starts being whatever actually _answers the
> question_. Let me show you the difference live."

---

## Beat 1 — Frame it (before you run anything)

> "We've got one file, `reranking.py`, and it reuses the exact `HybridRetriever`
> from the last lesson — same search, no new tricks there. What's new is a
> second stage bolted on top. So retrieval is really two jobs."

_Scroll to the top banner: bi-encoder vs cross-encoder._

> "Stage one — _recall_. Find everything that might be relevant. Fast,
> approximate, casts a wide net. That's our hybrid search. Stage two —
> _precision_. Take that shortlist and rank it by _actual_ relevance. Slower,
> but accurate. That's the re-ranker. Retrieve many, rank carefully, return
> few."

---

## Beat 2 — Why a cross-encoder is smarter (the architecture)

_Point at the `# 1.` banner comment block in the file._

> "Here's the why. Your hybrid search uses a _bi-encoder_. The query gets turned
> into a vector, each document gets turned into a vector — _separately_ — and we
> compare them with cosine similarity. It's fast because we pre-compute every
> document vector once. But the query and the document never actually _see_ each
> other. They're judged in isolation and compared at the end."

> "A _cross-encoder_ is different. It takes the query and one document
> _together_, as a pair, and reads them as a single input. So it can notice
> things like 'this paragraph mentions refunds, but it's about the _timing_, not
> _which plan_.' That's a judgment a bi-encoder literally can't make, because it
> never saw them side by side. The price: you can't pre-compute anything. You
> score live. Which is exactly why you only run it on the 6 or 10 candidates
> stage one already shortlisted — never the whole database."

---

## Beat 3 — Run stage one: watch hybrid pick the WRONG #1

> "Let's run it. The question is: _'Which plan can I get my money back on?'_
> Notice — I said 'money back,' the docs say 'refundable.' Different words, same
> meaning. Hold that thought."

_Run it. Point at the STAGE 1 block._

> "Hybrid search top five. And look at its number one — _'Approved refunds are
> processed within 5-7 business days,'_ at 0.812. That chunk is _stuffed_ with
> the word 'refund,' so the keyword half of hybrid loves it. But read it. It
> doesn't tell you _which plan_ anything. It's about how long a refund takes to
> hit your card. Wrong question."

> "The _right_ answer — _'Annual plans are refundable on a prorated basis'_ — is
> sitting at number two, 0.640. It's there. Hybrid found it. It just ranked it
> below a keyword magnet. And look further down: 'contact billing support' and
> 'free trials' at three and four — pure noise that happened to score okay."

---

## Beat 4 — Run stage two: the flip (the payoff)

_Point at the STAGE 2 block._

> "Now the re-ranker reads each of those candidates _with_ the question and
> re-scores them. Watch what happens. Number one is now _'Annual plans are
> refundable'_ — the correct answer, promoted from second place. The keyword
> magnet that hybrid crowned? _Gone._ It's not even in the top three anymore."

> "And look what climbed into second: _'Monthly plans are **not** refundable.'_
> That's the cross-encoder being smart. The question is 'which plan,' so the
> _other_ plan — even stated as a negative — is genuinely relevant context.
> Bi-encoders are famously bad at the word 'not'; they'd often treat 'is
> refundable' and 'is not refundable' as nearly identical. The cross-encoder
> reads the whole sentence and gets it."

> "So in one extra step: the wrong answer dropped out, the right answer rose to
> the top, and the noise got cleared. That's the secret weapon."

---

## Beat 5 — The honest part: those scores are negative

> "Now — full transparency, because I don't want this to trip you up. Look at
> the re-ranked scores. They're _negative_. Minus six, minus seven, minus eight.
> Your instinct is 'negative means bad, why is the best answer negative?'"

> "Here's the deal: a cross-encoder outputs a raw _relevance logit_, not a tidy
> zero-to-one probability. The absolute number is meaningless on its own — what
> matters is the _order_, and the _gaps_ between them. Minus six beats minus
> seven beats minus eight. If you genuinely need a 0-to-1 score — say, to set a
> 'drop anything below 0.5' threshold — you run the logits through a sigmoid.
> But for pure ranking, which is all we need here, you don't touch them. Read
> the order, ignore the sign."

_Optional, ~30s: show it live if you want —_

```python
import numpy as np
1 / (1 + np.exp(-(-6.032)))   # -> ~0.0024  (sigmoid turns the logit into 0-1)
```

> "That's the only gotcha in the whole technique. Everybody hits it once."

---

## Beat 6 — The pipeline, and when to bother

_Scroll to the `RAGWithReranking` class._

> "The whole thing wraps into one class. `retrieve` does stage one — hybrid
> search for, say, six candidates — then hands the raw text to stage two, which
> returns the best three. And see this line — `[c['chunk'] for c in
> candidates]`? That's the one bridge that matters. Stage two expects plain
> strings. If you hand it the dicts or tuples your retriever returns, it'll
> happily score garbage and never throw an error. Unpack first. That's the bug
> I'd bet money you'll hit."

> "When should you actually do this? Re-rank when accuracy matters more than the
> last few milliseconds, when stage one hands back ten-plus candidates, or when
> the question is complex. _Skip_ it when you need sub-100ms latency, when it's a
> dead-simple keyword lookup, or when you've only got a handful of chunks
> anyway. Re-ranking five things into three is rarely worth the round trip."

---

## Beat 7 — Production: don't run a model, call an API

_Scroll to `CohereReranker`._

> "Locally we're running `ms-marco-MiniLM` — small, free, downloads once, great
> for learning and plenty for a lot of apps. In production, you'll often reach
> for a hosted re-ranker instead: no model to host, faster, usually more
> accurate. Cohere's Rerank is the common one — current model's `rerank-v3.5`,
> called through `cohere.ClientV2`. Same shape: query in, documents in, ranked
> results out. BGE rerankers are the strong open-source option if you want to
> self-host — `bge-reranker-base` for speed, `large` for maximum accuracy."

> "Quick note — the older `cohere.Client` with `rerank-english-v2.0` still works
> and you'll see it in a lot of tutorials, but it's the legacy API. Use V2."

---

## Beat 8 — The takeaway

> "So, three things to remember. One: retrieval is two stages — cast a wide net
> fast, then rank it accurately. Cross-encoders are slow but smart _because_ the
> query and document are read together. Two: the typical pipeline is retrieve
> ten-to-twenty, re-rank, return three-to-five. Three: the win isn't subtle —
> you literally watched the wrong answer get demoted and the right one promoted
> in a single step. That's why re-ranking is the cheapest accuracy upgrade in
> RAG."

> "Next up — we've got better chunks, better search, and now better ranking. One
> last optimization: what if we could squeeze the retrieved text down, strip the
> noise, and keep only the signal before it ever hits the model? That's context
> compression."

---

## Delivery notes

- **First run downloads ~80MB** for `cross-encoder/ms-marco-MiniLM-L-6-v2` (and
  ~90MB for the 4.4 embedding model if it's not already cached). Run it once
  before recording so it's warm, or narrate the wait ("first time it grabs the
  models").
- Set `TOKENIZERS_PARALLELISM=false` before running to suppress a cosmetic
  HuggingFace warning that otherwise clutters the output.
- The model emits a tiny `Loading weights: 100%` progress line per encoder —
  harmless, but you can scroll past it.
- **Highest-impact version of Beat 4:** before running, ask the room to predict
  which chunk wins. Most people pick the one with the most "refund"s in it —
  i.e. hybrid's wrong #1. Then run it and let the flip land.

---

## Reference — expected output

```
QUERY: 'Which plan can I get my money back on?'

-- STAGE 1: HYBRID SEARCH top 5 (fast, approximate) --
  1. [0.812] Approved refunds are processed within 5-7 business days   <- keyword magnet, WRONG question
  2. [0.640] Annual plans are refundable on a prorated basis           <- correct answer, stuck at #2
  3. [0.589] Contact billing support to dispute a charge               <- noise
  4. [0.500] Free trials convert to paid plans automatically           <- noise
  5. [0.471] Monthly plans are not refundable                          <- relevant contrast, buried

-- STAGE 2: RE-RANKED top 3 (cross-encoder, accurate) --
  1. [-6.032] Annual plans are refundable on a prorated basis          <- PROMOTED to #1 (correct)
  2. [-7.544] Monthly plans are not refundable                         <- contrast surfaced; "not" understood
  3. [-8.485] Refunds are available within 30 days of purchase
              # the keyword magnet dropped OUT of the top 3 entirely
```

> Scores are raw relevance logits — read the **order**, not the absolute value.
> Exact numbers can shift slightly across `sentence-transformers` versions; the
> _ranking_ is what the lesson depends on.
