"""Second-hop retrieval, for questions whose answer is in two places.

The problem this exists for is visible in the per-type evaluation: on
``multi_step`` questions the production pipeline scores Recall@1 of 0.4524 with
a *perfect* MRR of 1.0000. Those two numbers together say something precise. The
first relevant chunk is always ranked first; what is missing is the second one.
No amount of reordering a single ranked list fixes that, because the second
piece of evidence is not in the list to be reordered.

The fix here is pseudo-relevance feedback, not a language model. The first
retrieval is assumed to be right about its top few chunks; the terms that make
those chunks distinctive — and that the question did not already contain — are
appended to the query, and retrieval is run again. Fusing the two ranked lists
with the same Reciprocal Rank Fusion used for dense and sparse gives one list
containing both hops.

Two properties are deliberate:

*   **No model is called.** Query decomposition with a language model would work
    too, and would cost an API key, a network round trip and a dependency on the
    thing the rest of the pipeline is careful not to trust. This is arithmetic
    over the text already retrieved.
*   **Reranking still judges against the user's question**, never the expanded
    one. The expansion is a way to widen the candidate pool; it is not a
    restatement of what the user asked, and scoring against it would reward
    chunks that match the system's guess rather than the question.

Whether it actually helps is an empirical question, and the evaluation answers
it as a separate configuration rather than by switching the default.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass

from app.ingestion.text_utils import STOPWORDS, tokenize
from app.models.document import RetrievedChunk

logger = logging.getLogger(__name__)

#: A term has to look like a term. Pure digits are page numbers and counts;
#: single characters are noise; very long tokens are usually broken text.
_MIN_TERM_LEN = 3
_MAX_TERM_LEN = 40
_DIGITS_ONLY = re.compile(r"^\d+$")


@dataclass(frozen=True)
class SecondHopConfig:
    """How wide the second hop reaches.

    ``feedback_chunks`` is small on purpose: the assumption being made is that
    the top of the first ranking is right, and that assumption gets weaker fast.
    """

    feedback_chunks: int = 2
    n_terms: int = 8
    weight: float = 0.5
    rrf_k: int = 60


def select_expansion_terms(question: str, candidates: list[RetrievedChunk],
                           config: SecondHopConfig) -> list[str]:
    """Terms that are frequent in the top chunks and absent from the question.

    A term already in the question adds nothing: the first retrieval has
    already used it. What is wanted is the vocabulary the answer's neighbourhood
    uses and the asker did not.
    """
    asked = set(tokenize(question, remove_stopwords=True))
    counts: Counter[str] = Counter()
    for chunk in candidates[: max(0, config.feedback_chunks)]:
        for term in tokenize(chunk.chunk.text, remove_stopwords=True):
            if term in asked or term in STOPWORDS:
                continue
            if not (_MIN_TERM_LEN <= len(term) <= _MAX_TERM_LEN):
                continue
            if _DIGITS_ONLY.match(term):
                continue
            counts[term] += 1

    # Ties broken alphabetically so the expansion — and therefore the whole
    # second hop — is deterministic and can be asserted in a test.
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [term for term, _ in ranked[: max(0, config.n_terms)]]


def build_expansion_query(question: str, terms: list[str]) -> str:
    """The second hop still asks the user's question, with more vocabulary."""
    return question if not terms else f"{question} {' '.join(terms)}"


def fuse_hops(first: list[RetrievedChunk], second: list[RetrievedChunk],
              config: SecondHopConfig, top_k: int) -> list[RetrievedChunk]:
    """Merge two hops with RRF, keeping the first hop's evidence dominant.

    The same fusion rule as dense-plus-sparse, for the same reason: the two
    lists are not on a comparable score scale, and one of them was produced from
    a query the user never wrote.
    """
    from app.retrieval.hybrid import reciprocal_rank_fusion

    by_id: dict[str, RetrievedChunk] = {}
    for hit in list(second) + list(first):   # first hop wins on conflict
        by_id[hit.chunk.chunk_id] = hit

    fused = reciprocal_rank_fusion(
        {"hop1": [h.chunk.chunk_id for h in first],
         "hop2": [h.chunk.chunk_id for h in second]},
        k=config.rrf_k,
        weights={"hop1": 1.0, "hop2": config.weight},
    )

    out: list[RetrievedChunk] = []
    for rank, (chunk_id, score) in enumerate(fused[:top_k], start=1):
        found = by_id.get(chunk_id)
        if found is None:    # cannot happen; a fused ID came from one of the lists
            continue
        hit = found
        components = dict(hit.component_scores)
        components["second_hop"] = round(score, 6)
        out.append(hit.model_copy(update={
            "score": score,
            "rank": rank,
            "retriever": f"{hit.retriever}+2hop",
            "component_scores": components,
        }))
    return out
