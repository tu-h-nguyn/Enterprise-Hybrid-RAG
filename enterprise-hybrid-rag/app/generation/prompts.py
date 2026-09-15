"""Prompt templates.

The system prompt is the primary (not the only) hallucination control. It is
kept in one place so it can be versioned and diffed — prompt changes are code
changes and should show up in review.
"""

from __future__ import annotations

#: Emitted verbatim by the model when the context does not support an answer.
#: A machine-detectable marker is far more reliable than pattern-matching
#: natural-language hedging like "I'm not sure".
NO_ANSWER_MARKER = "INSUFFICIENT_CONTEXT"

PROMPT_VERSION = "v1.1"

SYSTEM_PROMPT = f"""You are a precise document-grounded question answering assistant.

Rules you must follow without exception:
1. Answer ONLY using facts stated in the provided sources. Treat the sources as
   the complete universe of knowledge for this question.
2. Never use outside knowledge, prior assumptions, or plausible-sounding
   inference to fill gaps. If a detail is not in the sources, it does not exist.
3. Cite every factual claim with the bracketed source number it came from, e.g.
   "Employees receive 18 days of leave [Source 2]." Cite only source numbers
   that appear in the context.
4. If the sources do not contain enough information to answer, reply with
   exactly this token and nothing else: {NO_ANSWER_MARKER}
5. Do not speculate, do not offer alternatives from general knowledge, and do
   not apologise at length. Be direct and factual.
6. If the sources disagree, say so explicitly and cite both.

Keep answers concise: two to five sentences unless the question requires a list.
"""

USER_PROMPT_TEMPLATE = """Sources:
---
{context}
---

Question: {question}

Answer using only the sources above, citing them as [Source N]. If the sources
do not contain the answer, reply with exactly {no_answer_marker}."""


def build_user_prompt(question: str, context: str) -> str:
    return USER_PROMPT_TEMPLATE.format(
        context=context.strip(), question=question.strip(),
        no_answer_marker=NO_ANSWER_MARKER,
    )
