# Retrieval benchmark report

Generated: `2026-09-15T15:58:55+00:00`  
Dataset: `northwind_demo` — 50 questions (43 answerable, 7 unanswerable)  
Index: 22 documents, 77 chunks at `chunk_size_tokens=256` / `chunk_overlap_tokens=51`

## Backends that produced these numbers

| Component | Backend | Detail |
|---|---|---|
| Embedding | `sentence_transformers` | sentence-transformers/all-MiniLM-L6-v2, dim 384 |
| Reranker | `cross_encoder` | neural: True |
| LLM | `extractive` | extractive-offline |
| Vector store | `chroma` | |

> **Read this before quoting any number below.**
>
> * Generation used the **offline extractive backend**, which selects supporting sentences from the context rather than generating text. Generation metrics below describe that behaviour, not an LLM's.

## Question type mix

| Type | Count |
|---|---|
| factual | 14 |
| keyword | 7 |
| multi_step | 7 |
| paraphrased | 9 |
| terminology | 6 |
| unanswerable | 7 |

## Retrieval comparison

Recall is computed over the answer-span ground truth resolved at evaluation time. Unanswerable questions carry no relevant chunks and are excluded from these metrics (they are scored by refusal behaviour under *Generation*).

| Configuration | R@1 | R@3 | R@5 | R@10 | MRR | nDCG@5 | P@5 | mean ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|
| Dense only | 0.5853 | 0.8256 | 0.8953 | 0.9302 | 0.7740 | 0.7869 | 0.2186 | 15.61 | 17.56 |
| BM25 only | 0.6434 | 0.8023 | 0.8023 | 0.8023 | 0.7597 | 0.7660 | 0.2000 | 0.74 | 0.84 |
| Hybrid (RRF) | 0.6434 | 0.8023 | 0.8488 | 1.0000 | 0.7910 | 0.7862 | 0.2093 | 15.91 | 17.04 |
| Hybrid + Reranker | 0.7946 | 0.8953 | 0.9070 | 1.0000 | 0.9085 | 0.8937 | 0.2233 | 693.19 | 758.69 |
| Hybrid + Rerank + 2-hop | 0.7946 | 0.8953 | 0.9070 | 1.0000 | 0.9085 | 0.8937 | 0.2233 | 711.21 | 774.10 |

### Recall@1 by question type

This is where the retrievers actually differ: keyword questions turn on rare reference codes, paraphrased questions share no vocabulary with the source passage.

| Configuration | factual | keyword | multi_step | paraphrased | terminology |
|---|---|---|---|---|---|
| Dense only | 0.9286 | 0.4286 | 0.4524 | 0.4444 | 0.3333 |
| BM25 only | 0.8571 | 1.0000 | 0.3810 | 0.1111 | 0.8333 |
| Hybrid (RRF) | 1.0000 | 0.5714 | 0.3810 | 0.2222 | 0.8333 |
| Hybrid + Reranker | 1.0000 | 1.0000 | 0.4524 | 0.4444 | 1.0000 |
| Hybrid + Rerank + 2-hop | 1.0000 | 1.0000 | 0.4524 | 0.4444 | 1.0000 |

### MRR by question type

| Configuration | factual | keyword | multi_step | paraphrased | terminology |
|---|---|---|---|---|---|
| Dense only | 0.9643 | 0.4762 | 1.0000 | 0.6056 | 0.6667 |
| BM25 only | 0.9167 | 1.0000 | 0.9286 | 0.1111 | 0.8889 |
| Hybrid (RRF) | 1.0000 | 0.6766 | 0.9286 | 0.3825 | 0.8889 |
| Hybrid + Reranker | 1.0000 | 1.0000 | 1.0000 | 0.5626 | 1.0000 |
| Hybrid + Rerank + 2-hop | 1.0000 | 1.0000 | 1.0000 | 0.5626 | 1.0000 |

## Chunk-size ablation

The corpus is re-ingested and both indexes are rebuilt at each size, and ground truth is re-resolved against the new chunks (labels are answer spans, not chunk IDs, precisely so this comparison stays valid).

| Chunk size | Overlap | Chunks | Mean tokens | Embed dim | Configuration | R@1 | R@5 | MRR | nDCG@5 |
|---|---|---|---|---|---|---|---|---|---|
| 256 | 51 | 77 | 192.6 | 384 | Dense only | 0.5853 | 0.8953 | 0.7740 | 0.7869 |
|  |  |  |  |  | BM25 only | 0.6434 | 0.8023 | 0.7597 | 0.7660 |
|  |  |  |  |  | Hybrid (RRF) | 0.6434 | 0.8488 | 0.7910 | 0.7862 |
|  |  |  |  |  | Hybrid + Reranker | 0.7946 | 0.9070 | 0.9085 | 0.8937 |
| 500 | 100 | 44 | 337.4 | 384 | Dense only | 0.5155 | 0.8295 | 0.6906 | 0.7199 |
|  |  |  |  |  | BM25 only | 0.5969 | 0.9070 | 0.7694 | 0.8001 |
|  |  |  |  |  | Hybrid (RRF) | 0.5620 | 0.8372 | 0.7362 | 0.7513 |
|  |  |  |  |  | Hybrid + Reranker | 0.7016 | 0.9186 | 0.8568 | 0.8610 |
| 800 | 160 | 25 | 594.2 | 384 | Dense only | 0.5775 | 0.7791 | 0.7225 | 0.7196 |
|  |  |  |  |  | BM25 only | 0.6318 | 0.8837 | 0.7966 | 0.8090 |
|  |  |  |  |  | Hybrid (RRF) | 0.6318 | 0.8527 | 0.7870 | 0.7850 |
|  |  |  |  |  | Hybrid + Reranker | 0.7481 | 0.9535 | 0.8747 | 0.8857 |

## Generation

Provider: `extractive` / `extractive-offline`, top_k=5, LLM judge used: False

| Metric | Value |
|---|---|
| n_questions | 50 |
| n_answerable | 43 |
| n_unanswerable | 7 |
| groundedness | 0.6434 |
| answer_f1 | 0.3137 |
| span_coverage | 0.5078 |
| citation_precision | 0.6512 |
| uncited_sentence_rate | 0.3488 |
| mean_citations | 0.8372 |
| over_refusal_rate | 0.3488 |
| correct_refusal_rate | 0.8571 |
| mean_latency_ms | 691.8794 |

---

Produced by `scripts/benchmark.py`. Raw per-question records are in `results.json`.
