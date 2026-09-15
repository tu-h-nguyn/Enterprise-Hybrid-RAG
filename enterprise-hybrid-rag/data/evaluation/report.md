# Retrieval benchmark report

Generated: `2026-09-15T05:28:36+00:00`  
Dataset: `northwind_demo` — 50 questions (43 answerable, 7 unanswerable)  
Index: 22 documents, 44 chunks at `chunk_size_tokens=500` / `chunk_overlap_tokens=100`

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
| Dense only | 0.5155 | 0.7946 | 0.8295 | 0.8527 | 0.6906 | 0.7199 | 0.2047 | 9.15 | 10.09 |
| BM25 only | 0.5969 | 0.8256 | 0.9070 | 0.9302 | 0.7694 | 0.8001 | 0.2233 | 0.50 | 0.57 |
| Hybrid (RRF) | 0.5620 | 0.7946 | 0.8372 | 0.8837 | 0.7362 | 0.7513 | 0.2093 | 9.41 | 11.03 |
| Hybrid + Reranker | 0.7016 | 0.8953 | 0.9186 | 0.9767 | 0.8568 | 0.8610 | 0.2233 | 783.67 | 850.19 |

### Recall@1 by question type

This is where the retrievers actually differ: keyword questions turn on rare reference codes, paraphrased questions share no vocabulary with the source passage.

| Configuration | factual | keyword | multi_step | paraphrased | terminology |
|---|---|---|---|---|---|
| Dense only | 0.8571 | 0.2857 | 0.3095 | 0.3333 | 0.5000 |
| BM25 only | 0.8571 | 0.8571 | 0.3810 | 0.0000 | 0.8333 |
| Hybrid (RRF) | 0.8571 | 0.2857 | 0.4524 | 0.2222 | 0.8333 |
| Hybrid + Reranker | 0.9286 | 1.0000 | 0.4524 | 0.2222 | 0.8333 |

### MRR by question type

| Configuration | factual | keyword | multi_step | paraphrased | terminology |
|---|---|---|---|---|---|
| Dense only | 0.9286 | 0.3492 | 0.8571 | 0.4907 | 0.6389 |
| BM25 only | 0.9107 | 0.9286 | 0.9286 | 0.2037 | 0.9167 |
| Hybrid (RRF) | 0.9286 | 0.4762 | 1.0000 | 0.3472 | 0.8667 |
| Hybrid + Reranker | 0.9643 | 1.0000 | 1.0000 | 0.4270 | 0.9167 |

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
| groundedness | 0.6437 |
| answer_f1 | 0.3138 |
| span_coverage | 0.5078 |
| citation_precision | 0.6512 |
| uncited_sentence_rate | 0.3488 |
| mean_citations | 0.8372 |
| over_refusal_rate | 0.3488 |
| correct_refusal_rate | 0.8571 |
| mean_latency_ms | 798.7364 |

---

Produced by `scripts/benchmark.py`. Raw per-question records are in `results.json`.
