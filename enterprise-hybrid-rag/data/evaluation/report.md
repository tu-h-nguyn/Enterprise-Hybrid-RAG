# Retrieval benchmark report

Generated: `2026-09-15T02:54:56+00:00`  
Dataset: `northwind_demo` — 50 questions (43 answerable, 7 unanswerable)  
Index: 22 documents, 44 chunks at `chunk_size_tokens=500` / `chunk_overlap_tokens=100`

## Backends that produced these numbers

| Component | Backend | Detail |
|---|---|---|
| Embedding | `tfidf_svd` | tfidf(word1-2 + char3-5)+svd, dim 43 |
| Reranker | `lexical` | neural: False |
| LLM | `extractive` | extractive-offline |
| Vector store | `chroma` | |

> **Read this before quoting any number below.**
>
> * Embeddings came from the **offline `tfidf_svd` fallback** (`tfidf(word1-2 + char3-5)+svd`, dim 43), not a neural sentence encoder. Dense numbers here are lexical-semantic and are a floor, not a measurement of `all-MiniLM-L6-v2`.
> * The reranker was the **non-neural `lexical` fallback**, a feature-based scorer. It is not a cross-encoder and its numbers must not be reported as `ms-marco-MiniLM-L-6-v2` numbers.
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
| Dense only | 0.6550 | 0.8837 | 0.8837 | 0.9302 | 0.8128 | 0.8261 | 0.2186 | 8.34 | 9.51 |
| BM25 only | 0.5969 | 0.8256 | 0.9070 | 0.9302 | 0.7694 | 0.8001 | 0.2233 | 0.62 | 0.78 |
| Hybrid (RRF) | 0.6318 | 0.8605 | 0.8837 | 0.9070 | 0.7953 | 0.8112 | 0.2186 | 7.49 | 8.11 |
| Hybrid + Reranker | 0.6667 | 0.8023 | 0.8488 | 0.8837 | 0.7884 | 0.7956 | 0.2093 | 10.20 | 11.16 |

### Recall@1 by question type

This is where the retrievers actually differ: keyword questions turn on rare reference codes, paraphrased questions share no vocabulary with the source passage.

| Configuration | factual | keyword | multi_step | paraphrased | terminology |
|---|---|---|---|---|---|
| Dense only | 0.8571 | 1.0000 | 0.4524 | 0.1111 | 0.8333 |
| BM25 only | 0.8571 | 0.8571 | 0.3810 | 0.0000 | 0.8333 |
| Hybrid (RRF) | 0.8571 | 1.0000 | 0.4524 | 0.0000 | 0.8333 |
| Hybrid + Reranker | 1.0000 | 1.0000 | 0.3810 | 0.1111 | 0.6667 |

### MRR by question type

| Configuration | factual | keyword | multi_step | paraphrased | terminology |
|---|---|---|---|---|---|
| Dense only | 0.9031 | 1.0000 | 1.0000 | 0.3122 | 0.9167 |
| BM25 only | 0.9107 | 0.9286 | 0.9286 | 0.2037 | 0.9167 |
| Hybrid (RRF) | 0.9048 | 1.0000 | 1.0000 | 0.2259 | 0.9167 |
| Hybrid + Reranker | 1.0000 | 1.0000 | 0.9286 | 0.2111 | 0.7500 |

## Chunk-size ablation

The corpus is re-ingested and both indexes are rebuilt at each size, and ground truth is re-resolved against the new chunks (labels are answer spans, not chunk IDs, precisely so this comparison stays valid).

| Chunk size | Overlap | Chunks | Mean tokens | Embed dim | Configuration | R@1 | R@5 | MRR | nDCG@5 |
|---|---|---|---|---|---|---|---|---|---|
| 256 | 51 | 77 | 192.6 | 76 | Dense only | 0.6550 | 0.8023 | 0.7785 | 0.7746 |
|  |  |  |  |  | BM25 only | 0.6434 | 0.8023 | 0.7597 | 0.7660 |
|  |  |  |  |  | Hybrid (RRF) | 0.6550 | 0.8023 | 0.7784 | 0.7727 |
|  |  |  |  |  | Hybrid + Reranker | 0.6434 | 0.7791 | 0.7541 | 0.7468 |
| 500 | 100 | 44 | 337.4 | 43 | Dense only | 0.6550 | 0.8837 | 0.8128 | 0.8261 |
|  |  |  |  |  | BM25 only | 0.5969 | 0.9070 | 0.7694 | 0.8001 |
|  |  |  |  |  | Hybrid (RRF) | 0.6318 | 0.8837 | 0.7953 | 0.8112 |
|  |  |  |  |  | Hybrid + Reranker | 0.6667 | 0.8488 | 0.7884 | 0.7956 |
| 800 | 160 | 25 | 594.2 | 24 | Dense only | 0.6550 | 0.9070 | 0.8134 | 0.8351 |
|  |  |  |  |  | BM25 only | 0.6318 | 0.8837 | 0.7966 | 0.8090 |
|  |  |  |  |  | Hybrid (RRF) | 0.6550 | 0.8837 | 0.8117 | 0.8243 |
|  |  |  |  |  | Hybrid + Reranker | 0.6550 | 0.8721 | 0.7953 | 0.8064 |

## Generation

Provider: `extractive` / `extractive-offline`, top_k=5, LLM judge used: False

| Metric | Value |
|---|---|
| n_questions | 50 |
| n_answerable | 43 |
| n_unanswerable | 7 |
| groundedness | 0.6398 |
| answer_f1 | 0.3000 |
| span_coverage | 0.5078 |
| citation_precision | 0.6512 |
| uncited_sentence_rate | 0.3488 |
| mean_citations | 0.9070 |
| over_refusal_rate | 0.3488 |
| correct_refusal_rate | 0.8571 |
| mean_latency_ms | 11.6922 |

---

Produced by `scripts/benchmark.py`. Raw per-question records are in `results.json`.
