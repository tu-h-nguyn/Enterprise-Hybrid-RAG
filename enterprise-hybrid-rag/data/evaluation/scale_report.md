# Corpus-scale experiment

Generated 2026-09-15T12:57:16+00:00 · 50 questions (43 answerable), held fixed throughout.

* Embedder: `sentence_transformers` / `sentence-transformers/all-MiniLM-L6-v2` (dim 384)
* Reranker: `cross_encoder` (neural: True)
* Vector stores: `numpy`, `chroma`
* Distractor corpus: seed 20250915, 3400 documents available

> ⚠️ Generation used the **offline extractive backend**, which selects supporting sentences from the context rather than generating text. Generation metrics below describe that behaviour, not an LLM's.

## Method

The 50 labelled questions, their answer spans and every retrieval setting are
held fixed. Only the size of the haystack changes: in-domain distractor
documents on the same topics and in the same register as the real corpus, for
other fictional companies. Sizes are nested, so each corpus contains the
previous one. The run aborts if a distractor contains a labelled answer span,
or if the resolved gold chunks differ from the smallest corpus.

Every size is run against both vector stores. `numpy` is exhaustive cosine and
is therefore exact; `chroma` is HNSW and is approximate. The difference between
the two rows at the same size is the recall the approximate index gives up.

`distr@1` is the share of answerable questions whose top hit is a distractor;
`distr%@5` is the mean share of the top 5 they occupy. These are the evidence
that the added documents are genuinely competitive rather than easy padding.

## Results

| Chunk | Store | Chunks | Gold share | Configuration | R@1 | R@5 | MRR | mean ms | distr@1 | distr%@5 |
|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 500 | `numpy` | 44 | 68.18% | Dense only | 0.5155 | 0.8295 | 0.6906 | 10.73 | 0.0000 | 0.0000 |
|  |  |  |  | BM25 only | 0.5969 | 0.9070 | 0.7694 | 0.19 | 0.0000 | 0.0000 |
|  |  |  |  | Hybrid (RRF) | 0.5620 | 0.8372 | 0.7362 | 11.11 | 0.0000 | 0.0000 |
|  |  |  |  | Hybrid + Reranker | 0.7016 | 0.9186 | 0.8568 | 1606.19 | 0.0000 | 0.0000 |
| 500 | `numpy` | 235 | 12.77% | Dense only | 0.4574 | 0.6783 | 0.5999 | 10.77 | 0.2326 | 0.5535 |
|  |  |  |  | BM25 only | 0.5504 | 0.7791 | 0.7062 | 0.46 | 0.0698 | 0.3907 |
|  |  |  |  | Hybrid (RRF) | 0.5039 | 0.7597 | 0.6716 | 12.00 | 0.2558 | 0.5209 |
|  |  |  |  | Hybrid + Reranker | 0.6550 | 0.8721 | 0.8101 | 1580.48 | 0.1163 | 0.4233 |
| 500 | `numpy` | 936 | 3.21% | Dense only | 0.4109 | 0.5969 | 0.5478 | 10.89 | 0.3488 | 0.6837 |
|  |  |  |  | BM25 only | 0.5504 | 0.7791 | 0.6991 | 1.56 | 0.0698 | 0.4372 |
|  |  |  |  | Hybrid (RRF) | 0.4806 | 0.7713 | 0.6565 | 13.83 | 0.3256 | 0.5674 |
|  |  |  |  | Hybrid + Reranker | 0.6318 | 0.8566 | 0.7950 | 1568.24 | 0.1860 | 0.5581 |
| 500 | `numpy` | 4658 | 0.64% | Dense only | 0.3643 | 0.5736 | 0.4957 | 21.50 | 0.4186 | 0.7814 |
|  |  |  |  | BM25 only | 0.5736 | 0.7558 | 0.7078 | 9.33 | 0.0698 | 0.4279 |
|  |  |  |  | Hybrid (RRF) | 0.4457 | 0.7597 | 0.6287 | 45.55 | 0.3953 | 0.6093 |
|  |  |  |  | Hybrid + Reranker | 0.6318 | 0.8566 | 0.7868 | 1631.75 | 0.1628 | 0.6093 |
| 500 | `chroma` | 44 | 68.18% | Dense only | 0.5155 | 0.8295 | 0.6906 | 15.60 | 0.0000 | 0.0000 |
|  |  |  |  | BM25 only | 0.5969 | 0.9070 | 0.7694 | 0.67 | 0.0000 | 0.0000 |
|  |  |  |  | Hybrid (RRF) | 0.5620 | 0.8372 | 0.7362 | 16.15 | 0.0000 | 0.0000 |
|  |  |  |  | Hybrid + Reranker | 0.7016 | 0.9186 | 0.8568 | 1615.28 | 0.0000 | 0.0000 |
| 500 | `chroma` | 235 | 12.77% | Dense only | 0.4574 | 0.6783 | 0.5999 | 15.76 | 0.2326 | 0.5535 |
|  |  |  |  | BM25 only | 0.5504 | 0.7791 | 0.7062 | 0.99 | 0.0698 | 0.3907 |
|  |  |  |  | Hybrid (RRF) | 0.5039 | 0.7597 | 0.6716 | 16.84 | 0.2558 | 0.5209 |
|  |  |  |  | Hybrid + Reranker | 0.6550 | 0.8721 | 0.8101 | 1588.32 | 0.1163 | 0.4233 |
| 500 | `chroma` | 936 | 3.21% | Dense only | 0.4109 | 0.5969 | 0.5478 | 16.02 | 0.3488 | 0.6837 |
|  |  |  |  | BM25 only | 0.5504 | 0.7791 | 0.6991 | 2.14 | 0.0698 | 0.4372 |
|  |  |  |  | Hybrid (RRF) | 0.4806 | 0.7713 | 0.6565 | 18.85 | 0.3256 | 0.5674 |
|  |  |  |  | Hybrid + Reranker | 0.6318 | 0.8566 | 0.7950 | 1566.78 | 0.1860 | 0.5581 |
| 500 | `chroma` | 4658 | 0.64% | Dense only | 0.3643 | 0.5504 | 0.4841 | 17.93 | 0.4186 | 0.7860 |
|  |  |  |  | BM25 only | 0.5736 | 0.7558 | 0.7078 | 10.12 | 0.0698 | 0.4279 |
|  |  |  |  | Hybrid (RRF) | 0.4457 | 0.7364 | 0.6227 | 31.27 | 0.3953 | 0.6233 |
|  |  |  |  | Hybrid + Reranker | 0.6318 | 0.8566 | 0.7868 | 1581.70 | 0.1628 | 0.6093 |
| 256 | `numpy` | 77 | 51.95% | Dense only | 0.5853 | 0.8953 | 0.7740 | 10.84 | 0.0000 | 0.0000 |
|  |  |  |  | BM25 only | 0.6434 | 0.8023 | 0.7597 | 0.22 | 0.0000 | 0.0000 |
|  |  |  |  | Hybrid (RRF) | 0.6434 | 0.8488 | 0.7910 | 11.46 | 0.0000 | 0.0000 |
|  |  |  |  | Hybrid + Reranker | 0.7946 | 0.9070 | 0.9085 | 796.30 | 0.0000 | 0.0000 |
| 256 | `numpy` | 423 | 9.46% | Dense only | 0.4690 | 0.7209 | 0.6551 | 11.01 | 0.2791 | 0.5302 |
|  |  |  |  | BM25 only | 0.6085 | 0.7558 | 0.7306 | 0.70 | 0.0465 | 0.3349 |
|  |  |  |  | Hybrid (RRF) | 0.5853 | 0.8023 | 0.7369 | 12.53 | 0.2326 | 0.4605 |
|  |  |  |  | Hybrid + Reranker | 0.7248 | 0.8760 | 0.8384 | 768.20 | 0.1628 | 0.4651 |
| 256 | `numpy` | 1704 | 2.35% | Dense only | 0.4341 | 0.6357 | 0.6061 | 25.42 | 0.3256 | 0.6698 |
|  |  |  |  | BM25 only | 0.5969 | 0.7558 | 0.7161 | 3.28 | 0.0465 | 0.3721 |
|  |  |  |  | Hybrid (RRF) | 0.5620 | 0.8178 | 0.7324 | 37.77 | 0.2558 | 0.5488 |
|  |  |  |  | Hybrid + Reranker | 0.7248 | 0.8411 | 0.8289 | 824.89 | 0.1628 | 0.5581 |
| 256 | `numpy` | 8548 | 0.47% | Dense only | 0.3798 | 0.5426 | 0.5223 | 25.50 | 0.4186 | 0.7535 |
|  |  |  |  | BM25 only | 0.5969 | 0.7558 | 0.7158 | 17.46 | 0.0465 | 0.3302 |
|  |  |  |  | Hybrid (RRF) | 0.4922 | 0.7829 | 0.6805 | 52.78 | 0.3488 | 0.5395 |
|  |  |  |  | Hybrid + Reranker | 0.7248 | 0.8062 | 0.8285 | 816.11 | 0.1628 | 0.5814 |

## What approximate search costs

Recall@1 under exhaustive cosine (`numpy`) minus Recall@1 under HNSW
(`chroma`), at the same corpus size. A positive number is recall the
approximate index lost. BM25 never touches the vector store, so its
rows must read zero — they are the control that says nothing else
changed between the two runs.

| Chunks | Configuration | exact R@1 | HNSW R@1 | lost |
|---:|---|---:|---:|---:|
| 44 | BM25 only | 0.5969 | 0.5969 | +0.0000 |
| 44 | Dense only | 0.5155 | 0.5155 | +0.0000 |
| 44 | Hybrid (RRF) | 0.5620 | 0.5620 | +0.0000 |
| 44 | Hybrid + Reranker | 0.7016 | 0.7016 | +0.0000 |
| 235 | BM25 only | 0.5504 | 0.5504 | +0.0000 |
| 235 | Dense only | 0.4574 | 0.4574 | +0.0000 |
| 235 | Hybrid (RRF) | 0.5039 | 0.5039 | +0.0000 |
| 235 | Hybrid + Reranker | 0.6550 | 0.6550 | +0.0000 |
| 936 | BM25 only | 0.5504 | 0.5504 | +0.0000 |
| 936 | Dense only | 0.4109 | 0.4109 | +0.0000 |
| 936 | Hybrid (RRF) | 0.4806 | 0.4806 | +0.0000 |
| 936 | Hybrid + Reranker | 0.6318 | 0.6318 | +0.0000 |
| 4658 | BM25 only | 0.5736 | 0.5736 | +0.0000 |
| 4658 | Dense only | 0.3643 | 0.3643 | +0.0000 |
| 4658 | Hybrid (RRF) | 0.4457 | 0.4457 | +0.0000 |
| 4658 | Hybrid + Reranker | 0.6318 | 0.6318 | +0.0000 |

## Does the chunk size that wins on a small corpus still win?

Recall@1 for the production configuration (hybrid + reranker) under
exhaustive search, at each chunk size. Corpus sizes are matched by how
many distractor *documents* were added, because a different chunk size
turns the same documents into a different number of chunks — that
count is shown for each.

| Distractor docs | 256 tok — chunks / R@1 / MRR | 500 tok — chunks / R@1 / MRR | best |
|---:|---:|---:|---|
| 0 | 77 / 0.7946 / 0.9085 | 44 / 0.7016 / 0.8568 | 256 tok |
| 140 | 423 / 0.7248 / 0.8384 | 235 / 0.6550 / 0.8101 | 256 tok |
| 655 | 1704 / 0.7248 / 0.8289 | 936 / 0.6318 / 0.7950 | 256 tok |
| 3400 | 8548 / 0.7248 / 0.8285 | 4658 / 0.6318 / 0.7868 | 256 tok |

The shipped default is 500 tokens. A chunk size that wins only at the smallest corpus is a finding about the corpus, not about chunking — which is the whole reason this table is not just the chunk-size ablation run again.

## Index build cost

| Store | Chunks | Documents | Embedding dim | Index build (s) |
|---|---:|---:|---:|---:|
| `numpy` | 44 | 22 | 384 | 8.03 |
| `numpy` | 235 | 162 | 384 | 10.86 |
| `numpy` | 936 | 677 | 384 | 37.30 |
| `numpy` | 4658 | 3422 | 384 | 179.66 |
| `chroma` | 44 | 22 | 384 | 3.73 |
| `chroma` | 235 | 162 | 384 | 10.96 |
| `chroma` | 936 | 677 | 384 | 38.14 |
| `chroma` | 4658 | 3422 | 384 | 182.47 |
| `numpy` | 77 | 22 | 384 | 4.14 |
| `numpy` | 423 | 162 | 384 | 14.45 |
| `numpy` | 1704 | 677 | 384 | 53.35 |
| `numpy` | 8548 | 3422 | 384 | 261.77 |

Distractors are synthetic. They are hard negatives by construction — same
topics, same register, same policy vocabulary — but a real corpus of this
size would contain both easier negatives (off-topic material) and harder
ones (near-duplicate revisions of the same policy).
