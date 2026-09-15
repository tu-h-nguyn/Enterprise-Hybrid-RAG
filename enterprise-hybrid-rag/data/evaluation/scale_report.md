# Corpus-scale experiment

Generated 2026-09-15T10:25:47+00:00 · 50 questions (43 answerable), held fixed throughout.

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

| Store | Chunks | Gold share | Configuration | R@1 | R@5 | MRR | mean ms | distr@1 | distr%@5 |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| `numpy` | 44 | 68.18% | Dense only | 0.5155 | 0.8295 | 0.6906 | 11.89 | 0.0000 | 0.0000 |
|  |  |  | BM25 only | 0.5969 | 0.9070 | 0.7694 | 0.22 | 0.0000 | 0.0000 |
|  |  |  | Hybrid (RRF) | 0.5620 | 0.8372 | 0.7362 | 11.90 | 0.0000 | 0.0000 |
|  |  |  | Hybrid + Reranker | 0.7016 | 0.9186 | 0.8568 | 1418.01 | 0.0000 | 0.0000 |
| `numpy` | 235 | 12.77% | Dense only | 0.4574 | 0.6783 | 0.5999 | 11.04 | 0.2326 | 0.5535 |
|  |  |  | BM25 only | 0.5504 | 0.7791 | 0.7062 | 0.51 | 0.0698 | 0.3907 |
|  |  |  | Hybrid (RRF) | 0.5039 | 0.7597 | 0.6716 | 11.91 | 0.2558 | 0.5209 |
|  |  |  | Hybrid + Reranker | 0.6550 | 0.8721 | 0.8101 | 1386.93 | 0.1163 | 0.4233 |
| `numpy` | 936 | 3.21% | Dense only | 0.4109 | 0.5969 | 0.5478 | 11.53 | 0.3488 | 0.6837 |
|  |  |  | BM25 only | 0.5504 | 0.7791 | 0.6991 | 1.91 | 0.0698 | 0.4372 |
|  |  |  | Hybrid (RRF) | 0.4806 | 0.7713 | 0.6565 | 14.35 | 0.3256 | 0.5674 |
|  |  |  | Hybrid + Reranker | 0.6318 | 0.8566 | 0.7950 | 1366.73 | 0.1860 | 0.5581 |
| `numpy` | 4658 | 0.64% | Dense only | 0.3643 | 0.5736 | 0.4957 | 23.86 | 0.4186 | 0.7814 |
|  |  |  | BM25 only | 0.5736 | 0.7558 | 0.7078 | 12.45 | 0.0698 | 0.4279 |
|  |  |  | Hybrid (RRF) | 0.4457 | 0.7597 | 0.6287 | 39.20 | 0.3953 | 0.6093 |
|  |  |  | Hybrid + Reranker | 0.6318 | 0.8566 | 0.7868 | 1432.38 | 0.1628 | 0.6093 |
| `chroma` | 44 | 68.18% | Dense only | 0.5155 | 0.8295 | 0.6906 | 16.54 | 0.0000 | 0.0000 |
|  |  |  | BM25 only | 0.5969 | 0.9070 | 0.7694 | 0.74 | 0.0000 | 0.0000 |
|  |  |  | Hybrid (RRF) | 0.5620 | 0.8372 | 0.7362 | 17.17 | 0.0000 | 0.0000 |
|  |  |  | Hybrid + Reranker | 0.7016 | 0.9186 | 0.8568 | 1404.04 | 0.0000 | 0.0000 |
| `chroma` | 235 | 12.77% | Dense only | 0.4574 | 0.6783 | 0.5999 | 17.04 | 0.2326 | 0.5535 |
|  |  |  | BM25 only | 0.5504 | 0.7791 | 0.7062 | 1.16 | 0.0698 | 0.3907 |
|  |  |  | Hybrid (RRF) | 0.5039 | 0.7597 | 0.6716 | 18.43 | 0.2558 | 0.5209 |
|  |  |  | Hybrid + Reranker | 0.6550 | 0.8721 | 0.8101 | 1377.85 | 0.1163 | 0.4233 |
| `chroma` | 936 | 3.21% | Dense only | 0.4109 | 0.5969 | 0.5478 | 17.02 | 0.3488 | 0.6837 |
|  |  |  | BM25 only | 0.5504 | 0.7791 | 0.6991 | 2.76 | 0.0698 | 0.4372 |
|  |  |  | Hybrid (RRF) | 0.4806 | 0.7713 | 0.6565 | 20.05 | 0.3256 | 0.5674 |
|  |  |  | Hybrid + Reranker | 0.6318 | 0.8566 | 0.7950 | 1361.12 | 0.1860 | 0.5581 |
| `chroma` | 4658 | 0.64% | Dense only | 0.3643 | 0.5504 | 0.4841 | 19.08 | 0.4186 | 0.7860 |
|  |  |  | BM25 only | 0.5736 | 0.7558 | 0.7078 | 14.76 | 0.0698 | 0.4279 |
|  |  |  | Hybrid (RRF) | 0.4457 | 0.7364 | 0.6203 | 34.56 | 0.3953 | 0.6186 |
|  |  |  | Hybrid + Reranker | 0.6318 | 0.8566 | 0.7868 | 1385.34 | 0.1628 | 0.6047 |

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

## Index build cost

| Store | Chunks | Documents | Embedding dim | Index build (s) |
|---|---:|---:|---:|---:|
| `numpy` | 44 | 22 | 384 | 7.47 |
| `numpy` | 235 | 162 | 384 | 8.79 |
| `numpy` | 936 | 677 | 384 | 32.01 |
| `numpy` | 4658 | 3422 | 384 | 156.61 |
| `chroma` | 44 | 22 | 384 | 2.77 |
| `chroma` | 235 | 162 | 384 | 10.09 |
| `chroma` | 936 | 677 | 384 | 32.77 |
| `chroma` | 4658 | 3422 | 384 | 159.83 |

Distractors are synthetic. They are hard negatives by construction — same
topics, same register, same policy vocabulary — but a real corpus of this
size would contain both easier negatives (off-topic material) and harder
ones (near-duplicate revisions of the same policy).
