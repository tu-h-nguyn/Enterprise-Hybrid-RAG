# Corpus-scale experiment

Generated 2026-09-15T13:13:21+00:00 · 50 questions (43 answerable), held fixed throughout.

* Embedder: `tfidf_svd` / `tfidf(word1-2 + char3-5)+svd` (dim 76)
* Reranker: `lexical` (neural: False)
* Vector stores: `numpy`, `chroma`
* Distractor corpus: seed 20250915, 3400 documents available

> ⚠️ Embeddings came from the **offline `tfidf_svd` fallback** (`tfidf(word1-2 + char3-5)+svd`, dim 76), not a neural sentence encoder. Dense numbers here are lexical-semantic and are a floor, not a measurement of `all-MiniLM-L6-v2`.

> ⚠️ The reranker was the **non-neural `lexical` fallback**, a feature-based scorer. It is not a cross-encoder and its numbers must not be reported as `ms-marco-MiniLM-L-6-v2` numbers.

> ⚠️ Generation used the **offline extractive backend**, which selects supporting sentences from the context rather than generating text. Generation metrics below describe that behaviour, not an LLM's.

> ⚠️ The embedding dimension was not constant across sizes ([43, 76, 234, 384]). The offline `tfidf_svd` fallback derives its width from the corpus, so these rows vary the encoder as well as the corpus and the two effects cannot be separated. A run on `sentence_transformers` holds the dimension at 384 and is the one to read.

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
| 256 | `numpy` | 77 | 51.95% | Dense only | 0.6550 | 0.8023 | 0.7785 | 3.91 | 0.0000 | 0.0000 |
|  |  |  |  | BM25 only | 0.6434 | 0.8023 | 0.7597 | 0.23 | 0.0000 | 0.0000 |
|  |  |  |  | Hybrid (RRF) | 0.6550 | 0.8023 | 0.7784 | 4.41 | 0.0000 | 0.0000 |
|  |  |  |  | Hybrid + Reranker | 0.6434 | 0.7791 | 0.7541 | 6.27 | 0.0000 | 0.0000 |
| 256 | `numpy` | 423 | 9.46% | Dense only | 0.6085 | 0.7326 | 0.7174 | 23.06 | 0.1628 | 0.5395 |
|  |  |  |  | BM25 only | 0.6085 | 0.7558 | 0.7306 | 0.81 | 0.0465 | 0.3349 |
|  |  |  |  | Hybrid (RRF) | 0.6318 | 0.7558 | 0.7434 | 22.41 | 0.0930 | 0.4744 |
|  |  |  |  | Hybrid + Reranker | 0.6434 | 0.7791 | 0.7488 | 24.14 | 0.0930 | 0.3767 |
| 256 | `numpy` | 1704 | 2.35% | Dense only | 0.4690 | 0.6628 | 0.6202 | 25.46 | 0.3023 | 0.6047 |
|  |  |  |  | BM25 only | 0.5969 | 0.7558 | 0.7161 | 3.76 | 0.0465 | 0.3721 |
|  |  |  |  | Hybrid (RRF) | 0.5620 | 0.6860 | 0.6802 | 31.02 | 0.2093 | 0.5488 |
|  |  |  |  | Hybrid + Reranker | 0.6434 | 0.7674 | 0.7488 | 33.87 | 0.0930 | 0.4372 |
| 256 | `numpy` | 8548 | 0.47% | Dense only | 0.3992 | 0.5271 | 0.5047 | 38.17 | 0.3953 | 0.6233 |
|  |  |  |  | BM25 only | 0.5969 | 0.7558 | 0.7158 | 27.47 | 0.0465 | 0.3302 |
|  |  |  |  | Hybrid (RRF) | 0.3992 | 0.6512 | 0.5624 | 59.23 | 0.3488 | 0.5721 |
|  |  |  |  | Hybrid + Reranker | 0.6434 | 0.7674 | 0.7488 | 61.85 | 0.1163 | 0.4326 |
| 256 | `chroma` | 77 | 51.95% | Dense only | 0.6550 | 0.8023 | 0.7785 | 10.71 | 0.0000 | 0.0000 |
|  |  |  |  | BM25 only | 0.6434 | 0.8023 | 0.7597 | 0.93 | 0.0000 | 0.0000 |
|  |  |  |  | Hybrid (RRF) | 0.6550 | 0.8023 | 0.7784 | 11.04 | 0.0000 | 0.0000 |
|  |  |  |  | Hybrid + Reranker | 0.6434 | 0.7791 | 0.7541 | 13.98 | 0.0000 | 0.0000 |
| 256 | `chroma` | 423 | 9.46% | Dense only | 0.6085 | 0.7326 | 0.7174 | 46.64 | 0.1628 | 0.5395 |
|  |  |  |  | BM25 only | 0.6085 | 0.7558 | 0.7306 | 2.07 | 0.0465 | 0.3349 |
|  |  |  |  | Hybrid (RRF) | 0.6318 | 0.7558 | 0.7434 | 39.08 | 0.0930 | 0.4744 |
|  |  |  |  | Hybrid + Reranker | 0.6434 | 0.7791 | 0.7488 | 40.69 | 0.0930 | 0.3767 |
| 256 | `chroma` | 1704 | 2.35% | Dense only | 0.4225 | 0.5698 | 0.5563 | 69.94 | 0.3721 | 0.6326 |
|  |  |  |  | BM25 only | 0.5969 | 0.7558 | 0.7161 | 6.26 | 0.0465 | 0.3721 |
|  |  |  |  | Hybrid (RRF) | 0.4690 | 0.5930 | 0.5963 | 46.86 | 0.2791 | 0.5767 |
|  |  |  |  | Hybrid + Reranker | 0.6434 | 0.7674 | 0.7488 | 54.97 | 0.0930 | 0.4372 |
| 256 | `chroma` | 8548 | 0.47% | Dense only | 0.3992 | 0.5039 | 0.4988 | 79.41 | 0.3953 | 0.6279 |
|  |  |  |  | BM25 only | 0.5969 | 0.7558 | 0.7158 | 28.43 | 0.0465 | 0.3302 |
|  |  |  |  | Hybrid (RRF) | 0.3760 | 0.6279 | 0.5427 | 90.34 | 0.3488 | 0.5721 |
|  |  |  |  | Hybrid + Reranker | 0.6434 | 0.7674 | 0.7488 | 83.08 | 0.1163 | 0.4326 |
| 500 | `numpy` | 44 | 68.18% | Dense only | 0.6550 | 0.8837 | 0.8128 | 3.58 | 0.0000 | 0.0000 |
|  |  |  |  | BM25 only | 0.5969 | 0.9070 | 0.7694 | 0.21 | 0.0000 | 0.0000 |
|  |  |  |  | Hybrid (RRF) | 0.6318 | 0.8837 | 0.7953 | 4.12 | 0.0000 | 0.0000 |
|  |  |  |  | Hybrid + Reranker | 0.6667 | 0.8488 | 0.7884 | 7.50 | 0.0000 | 0.0000 |
| 500 | `numpy` | 235 | 12.77% | Dense only | 0.5620 | 0.7791 | 0.7138 | 17.15 | 0.2326 | 0.5860 |
|  |  |  |  | BM25 only | 0.5504 | 0.7791 | 0.7062 | 0.65 | 0.0698 | 0.3907 |
|  |  |  |  | Hybrid (RRF) | 0.5736 | 0.7907 | 0.7243 | 18.61 | 0.1395 | 0.5163 |
|  |  |  |  | Hybrid + Reranker | 0.6667 | 0.7907 | 0.7853 | 21.51 | 0.0698 | 0.4372 |
| 500 | `numpy` | 936 | 3.21% | Dense only | 0.4109 | 0.6085 | 0.5619 | 24.66 | 0.3721 | 0.6744 |
|  |  |  |  | BM25 only | 0.5504 | 0.7791 | 0.6991 | 2.83 | 0.0698 | 0.4372 |
|  |  |  |  | Hybrid (RRF) | 0.4574 | 0.7248 | 0.6293 | 28.56 | 0.2558 | 0.6140 |
|  |  |  |  | Hybrid + Reranker | 0.6434 | 0.7791 | 0.7736 | 31.08 | 0.0930 | 0.5256 |
| 500 | `numpy` | 4658 | 0.64% | Dense only | 0.3876 | 0.5465 | 0.5097 | 35.28 | 0.4651 | 0.6651 |
|  |  |  |  | BM25 only | 0.5736 | 0.7558 | 0.7078 | 14.24 | 0.0698 | 0.4279 |
|  |  |  |  | Hybrid (RRF) | 0.4574 | 0.7209 | 0.6087 | 45.36 | 0.3488 | 0.5907 |
|  |  |  |  | Hybrid + Reranker | 0.6434 | 0.7558 | 0.7568 | 50.07 | 0.0698 | 0.5070 |

## What approximate search costs

Recall@1 under exhaustive cosine (`numpy`) minus Recall@1 under HNSW
(`chroma`), at the same corpus size. A positive number is recall the
approximate index lost. BM25 never touches the vector store, so its
rows must read zero — they are the control that says nothing else
changed between the two runs.

| Chunks | Configuration | exact R@1 | HNSW R@1 | lost |
|---:|---|---:|---:|---:|
| 77 | BM25 only | 0.6434 | 0.6434 | +0.0000 |
| 77 | Dense only | 0.6550 | 0.6550 | +0.0000 |
| 77 | Hybrid (RRF) | 0.6550 | 0.6550 | +0.0000 |
| 77 | Hybrid + Reranker | 0.6434 | 0.6434 | +0.0000 |
| 423 | BM25 only | 0.6085 | 0.6085 | +0.0000 |
| 423 | Dense only | 0.6085 | 0.6085 | +0.0000 |
| 423 | Hybrid (RRF) | 0.6318 | 0.6318 | +0.0000 |
| 423 | Hybrid + Reranker | 0.6434 | 0.6434 | +0.0000 |
| 1704 | BM25 only | 0.5969 | 0.5969 | +0.0000 |
| 1704 | Dense only | 0.4690 | 0.4225 | +0.0465 |
| 1704 | Hybrid (RRF) | 0.5620 | 0.4690 | +0.0930 |
| 1704 | Hybrid + Reranker | 0.6434 | 0.6434 | +0.0000 |
| 8548 | BM25 only | 0.5969 | 0.5969 | +0.0000 |
| 8548 | Dense only | 0.3992 | 0.3992 | +0.0000 |
| 8548 | Hybrid (RRF) | 0.3992 | 0.3760 | +0.0232 |
| 8548 | Hybrid + Reranker | 0.6434 | 0.6434 | +0.0000 |

## Does the chunk size that wins on a small corpus still win?

Recall@1 for the production configuration (hybrid + reranker) under
exhaustive search, at each chunk size. Corpus sizes are matched by how
many distractor *documents* were added, because a different chunk size
turns the same documents into a different number of chunks — that
count is shown for each.

| Distractor docs | 256 tok — chunks / R@1 / MRR | 500 tok — chunks / R@1 / MRR | best |
|---:|---:|---:|---|
| 0 | 77 / 0.6434 / 0.7541 | 44 / 0.6667 / 0.7884 | 500 tok |
| 140 | 423 / 0.6434 / 0.7488 | 235 / 0.6667 / 0.7853 | 500 tok |
| 655 | 1704 / 0.6434 / 0.7488 | 936 / 0.6434 / 0.7736 | tie |
| 3400 | 8548 / 0.6434 / 0.7488 | 4658 / 0.6434 / 0.7568 | tie |

The shipped default is 256 tokens. A chunk size that wins only at the smallest corpus is a finding about the corpus, not about chunking — which is the whole reason this table is not just the chunk-size ablation run again.

## Index build cost

| Store | Chunks | Documents | Embedding dim | Index build (s) |
|---|---:|---:|---:|---:|
| `numpy` | 77 | 22 | 76 | 13.55 |
| `numpy` | 423 | 162 | 384 | 6.26 |
| `numpy` | 1704 | 677 | 384 | 16.49 |
| `numpy` | 8548 | 3422 | 384 | 64.51 |
| `chroma` | 77 | 22 | 76 | 1.19 |
| `chroma` | 423 | 162 | 384 | 6.61 |
| `chroma` | 1704 | 677 | 384 | 17.50 |
| `chroma` | 8548 | 3422 | 384 | 71.76 |
| `numpy` | 44 | 22 | 43 | 0.62 |
| `numpy` | 235 | 162 | 234 | 3.29 |
| `numpy` | 936 | 677 | 384 | 12.78 |
| `numpy` | 4658 | 3422 | 384 | 51.21 |

Distractors are synthetic. They are hard negatives by construction — same
topics, same register, same policy vocabulary — but a real corpus of this
size would contain both easier negatives (off-topic material) and harder
ones (near-duplicate revisions of the same policy).
