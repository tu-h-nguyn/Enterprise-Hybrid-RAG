# Corpus-scale experiment

Generated 2026-09-15T08:38:17+00:00 · 50 questions (43 answerable), held fixed throughout.

* Embedder: `tfidf_svd` / `tfidf(word1-2 + char3-5)+svd` (dim 43)
* Reranker: `lexical` (neural: False)
* Vector stores: `numpy`, `chroma`
* Distractor corpus: seed 20250915, 3400 documents available

> ⚠️ Embeddings came from the **offline `tfidf_svd` fallback** (`tfidf(word1-2 + char3-5)+svd`, dim 43), not a neural sentence encoder. Dense numbers here are lexical-semantic and are a floor, not a measurement of `all-MiniLM-L6-v2`.

> ⚠️ The reranker was the **non-neural `lexical` fallback**, a feature-based scorer. It is not a cross-encoder and its numbers must not be reported as `ms-marco-MiniLM-L-6-v2` numbers.

> ⚠️ Generation used the **offline extractive backend**, which selects supporting sentences from the context rather than generating text. Generation metrics below describe that behaviour, not an LLM's.

> ⚠️ The embedding dimension was not constant across sizes ([43, 234, 384]). The offline `tfidf_svd` fallback derives its width from the corpus, so these rows vary the encoder as well as the corpus and the two effects cannot be separated. A run on `sentence_transformers` holds the dimension at 384 and is the one to read.

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
| `numpy` | 44 | 68.18% | Dense only | 0.6550 | 0.8837 | 0.8128 | 2.89 | 0.0000 | 0.0000 |
|  |  |  | BM25 only | 0.5969 | 0.9070 | 0.7694 | 0.20 | 0.0000 | 0.0000 |
|  |  |  | Hybrid (RRF) | 0.6318 | 0.8837 | 0.7953 | 3.23 | 0.0000 | 0.0000 |
|  |  |  | Hybrid + Reranker | 0.6667 | 0.8488 | 0.7884 | 6.31 | 0.0000 | 0.0000 |
| `numpy` | 235 | 12.77% | Dense only | 0.5620 | 0.7791 | 0.7138 | 13.28 | 0.2326 | 0.5860 |
|  |  |  | BM25 only | 0.5504 | 0.7791 | 0.7062 | 0.58 | 0.0698 | 0.3907 |
|  |  |  | Hybrid (RRF) | 0.5736 | 0.7907 | 0.7243 | 14.42 | 0.1395 | 0.5163 |
|  |  |  | Hybrid + Reranker | 0.6667 | 0.7907 | 0.7853 | 18.10 | 0.0698 | 0.4372 |
| `numpy` | 936 | 3.21% | Dense only | 0.4109 | 0.6085 | 0.5619 | 23.65 | 0.3721 | 0.6744 |
|  |  |  | BM25 only | 0.5504 | 0.7791 | 0.6991 | 2.13 | 0.0698 | 0.4372 |
|  |  |  | Hybrid (RRF) | 0.4574 | 0.7248 | 0.6293 | 27.52 | 0.2558 | 0.6140 |
|  |  |  | Hybrid + Reranker | 0.6434 | 0.7791 | 0.7736 | 30.04 | 0.0930 | 0.5256 |
| `numpy` | 4658 | 0.64% | Dense only | 0.3876 | 0.5465 | 0.5097 | 33.59 | 0.4651 | 0.6651 |
|  |  |  | BM25 only | 0.5736 | 0.7558 | 0.7078 | 13.99 | 0.0698 | 0.4279 |
|  |  |  | Hybrid (RRF) | 0.4574 | 0.7209 | 0.6087 | 45.26 | 0.3488 | 0.5907 |
|  |  |  | Hybrid + Reranker | 0.6434 | 0.7558 | 0.7568 | 47.79 | 0.0698 | 0.5070 |
| `chroma` | 44 | 68.18% | Dense only | 0.6550 | 0.8837 | 0.8128 | 9.16 | 0.0000 | 0.0000 |
|  |  |  | BM25 only | 0.5969 | 0.9070 | 0.7694 | 0.88 | 0.0000 | 0.0000 |
|  |  |  | Hybrid (RRF) | 0.6318 | 0.8837 | 0.7953 | 9.49 | 0.0000 | 0.0000 |
|  |  |  | Hybrid + Reranker | 0.6667 | 0.8488 | 0.7884 | 12.54 | 0.0000 | 0.0000 |
| `chroma` | 235 | 12.77% | Dense only | 0.5620 | 0.7791 | 0.7138 | 32.17 | 0.2326 | 0.5860 |
|  |  |  | BM25 only | 0.5504 | 0.7791 | 0.7062 | 1.28 | 0.0698 | 0.3907 |
|  |  |  | Hybrid (RRF) | 0.5736 | 0.7907 | 0.7243 | 22.78 | 0.1395 | 0.5163 |
|  |  |  | Hybrid + Reranker | 0.6667 | 0.7907 | 0.7853 | 24.72 | 0.0698 | 0.4372 |
| `chroma` | 936 | 3.21% | Dense only | 0.3876 | 0.5620 | 0.5244 | 48.74 | 0.4186 | 0.7023 |
|  |  |  | BM25 only | 0.5504 | 0.7791 | 0.6991 | 3.08 | 0.0698 | 0.4372 |
|  |  |  | Hybrid (RRF) | 0.4341 | 0.6550 | 0.5885 | 47.69 | 0.3256 | 0.6279 |
|  |  |  | Hybrid + Reranker | 0.6434 | 0.7791 | 0.7736 | 43.56 | 0.0930 | 0.5256 |
| `chroma` | 4658 | 0.64% | Dense only | 0.3643 | 0.5233 | 0.4864 | 65.91 | 0.4884 | 0.6698 |
|  |  |  | BM25 only | 0.5736 | 0.7558 | 0.7078 | 15.04 | 0.0698 | 0.4279 |
|  |  |  | Hybrid (RRF) | 0.4341 | 0.7209 | 0.5901 | 60.40 | 0.3721 | 0.5907 |
|  |  |  | Hybrid + Reranker | 0.6434 | 0.7558 | 0.7568 | 57.31 | 0.0698 | 0.5070 |

## What approximate search costs

Recall@1 under exhaustive cosine (`numpy`) minus Recall@1 under HNSW
(`chroma`), at the same corpus size. A positive number is recall the
approximate index lost. BM25 never touches the vector store, so its
rows must read zero — they are the control that says nothing else
changed between the two runs.

| Chunks | Configuration | exact R@1 | HNSW R@1 | lost |
|---:|---|---:|---:|---:|
| 44 | BM25 only | 0.5969 | 0.5969 | +0.0000 |
| 44 | Dense only | 0.6550 | 0.6550 | +0.0000 |
| 44 | Hybrid (RRF) | 0.6318 | 0.6318 | +0.0000 |
| 44 | Hybrid + Reranker | 0.6667 | 0.6667 | +0.0000 |
| 235 | BM25 only | 0.5504 | 0.5504 | +0.0000 |
| 235 | Dense only | 0.5620 | 0.5620 | +0.0000 |
| 235 | Hybrid (RRF) | 0.5736 | 0.5736 | +0.0000 |
| 235 | Hybrid + Reranker | 0.6667 | 0.6667 | +0.0000 |
| 936 | BM25 only | 0.5504 | 0.5504 | +0.0000 |
| 936 | Dense only | 0.4109 | 0.3876 | +0.0233 |
| 936 | Hybrid (RRF) | 0.4574 | 0.4341 | +0.0233 |
| 936 | Hybrid + Reranker | 0.6434 | 0.6434 | +0.0000 |
| 4658 | BM25 only | 0.5736 | 0.5736 | +0.0000 |
| 4658 | Dense only | 0.3876 | 0.3643 | +0.0233 |
| 4658 | Hybrid (RRF) | 0.4574 | 0.4341 | +0.0233 |
| 4658 | Hybrid + Reranker | 0.6434 | 0.6434 | +0.0000 |

## Index build cost

| Store | Chunks | Documents | Embedding dim | Index build (s) |
|---|---:|---:|---:|---:|
| `numpy` | 44 | 22 | 43 | 6.32 |
| `numpy` | 235 | 162 | 234 | 3.33 |
| `numpy` | 936 | 677 | 384 | 12.84 |
| `numpy` | 4658 | 3422 | 384 | 49.93 |
| `chroma` | 44 | 22 | 43 | 0.99 |
| `chroma` | 235 | 162 | 234 | 3.40 |
| `chroma` | 936 | 677 | 384 | 13.15 |
| `chroma` | 4658 | 3422 | 384 | 54.75 |

Distractors are synthetic. They are hard negatives by construction — same
topics, same register, same policy vocabulary — but a real corpus of this
size would contain both easier negatives (off-topic material) and harder
ones (near-duplicate revisions of the same policy).
