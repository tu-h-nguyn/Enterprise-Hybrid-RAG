# Corpus-scale experiment

Generated 2026-09-15T14:33:00+00:00 · 50 questions (43 answerable), held fixed throughout.

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
| 256 | `numpy` | 77 | 51.95% | Dense only | 0.5853 | 0.8953 | 0.7740 | 14.55 | 0.0000 | 0.0000 |
|  |  |  |  | BM25 only | 0.6434 | 0.8023 | 0.7597 | 0.25 | 0.0000 | 0.0000 |
|  |  |  |  | Hybrid (RRF) | 0.6434 | 0.8488 | 0.7910 | 11.21 | 0.0000 | 0.0000 |
|  |  |  |  | Hybrid + Reranker | 0.7946 | 0.9070 | 0.9085 | 685.39 | 0.0000 | 0.0000 |
| 256 | `numpy` | 423 | 9.46% | Dense only | 0.4690 | 0.7209 | 0.6551 | 10.87 | 0.2791 | 0.5302 |
|  |  |  |  | BM25 only | 0.6085 | 0.7558 | 0.7306 | 0.71 | 0.0465 | 0.3349 |
|  |  |  |  | Hybrid (RRF) | 0.5853 | 0.8023 | 0.7369 | 12.17 | 0.2326 | 0.4605 |
|  |  |  |  | Hybrid + Reranker | 0.7248 | 0.8760 | 0.8384 | 657.61 | 0.1628 | 0.4651 |
| 256 | `numpy` | 1704 | 2.35% | Dense only | 0.4341 | 0.6357 | 0.6061 | 20.01 | 0.3256 | 0.6698 |
|  |  |  |  | BM25 only | 0.5969 | 0.7558 | 0.7161 | 3.16 | 0.0465 | 0.3721 |
|  |  |  |  | Hybrid (RRF) | 0.5620 | 0.8178 | 0.7324 | 31.42 | 0.2558 | 0.5488 |
|  |  |  |  | Hybrid + Reranker | 0.7248 | 0.8411 | 0.8289 | 725.62 | 0.1628 | 0.5581 |
| 256 | `numpy` | 8548 | 0.47% | Dense only | 0.3798 | 0.5426 | 0.5223 | 21.22 | 0.4186 | 0.7535 |
|  |  |  |  | BM25 only | 0.5969 | 0.7558 | 0.7158 | 16.47 | 0.0465 | 0.3302 |
|  |  |  |  | Hybrid (RRF) | 0.4922 | 0.7829 | 0.6805 | 46.81 | 0.3488 | 0.5395 |
|  |  |  |  | Hybrid + Reranker | 0.7248 | 0.8062 | 0.8285 | 723.22 | 0.1628 | 0.5814 |
| 256 | `chroma` | 77 | 51.95% | Dense only | 0.5853 | 0.8953 | 0.7740 | 15.51 | 0.0000 | 0.0000 |
|  |  |  |  | BM25 only | 0.6434 | 0.8023 | 0.7597 | 0.78 | 0.0000 | 0.0000 |
|  |  |  |  | Hybrid (RRF) | 0.6434 | 0.8488 | 0.7910 | 16.45 | 0.0000 | 0.0000 |
|  |  |  |  | Hybrid + Reranker | 0.7946 | 0.9070 | 0.9085 | 692.01 | 0.0000 | 0.0000 |
| 256 | `chroma` | 423 | 9.46% | Dense only | 0.4690 | 0.7209 | 0.6551 | 16.05 | 0.2791 | 0.5302 |
|  |  |  |  | BM25 only | 0.6085 | 0.7558 | 0.7306 | 1.29 | 0.0465 | 0.3349 |
|  |  |  |  | Hybrid (RRF) | 0.5853 | 0.8023 | 0.7369 | 17.21 | 0.2326 | 0.4605 |
|  |  |  |  | Hybrid + Reranker | 0.7248 | 0.8760 | 0.8384 | 667.29 | 0.1628 | 0.4651 |
| 256 | `chroma` | 1704 | 2.35% | Dense only | 0.4341 | 0.6357 | 0.6061 | 16.24 | 0.3256 | 0.6698 |
|  |  |  |  | BM25 only | 0.5969 | 0.7558 | 0.7161 | 3.14 | 0.0465 | 0.3721 |
|  |  |  |  | Hybrid (RRF) | 0.5620 | 0.8178 | 0.7324 | 21.05 | 0.2558 | 0.5488 |
|  |  |  |  | Hybrid + Reranker | 0.7248 | 0.8411 | 0.8289 | 664.42 | 0.1628 | 0.5581 |
| 256 | `chroma` | 8548 | 0.47% | Dense only | 0.3798 | 0.5426 | 0.5200 | 18.35 | 0.4186 | 0.7535 |
|  |  |  |  | BM25 only | 0.5969 | 0.7558 | 0.7158 | 17.92 | 0.0465 | 0.3302 |
|  |  |  |  | Hybrid (RRF) | 0.4922 | 0.7829 | 0.6805 | 41.18 | 0.3488 | 0.5395 |
|  |  |  |  | Hybrid + Reranker | 0.7248 | 0.8062 | 0.8285 | 677.86 | 0.1628 | 0.5814 |
| 500 | `numpy` | 44 | 68.18% | Dense only | 0.5155 | 0.8295 | 0.6906 | 10.88 | 0.0000 | 0.0000 |
|  |  |  |  | BM25 only | 0.5969 | 0.9070 | 0.7694 | 0.21 | 0.0000 | 0.0000 |
|  |  |  |  | Hybrid (RRF) | 0.5620 | 0.8372 | 0.7362 | 11.19 | 0.0000 | 0.0000 |
|  |  |  |  | Hybrid + Reranker | 0.7016 | 0.9186 | 0.8568 | 1370.06 | 0.0000 | 0.0000 |
| 500 | `numpy` | 235 | 12.77% | Dense only | 0.4574 | 0.6783 | 0.5999 | 10.88 | 0.2326 | 0.5535 |
|  |  |  |  | BM25 only | 0.5504 | 0.7791 | 0.7062 | 0.50 | 0.0698 | 0.3907 |
|  |  |  |  | Hybrid (RRF) | 0.5039 | 0.7597 | 0.6716 | 11.82 | 0.2558 | 0.5209 |
|  |  |  |  | Hybrid + Reranker | 0.6550 | 0.8721 | 0.8101 | 1344.73 | 0.1163 | 0.4233 |
| 500 | `numpy` | 936 | 3.21% | Dense only | 0.4109 | 0.5969 | 0.5478 | 11.05 | 0.3488 | 0.6837 |
|  |  |  |  | BM25 only | 0.5504 | 0.7791 | 0.6991 | 1.66 | 0.0698 | 0.4372 |
|  |  |  |  | Hybrid (RRF) | 0.4806 | 0.7713 | 0.6565 | 14.00 | 0.3256 | 0.5674 |
|  |  |  |  | Hybrid + Reranker | 0.6318 | 0.8566 | 0.7950 | 1335.62 | 0.1860 | 0.5581 |
| 500 | `numpy` | 4658 | 0.64% | Dense only | 0.3643 | 0.5736 | 0.4957 | 24.40 | 0.4186 | 0.7814 |
|  |  |  |  | BM25 only | 0.5736 | 0.7558 | 0.7078 | 9.20 | 0.0698 | 0.4279 |
|  |  |  |  | Hybrid (RRF) | 0.4457 | 0.7597 | 0.6287 | 38.54 | 0.3953 | 0.6093 |
|  |  |  |  | Hybrid + Reranker | 0.6318 | 0.8566 | 0.7868 | 1407.33 | 0.1628 | 0.6093 |

## What approximate search costs

Recall@1 under exhaustive cosine (`numpy`) minus Recall@1 under HNSW
(`chroma`), at the same corpus size. A positive number is recall the
approximate index lost. BM25 never touches the vector store, so its
rows must read zero — they are the control that says nothing else
changed between the two runs.

| Chunks | Configuration | exact R@1 | HNSW R@1 | lost |
|---:|---|---:|---:|---:|
| 77 | BM25 only | 0.6434 | 0.6434 | +0.0000 |
| 77 | Dense only | 0.5853 | 0.5853 | +0.0000 |
| 77 | Hybrid (RRF) | 0.6434 | 0.6434 | +0.0000 |
| 77 | Hybrid + Reranker | 0.7946 | 0.7946 | +0.0000 |
| 423 | BM25 only | 0.6085 | 0.6085 | +0.0000 |
| 423 | Dense only | 0.4690 | 0.4690 | +0.0000 |
| 423 | Hybrid (RRF) | 0.5853 | 0.5853 | +0.0000 |
| 423 | Hybrid + Reranker | 0.7248 | 0.7248 | +0.0000 |
| 1704 | BM25 only | 0.5969 | 0.5969 | +0.0000 |
| 1704 | Dense only | 0.4341 | 0.4341 | +0.0000 |
| 1704 | Hybrid (RRF) | 0.5620 | 0.5620 | +0.0000 |
| 1704 | Hybrid + Reranker | 0.7248 | 0.7248 | +0.0000 |
| 8548 | BM25 only | 0.5969 | 0.5969 | +0.0000 |
| 8548 | Dense only | 0.3798 | 0.3798 | +0.0000 |
| 8548 | Hybrid (RRF) | 0.4922 | 0.4922 | +0.0000 |
| 8548 | Hybrid + Reranker | 0.7248 | 0.7248 | +0.0000 |

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

The shipped default is 256 tokens. A chunk size that wins only at the smallest corpus is a finding about the corpus, not about chunking — which is the whole reason this table is not just the chunk-size ablation run again.

## Index build cost

| Store | Chunks | Documents | Embedding dim | Index build (s) |
|---|---:|---:|---:|---:|
| `numpy` | 77 | 22 | 384 | 8.87 |
| `numpy` | 423 | 162 | 384 | 12.97 |
| `numpy` | 1704 | 677 | 384 | 45.99 |
| `numpy` | 8548 | 3422 | 384 | 224.18 |
| `chroma` | 77 | 22 | 384 | 4.58 |
| `chroma` | 423 | 162 | 384 | 13.42 |
| `chroma` | 1704 | 677 | 384 | 47.50 |
| `chroma` | 8548 | 3422 | 384 | 230.28 |
| `numpy` | 44 | 22 | 384 | 3.30 |
| `numpy` | 235 | 162 | 384 | 9.75 |
| `numpy` | 936 | 677 | 384 | 32.37 |
| `numpy` | 4658 | 3422 | 384 | 153.93 |

Distractors are synthetic. They are hard negatives by construction — same
topics, same register, same policy vocabulary — but a real corpus of this
size would contain both easier negatives (off-topic material) and harder
ones (near-duplicate revisions of the same policy).
