# Enterprise Hybrid RAG

Document-grounded question answering over PDF, DOCX and Markdown, built without
a RAG framework: the retrieval core is plain Python, so every ranking decision
is inspectable and every number below is reproducible from this repository.

```
ingest → chunk → dense + BM25 index → RRF hybrid → rerank → context → grounded answer with citations
```

---

## The problem

An organisation's answers live in policy PDFs, contracts and runbooks. Two
naive approaches both fail:

* **Keyword search** returns documents, not answers, and misses anything phrased
  differently from the query.
* **An LLM alone** answers fluently and sometimes falsely, with no provenance and
  no way to tell the two apart.

What is needed is an answer *restricted to the corpus*, *carrying citations back
to a page*, and *willing to say it does not know*. That third property is the
one most systems skip, and it is the one this project treats as a first-class
feature: retrieval always returns a nearest neighbour, even when the corpus
contains nothing relevant.

## Architecture

```mermaid
flowchart TD
    subgraph Ingestion
        A[PDF / DOCX / Markdown] --> B[Loaders<br/>page + section provenance]
        B --> C[Cleaner<br/>boilerplate removal]
        C --> D[Metadata-aware chunker<br/>sentence-aligned overlap]
    end

    subgraph Indexing
        D --> E[Embedder]
        D --> F[BM25 Okapi]
        E --> G[(Chroma<br/>cosine)]
        F --> H[(BM25 index)]
    end

    subgraph Retrieval
        Q[Question] --> I[Dense retriever]
        Q --> J[Sparse retriever]
        G --> I
        H --> J
        I --> K[Reciprocal Rank Fusion]
        J --> K
        K --> L[Reranker]
    end

    subgraph Answering
        L --> M{Confidence gate}
        M -- below threshold --> N[Abstain:<br/>insufficient information]
        M -- above threshold --> O[Context builder<br/>dedup + token budget]
        O --> P[LLM<br/>grounded prompt]
        P --> R[Citation resolution<br/>invalid citations dropped]
        R --> S[Answer + citations + QueryTrace]
    end
```

## Pipeline

| Stage | What it does | Why it is built this way |
|---|---|---|
| **Loaders** | PDF via PyMuPDF, DOCX via python-docx, Markdown natively | Page numbers and section headings are captured **at load time** and never re-derived, which is what makes a citation point at a place rather than a file |
| **Cleaner** | Strips running headers/footers repeated across pages | Boilerplate otherwise dominates BM25 term statistics |
| **Chunker** | Groups blocks into sections, packs sentences to a token budget, carries a sentence-aligned overlap | A chunk never starts mid-sentence and never merges two sections — sections are the strongest free semantic boundary |
| **Indexing** | Dense and sparse indexes built together from one chunk list | Both must address the same `chunk_id` or fusion silently combines different documents |
| **Retrieval** | Dense, BM25, or both fused with RRF | See below |
| **Reranking** | Cross-encoder over ~20 candidates | Retrieval optimises recall over a large pool; reranking optimises precision over a small one |
| **Confidence gate** | Per-stage score threshold before generation | Prevents paying for an LLM call when nothing relevant was retrieved |
| **Context builder** | Near-duplicate removal, token budget, numbered `[Source N]` blocks | Chunk overlap means the top-5 often repeats a sentence, which wastes budget and biases the model |
| **Answer layer** | Resolves `[Source N]` back to chunks; drops unresolvable numbers | An unresolvable citation is a hallucinated provenance claim and must never render as real |

### Why hybrid, and why RRF specifically

Dense and sparse retrieval fail in *different* directions:

* **Dense** generalises across vocabulary — "time off" finds "annual leave" — but
  is weak on rare exact tokens such as `HB-7.2` or `SEV1`, because those get
  averaged away in a fixed-width vector.
* **BM25** is unbeatable on identifiers, acronyms and quoted terminology, needs no
  training, but cannot match a paraphrase.

Combining them by **score** does not work: cosine similarity lives in `[-1, 1]`,
BM25 is unbounded and corpus-dependent. Averaging or min-max normalising makes
the blend depend on the score *distribution* of one particular query, so a
single outlier BM25 hit can dominate. Reciprocal Rank Fusion discards magnitudes
and uses only ranks:

```
RRF(d) = Σᵢ  wᵢ / (k + rankᵢ(d))
```

`k = 60` (Cormack et al. 2009) damps the very top ranks, so a document ranked #1
by one retriever cannot single-handedly beat a document ranked #2–#3 by *both*.
Documents found by both accumulate two terms and rise — that agreement signal is
the entire point. The implementation is ~15 lines in `app/retrieval/hybrid.py`
and is unit-tested against a hand-computed example.

### Why rerank

A bi-encoder must compress a chunk into a vector *before it has seen the query*.
A cross-encoder reads `(query, chunk)` jointly with full attention and can judge
whether the chunk actually answers the question. The cost is quadratic attention
per pair, which is why it only ever sees the ~20 fused candidates rather than
the whole corpus.

---

## Evaluation methodology

**Ground truth is answer spans plus source documents, not pinned chunk IDs.**
Chunk IDs change when `chunk_size_tokens` changes, so a dataset of pinned IDs
would silently invalidate every label during the chunk-size ablation and read as
a retrieval collapse. Instead a chunk counts as relevant when it comes from a
listed document *and* contains one of the answer spans, resolved at evaluation
time.

`RelevanceResolver.audit()` is run before any metric is reported, and both CLIs
**refuse to print numbers** if it returns anything — an unresolvable label drives
recall to zero and is indistinguishable from a retrieval failure.

The dataset is **50 questions over all 22 documents**, written by reading the
corpus:

| Type | Count | What it isolates |
|---|---|---|
| `factual` | 14 | Direct lookup |
| `paraphrased` | 9 | Deliberately shares no vocabulary with the source passage — only a semantic match can find these |
| `multi_step` | 7 | Requires evidence from two or more documents |
| `keyword` | 7 | Turns on a rare reference code (`HB-7.2`, `FIN-TE-4`, `ISP-2024-03`) — BM25 territory |
| `terminology` | 6 | Definition lookup |
| `unanswerable` | 7 | Plausible but absent from the corpus; the only correct behaviour is refusal |

Retrieval metrics exclude unanswerable questions (they have no relevant chunks);
those are scored separately by refusal behaviour.

### The backends that produced these numbers

Two configurations were benchmarked, and **both are kept in the repository** because the difference between them is the most interesting result here.

| Component | Neural run *(primary)* | Offline fallback run |
|---|---|---|
| Embedding | `all-MiniLM-L6-v2`, 384-d | `tfidf_svd` (word 1–2 + char 3–5 → SVD), 43-d |
| Reranker | `ms-marco-MiniLM-L-6-v2` cross-encoder | `lexical` feature scorer |
| LLM | `extractive` | `extractive` |
| Artefact | `results.json` | `results_fallback.json` |

The neural run is produced by `.github/workflows/benchmark.yml` on a GitHub runner, which can reach the model hub — the development sandbox could not, which is why the fallback run exists at all. **No Hugging Face token is involved**: both models are public.

The workflow **pins** the neural backends instead of using `auto`, and asserts a 384-dimension `sentence_transformers` embedder in the index manifest *before* measuring. Under `auto`, an unreachable hub would fall back silently and publish fallback numbers under a workflow named "neural" — precisely the failure this project is built to avoid.

The LLM stays `extractive` in both runs, because CI has no API key. **Generation metrics therefore describe sentence selection, not a generative model**, in both columns.

Measured on Python 3.11, Linux x86-64, 22 documents / 44 chunks at `chunk_size_tokens=500`, `rrf_k=60`, `dense_top_k = sparse_top_k = 20`.

---

## Results

Reproduce with `python scripts/benchmark.py`, or read `data/evaluation/report.md`.

### Retrieval comparison — neural backends

| Configuration | R@1 | R@3 | R@5 | R@10 | MRR | nDCG@5 | P@5 | mean ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|
| Dense only | 0.5155 | 0.7946 | 0.8295 | 0.8527 | 0.6906 | 0.7199 | 0.2047 | 16.93 | 21.28 |
| BM25 only | 0.5969 | 0.8256 | 0.9070 | 0.9302 | 0.7694 | 0.8001 | 0.2233 | **0.72** | **0.83** |
| Hybrid (RRF) | 0.5620 | 0.7946 | 0.8372 | 0.8837 | 0.7362 | 0.7513 | 0.2093 | 17.14 | 20.69 |
| **Hybrid + Reranker** | **0.7016** | **0.8953** | **0.9186** | **0.9767** | **0.8568** | **0.8610** | **0.2233** | 1416.77 | 1460.45 |

Hybrid + Reranker leads every quality column, and costs **three orders of magnitude more per query than BM25** (1416.77 ms against 0.72 ms). Reranking, not retrieval, is where the request time goes.

Those timings are an order of magnitude, not a figure. The identical configuration measured 783.67 ms on a different CI runner — see `git show dd144a8:enterprise-hybrid-rag/data/evaluation/results.json` — so shared runners vary about twofold, whereas every quality column reproduced exactly across the two runs.

Recall@5 is high everywhere because 5 of 44 chunks is over 11% of the corpus, so **R@1 and MRR remain the discriminative metrics** and differences of a few points across 43 scored questions are within noise.

### What changed when real models replaced the fallbacks

| Configuration | R@1 fallback | R@1 neural | Δ | MRR fallback | MRR neural | Δ |
|---|---|---|---|---|---|---|
| Dense only | 0.6550 | 0.5155 | **−0.1395** | 0.8128 | 0.6906 | **−0.1222** |
| BM25 only | 0.5969 | 0.5969 | ±0.0000 | 0.7694 | 0.7694 | ±0.0000 |
| Hybrid (RRF) | 0.6318 | 0.5620 | **−0.0698** | 0.7953 | 0.7362 | **−0.0591** |
| **Hybrid + Reranker** | 0.6667 | **0.7016** | **+0.0349** | 0.7884 | **0.8568** | **+0.0684** |

Three of four configurations got **worse** with a better embedder. BM25 is unchanged, as it must be — it does not use the embedder, and the identical numbers are a useful sanity check that nothing else drifted between runs.

The explanation is entirely in the per-type breakdown below: the TF-IDF fallback's character n-grams matched reference codes **literally**, scoring a perfect 1.0000 on keyword questions. A sentence encoder compresses `HB-7.2` into a dense vector and loses it. What MiniLM buys instead is the only real paraphrase handling in the system.

### Recall@1 by question type — where the retrievers actually differ

| Configuration | factual | keyword | multi_step | paraphrased | terminology |
|---|---|---|---|---|---|
| Dense only | 0.8571 | 0.2857 | 0.3095 | **0.3333** | 0.5000 |
| BM25 only | 0.8571 | 0.8571 | 0.3810 | 0.0000 | 0.8333 |
| Hybrid (RRF) | 0.8571 | 0.2857 | **0.4524** | 0.2222 | 0.8333 |
| **Hybrid + Reranker** | **0.9286** | **1.0000** | **0.4524** | 0.2222 | 0.8333 |

### MRR by question type

| Configuration | factual | keyword | multi_step | paraphrased | terminology |
|---|---|---|---|---|---|
| Dense only | 0.9286 | 0.3492 | 0.8571 | **0.4907** | 0.6389 |
| BM25 only | 0.9107 | 0.9286 | 0.9286 | 0.2037 | 0.9167 |
| Hybrid (RRF) | 0.9286 | 0.4762 | **1.0000** | 0.3472 | 0.8667 |
| **Hybrid + Reranker** | **0.9643** | **1.0000** | **1.0000** | 0.4270 | **0.9167** |

Four findings, including the ones that are inconvenient:

1. **The two retrievers fail in opposite directions, and this is now measured rather than asserted.** Dense reaches 0.3333 on paraphrased questions where BM25 scores exactly 0.0000; BM25 reaches 0.8571 on keyword questions where dense manages 0.2857. Neither is better. This is the whole premise of hybrid retrieval, and it only became visible with a real encoder — under the fallback, the "dense" arm was itself lexical and scored 1.0000 on keyword questions by accident.

2. **The reranker is what makes the combination pay.** Plain RRF inherits dense's keyword weakness (0.2857) because fusion cannot recover a document neither arm ranked well. The cross-encoder, reading query and chunk jointly, restores keyword to a perfect 1.0000 while keeping the semantic gain — and unlike in the fallback run it improves R@5 and R@10 as well, so it is no longer trading recall for precision.

3. **Paraphrase remains the weakest retrieval story**, at 0.2222–0.3333. Better than the fallback's 0.0000–0.1111, but far from solved. Note that plain dense (0.3333) beats the full pipeline (0.2222) here: the cross-encoder is itself trained on lexical-ish relevance and sometimes demotes a semantically right chunk that shares no words with the query.

4. **`multi_step` is the weakest answerable category overall** (0.4524). These need evidence from two or more documents, and nothing in this pipeline decomposes a question or retrieves iteratively. That is the honest next problem.

### Chunk-size ablation

The corpus is re-ingested and both indexes rebuilt at each size, with ground truth re-resolved against the new chunks. Overlap is held at 20% so the comparison isolates chunk size. With a fixed 384-dimension encoder this comparison is finally clean — in the fallback run the embedding dimension was capped by corpus size and confounded it.

| Chunk size | Chunks | Mean tokens | Configuration | R@1 | R@5 | MRR | nDCG@5 |
|---|---|---|---|---|---|---|---|
| **256** | 77 | 192.6 | Dense only | 0.5853 | 0.8953 | 0.7740 | 0.7869 |
| | | | BM25 only | 0.6434 | 0.8023 | 0.7597 | 0.7660 |
| | | | Hybrid (RRF) | 0.6434 | 0.8488 | 0.7910 | 0.7862 |
| | | | **Hybrid + Reranker** | **0.7946** | 0.9070 | **0.9085** | **0.8937** |
| **500** *(default)* | 44 | 337.4 | Dense only | 0.5155 | 0.8295 | 0.6906 | 0.7199 |
| | | | BM25 only | 0.5969 | 0.9070 | 0.7694 | 0.8001 |
| | | | Hybrid (RRF) | 0.5620 | 0.8372 | 0.7362 | 0.7513 |
| | | | **Hybrid + Reranker** | 0.7016 | 0.9186 | 0.8568 | 0.8610 |
| **800** | 25 | 594.2 | Dense only | 0.5775 | 0.7791 | 0.7225 | 0.7196 |
| | | | BM25 only | 0.6318 | 0.8837 | 0.7966 | 0.8090 |
| | | | Hybrid (RRF) | 0.6318 | 0.8527 | 0.7870 | 0.7850 |
| | | | **Hybrid + Reranker** | 0.7481 | **0.9535** | 0.8747 | 0.8857 |

**The shipped default of 500 tokens is the worst of the three at R@1 and MRR for the production configuration** — 0.7016 against 0.7946 at 256, a gap of 9.3 points. Smaller chunks give the cross-encoder a tighter passage to judge and dilute each chunk's content less. 800 wins R@5 (0.9535), but a longer chunk makes a "hit" cover more text, so that metric flatters larger sizes by construction.

The default has not been changed here, because one 50-question corpus is not enough evidence to re-tune a default on, and because doing so would invalidate the comparison this table exists to make. It is recorded as the first thing to revisit.

### Generation

Produced by the **extractive** backend — sentence selection, not generation. These numbers describe that behaviour and are not LLM quality figures.

| Metric | Value |
|---|---|
| groundedness (answer bigrams present in context) | 0.6437 |
| answer_f1 (SQuAD-style token F1) | 0.3138 |
| span_coverage (labelled span appears in the answer) | 0.5078 |
| citation_precision | 0.6512 |
| uncited_sentence_rate | 0.3488 |
| mean citations per answer | 0.8372 |
| **over_refusal_rate** (answerable questions refused) | **0.3488** |
| **correct_refusal_rate** (unanswerable questions refused) | **0.8571** |
| mean latency | 1408.75 ms |

The gate correctly refuses **6 of 7** unanswerable questions and also refuses **15 of 43** answerable ones. Both rates are unchanged from the fallback run, which is itself informative: the gate's behaviour here is dominated by the extractive backend's exact-token matching rather than by retrieval quality, so improving the encoder did not move it. Measuring this properly needs a real LLM — point `LLM_BASE_URL` at a local Ollama server and re-run `scripts/evaluate.py --generation --judge`.

## Example

```bash
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"How many days of paid annual leave are employees entitled to?","top_k":5}'
```

```json
{
  "answer": "Employees are entitled to 22 days of paid annual leave per calendar year, in addition to public holidays observed in their country of employment [Source 2].",
  "citations": [
    {
      "source_index": 2,
      "chunk_id": "employee_handbook_001_000",
      "source": "employee_handbook.pdf",
      "page": 1,
      "section": "Introduction",
      "score": 3.308535
    }
  ],
  "metadata": {
    "retrieval_method": "hybrid", "reranker": "cross_encoder", "reranked": true,
    "n_candidates": 30, "n_final_contexts": 5, "context_tokens": 2390,
    "no_answer": false, "latency_ms": 1689.21,
    "stage_latency_ms": {"retrieval": 20.37, "rerank": 1664.59, "context": 1.49, "generation": 2.47}
  }
}
```

Captured from the container in CI, where the hub is reachable and `auto` therefore selects the neural backends. Note `rerank` at 1664.59 ms of a 1689.21 ms request: the cross-encoder *is* the latency budget.

A question the corpus cannot answer is refused rather than guessed:

```json
{
  "answer": "I could not find enough information in the indexed documents to answer this question confidently.",
  "citations": [],
  "metadata": {"no_answer": true, "no_answer_reason": "low_rerank_score", "top_score": 0.936242}
}
```

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness, index readiness, and **which backends were actually selected** |
| `POST` | `/query` | Ask a question; returns answer, citations, retrieved chunks and the trace |
| `POST` | `/documents/upload` | Upload a PDF, DOCX or Markdown file |
| `POST` | `/documents/index` | Rebuild both indexes |
| `GET` | `/documents` | List indexed documents and files awaiting indexing |
| `DELETE` | `/documents/{source}` | Remove a source file |

Interactive docs at `/docs`.

## Install and run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/ingest.py                    # 22 documents -> 44 chunks, both indexes
uvicorn app.main:app --reload               # API on :8000
API_URL=http://localhost:8000 streamlit run frontend/streamlit_app.py   # UI on :8501
```

Evaluation:

```bash
python scripts/evaluate.py --audit-only     # verify dataset labels resolve
python scripts/evaluate.py                  # four-configuration comparison
python scripts/benchmark.py                 # + chunk ablation -> results.json, report.md
```

`scripts/benchmark.py` builds the ablation indexes in a temporary directory, so
the index the API is serving is never replaced by an ablation build.

### Docker

```bash
docker compose up --build                   # api on :8000, ui on :8501
docker compose run --rm api python scripts/ingest.py
```

One image, two targets, built in CI through Buildx with the Actions layer cache
— plain `docker build` re-downloaded roughly 3 GB of wheels on every run, which
made that job take between 2 and 37 minutes depending on the runner. The build
pre-caches the neural weights so the first query is not a download; where the build host cannot reach the Hugging Face hub
the build still succeeds and the runtime falls back, reporting it in `/health`.
Containers run as a non-root user. **No secret is ever baked into the image** —
keys come from the environment or a gitignored `.env`.

Both targets are built in CI, which then ingests inside the container and
queries the running API, so the image is verified rather than merely written.
Note that `data/` is a bind mount by design — the corpus is not baked in — and
the container runs as uid 10001, so the mounted directory has to be writable by
that uid.

## Configuration

Everything is one Pydantic `Settings` object (`app/config/settings.py`); nothing
else reads `os.environ`. Any field can be set by environment variable.

| Variable | Default | Notes |
|---|---|---|
| `EMBEDDING_BACKEND` | `auto` | `auto` \| `sentence_transformers` \| `tfidf_svd` |
| `RERANKER_BACKEND` | `auto` | `auto` \| `cross_encoder` \| `lexical` \| `none` |
| `LLM_PROVIDER` | `auto` | `auto` \| `openai_compatible` \| `anthropic` \| `extractive` |
| `LLM_API_KEY` / `ANTHROPIC_API_KEY` | — | Never hard-coded |
| `CHUNK_SIZE_TOKENS` / `CHUNK_OVERLAP_TOKENS` | `500` / `100` | |
| `RETRIEVAL_METHOD` | `hybrid` | `dense` \| `sparse` \| `hybrid` |
| `RRF_K` | `60` | Fusion damping constant |
| `RERANK_TOP_K` | `5` | Contexts passed to the LLM |
| `NO_ANSWER_ENABLED` | `true` | Abstention gate |
| `LOG_CHUNK_TEXT` | `false` | Chunk text is never logged unless explicitly enabled |

`auto` prefers the neural models and degrades loudly — the substitution appears
in the logs, in `/health`, and in every evaluation artefact.

## Testing

```bash
pip install -r requirements-dev.txt

ruff check .             # lint and import order, configured in pyproject.toml
mypy                     # app/ and scripts/, 66 files, clean
pytest --cov=app         # 101 tests, 72% line coverage
```

`mypy` is configured pragmatically rather than strictly: `check_untyped_defs`,
`no_implicit_optional` and `strict_equality` are on, third-party libraries that
ship no stubs are excused by name, and the pydantic plugin is enabled — without
it every `Field(...)` default reads as a required argument and constructing
`Settings()` reports eighteen phantom missing arguments.

Coverage is **72%**, with CI failing below 70%. The floor sits just under the
current figure so it catches a regression without becoming a number to game.
What is uncovered is mostly deliberate: the Chroma vector store (tests use the
in-memory `NumpyVectorStore`, which is the point of having the abstraction) and
the DOCX loader.

Unit tests cover the PDF loader against a PDF generated in the test, chunker
page/section provenance, RRF against a hand-computed example, BM25 and dense
retrieval (the latter with a stub embedder and exact vectors), reranker
ordering, context budget and dedup, citation parsing and stripping, the
confidence gate per score scale, and the metric arithmetic. Integration tests
cover ingest → index → query end-to-end and the API via `TestClient`. Fixtures
build a hermetic index in `tmp_path`, so the suite needs no network, no API key
and no pre-existing index.

A third workflow, `.github/workflows/benchmark.yml`, re-measures on the neural
backends and then runs `scripts/compare_results.py` against the committed
numbers. It compares **quality metrics only** — those are deterministic given
the same corpus, dataset and backends, while latency is wall-clock and moves
every run — so results are re-committed only when a quality metric actually
changed, rather than on every run because timings drifted. Two independent runs
on different runners agreed on all **698** compared values, which is the
strongest reproducibility claim in this repository.

CI (`.github/workflows/ci.yml`) runs two jobs on every pull request:

* **Tests and dataset audit** — `ruff`, `mypy`, the suite under a 70% coverage
  floor, then ingest, then a label audit, then an API smoke test. The audit is a blocking gate on purpose: a chunker or
  loader change can stop answer spans resolving, which would silently drive
  recall to zero and look exactly like a retrieval regression. This job pins the
  offline backends so it is deterministic — model output is not a stable thing
  to gate a merge on.
* **Docker image builds and serves** — builds both image targets, ingests inside
  the container, then starts it and queries the live API. Backends are left on
  `auto` here, so this is the one place the neural path is exercised; the
  selected backend is printed as evidence rather than asserted on, so a hub
  hiccup cannot turn the build red for the wrong reason.

## Limitations

Stated plainly, because each one bounds how far the numbers above generalise.

1. **Retrieval is measured on real models; generation is not.** The tables above
   use `all-MiniLM-L6-v2` and the `ms-marco-MiniLM-L-6-v2` cross-encoder. The
   answer layer is still the extractive backend, because CI has no API key, so
   every generation metric describes sentence selection rather than a language
   model. Point `LLM_BASE_URL` at a local Ollama server to measure that half.
2. **Token counts are a `chars/4` heuristic, not a real tokenizer.** Every chunk
   size and context budget in this project is therefore approximate. A real
   tokenizer would shift chunk boundaries and change the ablation.
3. **The corpus is synthetic and small** — 22 documents, 44 chunks at the 500-token
   default. Recall@5 saturates, so R@1 and MRR are the only discriminative
   metrics, and differences of a few points across 43 scored questions are
   within noise. Nothing here has been shown to hold at enterprise scale, and
   the per-type rows rest on 6–14 questions each, which is few enough that a
   single question moves a row by 7–17 points.
4. **Reranking dominates latency** — 1416.77 ms per query against 0.72 ms for
   BM25 alone. Any deployment has to decide whether that precision is worth three
   orders of magnitude of latency, or whether to rerank only when the fusion
   margin is narrow. CI-runner timings vary about twofold between runs, so only
   the order of magnitude should be relied on.
5. **The no-answer threshold is tuned on one corpus** and is currently
   over-conservative: 34.9% of answerable questions are refused, including
   **every** paraphrased question. It has not been validated anywhere else and
   should be re-tuned per deployment — and re-tuned *after* the neural encoder
   is in place, since most of the over-refusal is a downstream symptom of
   lexical retrieval rather than a badly chosen constant.
6. **The LLM judge is approximate and is not ground truth.** It is
   non-deterministic, sensitive to the judge model, and biased toward the
   generator's style. It is reported in a separate block and never merged into
   the deterministic metrics. It did not run here — the extractive backend
   cannot judge, and refuses rather than emitting meaningless scores.
   The shipped `chunk_size_tokens=500` is also **not** the best value measured
   (256 beats it by 9.3 points of R@1); it is left as-is because one
   50-question corpus is thin evidence for re-tuning a default.
7. **Deterministic generation metrics measure lexical support, not truth.** A
   fully grounded answer can score below 1.0 simply by paraphrasing.
8. **Indexing is a full rebuild**, not an incremental upsert. Correct and fast at
   this size; wrong for a large corpus.
9. **The image is large: 7.29 GB.** `requirements.txt` pulls `torch` with its
   CUDA wheels even though nothing here uses a GPU. A CPU-only torch index
   would cut the bulk of that, at the cost of the image no longer matching the
   documented install. Not changed here because it trades one honest property
   for another, but it is the first thing to look at before shipping this
   anywhere real.

## Future work

* Measure generation against a real LLM (a local Ollama server needs no key and
  keeps the corpus on the machine), and turn the LLM judge on.
* Re-tune the abstention threshold against the over-refusal rate rather than
  inheriting a hand-picked constant.
* Incremental indexing to replace the full rebuild.
* A real tokenizer to replace the `chars/4` estimate.
* Query rewriting and multi-hop retrieval for the `multi_step` questions, which
  are the weakest answerable category (R@1 0.38–0.45).
* A larger, non-synthetic corpus, with human relevance judgements rather than
  answer-span resolution.
