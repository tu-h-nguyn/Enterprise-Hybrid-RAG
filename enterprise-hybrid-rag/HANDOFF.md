# HANDOFF — Enterprise Hybrid RAG

State of the repository, and exactly what is left to do. Written for whoever
(or whatever) picks this up next.

---

## 1. Status by phase

| Phase | Item | Status |
|---|---|---|
| 1 | Repo inspection / plan | done |
| 2 | Ingestion: PDF / DOCX / Markdown loaders, cleaner, metadata-aware chunker | **done, runs** |
| 3 | Indexing: ChromaDB dense index + persistent BM25 | **done, runs** |
| 4 | Dense retrieval | **done** |
| 5 | BM25 retrieval | **done** |
| 6 | Hybrid retrieval + explicit RRF | **done** |
| 7 | Cross-encoder reranker (+ offline fallback, + no-op for ablation) | **done** |
| 8 | Context builder (dedup + token budget + `[Source N]` blocks) | **done** |
| 9 | LLM layer: OpenAI-compatible, Anthropic, offline extractive | **done** |
| 10 | No-answer gate (`app/services/confidence.py`, per-stage thresholds) | **done, wired** |
| 11 | Eval dataset **schema + resolver** | done |
| 11 | Eval dataset **content (50 questions)** | **done** |
| 12 | Retrieval metrics (R@1/3/5/10, MRR, nDCG, precision) + evaluator | done, **bug fixed** |
| 13 | Generation metrics (groundedness, citation support, token-F1, LLM judge) | done |
| 14 | `app/evaluation/experiments.py` + ablation runner | **done, runs** |
| 15 | FastAPI app (`/health`, `/documents*`, `/query`) | done, **smoke-tested** |
| 16 | Streamlit UI | **done, driven in a browser** |
| 17 | Tests (`tests/unit`, `tests/integration`) | **done — 101 pass** |
| 18 | Observability (JSON logs, `Stopwatch`, `QueryTrace`) | done, **asserted in tests** |
| 19 | Dockerfile / docker-compose | **done, built and served in CI** |
| 20 | README | **done, numbers measured** |
| 22 | Final verification run | **done** |

Benchmark numbers now exist in `data/evaluation/results.json` and `report.md`,
and are quoted in the README. They were produced by real runs on the offline
fallback backends (see §3) and every artefact records which backends produced
it. **They are not MiniLM numbers.** Re-run `scripts/benchmark.py` on a machine
with hub access before quoting them as neural-model results.

CI (`.github/workflows/ci.yml`) now runs two jobs per PR: tests + label audit +
an API smoke test on pinned offline backends, and a Docker job that builds both
image targets, ingests inside the container and queries the live API. The Docker
job leaves backends on `auto`, and a runner can reach the hub, so it confirmed
that `all-MiniLM-L6-v2` and the cross-encoder load and answer correctly
(`/health` reports both; the annual-leave question cites `employee_handbook.pdf`
p.1). The build gap the previous session could not close is closed.

### What a later session still needs to do

1. **Re-run the whole benchmark with the neural backends.** The path is known to
   work now, but every number in the README and `results.json` is still a
   fallback number. The paraphrased row (R@1 0.0000–0.1111) is the one expected
   to move most. Budget for the cost: reranking one query took 1664.59 ms with
   the cross-encoder on a CPU runner, against 2.9 ms for the lexical fallback.
2. **Re-tune `min_lexical_rerank_score` / `min_rerank_score` afterwards.** The
   current gate over-refuses 15 of 43 answerable questions, but most of that is
   a downstream symptom of lexical retrieval rather than a bad constant, so
   tuning before the encoder swap would fit the wrong problem.
3. **Consider a CPU-only torch install.** The image is 7.29 GB, most of it CUDA
   wheels nothing here uses.

---

## 2. What actually runs right now

```bash
python scripts/make_demo_corpus.py   # 22 documents into data/raw (20 md, 2 pdf, 2 docx… see note)
python scripts/ingest.py             # -> 22 documents, 44 chunks, Chroma + BM25 built
```

Verified working: loaders (real PDF/DOCX bytes), chunking with page provenance,
Chroma upsert/query, BM25 query + persistence, RRF (hand-checked), lexical
reranker, context builder dedup, citation parsing, no-answer gate, RagService
construction.

---

## 3. Environment findings (important)

This was built in a sandbox where **`huggingface.co` is blocked** and there is
**no LLM API key**. Two consequences, both handled by design rather than by
faking:

* `EMBEDDING_BACKEND=auto` fell back to `tfidf_svd` (TF-IDF word+char n-grams →
  TruncatedSVD → L2-normalised). Real and learned, but **lexical-semantic, not
  neural**.
* `RERANKER_BACKEND=auto` fell back to `LexicalReranker` — a transparent
  feature-based scorer, **explicitly not a cross-encoder**.
* `LLM_PROVIDER=auto` fell back to `ExtractiveLLM` — selects supporting
  sentences from context and cites them. **Not generative.**

On a normal machine with network + a key, the same `auto` settings will pick
`all-MiniLM-L6-v2`, `ms-marco-MiniLM-L-6-v2` and the real LLM with **no code
change**. Run the benchmark on that machine — the numbers will mean far more.

Every evaluation artefact must record which backend produced it
(`bundle.manifest["embedder"]`, `reranker.describe()`, `llm.describe()`).
Never report fallback numbers as if they were MiniLM numbers.

---

## 4. Contracts you must not break

```python
# retrieval — every retriever returns list[RetrievedChunk]
RetrievedChunk(chunk: Chunk, score: float, rank: int, retriever: str,
               component_scores: dict[str, float], component_ranks: dict[str, int])

# fusion
reciprocal_rank_fusion(ranked_lists: dict[str, list[str]], k: int,
                       weights: dict[str, float]) -> list[tuple[str, float]]

# reranking
reranker.rerank(query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]

# the one entry point used by API, scripts AND evaluation
RagService.query(question, top_k=None, method=None, rerank=None,
                 include_chunks=True) -> RagAnswer
RagService.retrieve(question, method, top_k, rerank) -> (candidates, QueryTrace, Stopwatch)
RagService.from_disk(settings) -> RagService

# evaluation
EvalDataset.load(path) / RelevanceResolver(chunks).resolve(q) -> set[chunk_id]
RelevanceResolver(chunks).audit(dataset) -> list[dict]     # RUN THIS FIRST
RetrievalEvaluator(service, resolver).run(dataset, RetrievalConfig) -> RetrievalRunResult
```

Ground truth is **answer spans + source documents**, resolved to chunk IDs at
evaluation time. That is deliberate: chunk IDs change when `chunk_size_tokens`
changes, so pinned IDs would silently break the chunk-size ablation. Keep it.

---

## 5. Remaining work, in order

### 5.1 `data/evaluation/demo_questions.json` — 40–50 questions
Schema is `EvalDataset` in `app/evaluation/dataset.py`.

```json
{"name":"northwind_demo","description":"...","questions":[
 {"id":"q001","question":"How many days of paid annual leave are employees entitled to?",
  "ground_truth":"22 days per calendar year.","question_type":"factual",
  "relevant_documents":["employee_handbook.pdf"],
  "answer_spans":["22 days of paid annual leave"]}]}
```

Rules:
* `answer_spans` must be **verbatim substrings of the corpus** (whitespace/case
  insensitive). Verify with `RelevanceResolver.audit()` — it must return `[]`.
* `relevant_documents` uses the **file name** (`chunk.source`), e.g.
  `employee_handbook.pdf`.
* Required mix: `factual`, `paraphrased` (no shared keywords — these are what
  dense retrieval wins), `multi_step`, `keyword` (exact codes like `HB-7.2`,
  `SEV1`, `FIN-TE-4` — these are what BM25 wins), `terminology`,
  `unanswerable` (≥6; these must have **no** `relevant_documents`/`answer_spans`).
* Spread across all 22 documents, not just the handbook.
* Source the facts from `data/raw/` — do not write questions from memory.

### 5.2 `app/evaluation/experiments.py`
Run configs A=dense, B=bm25, C=hybrid, D=hybrid+rerank (plus a chunk-size
ablation that re-ingests at e.g. 256 / 500 / 800 and rebuilds the index).
Write `data/evaluation/results.json` (machine-readable, including the backend
manifest) and `data/evaluation/report.md` (human-readable table).

### 5.3 `scripts/evaluate.py`, `scripts/benchmark.py`
Thin CLIs over `RetrievalEvaluator` / `GenerationEvaluator` / `experiments.py`.
`--dataset`, `--top-k`, `--out`. Print the comparison table to stdout.

### 5.4 `tests/` (pytest)
Unit: PDF loader on a generated PDF; chunker page/section preservation;
`reciprocal_rank_fusion` against a **hand-computed** example (with `k=1`,
dense `[A,B,C]` + sparse `[B,A,D]` → A = 1/2+1/3 = 0.8333, B = 1/3+1/2, C = D = 0.25);
BM25 retrieval; dense retrieval with a stub embedder + `NumpyVectorStore`;
reranker ordering; context budget + dedup; `parse_citation_indices` /
`strip_invalid_citations`; `ConfidenceGate` abstention; metric maths
(recall/MRR) on fixed inputs.
Integration: ingest → index → `RagService.query` end-to-end, and the API
`/query` via `fastapi.testclient.TestClient`.

### 5.5 `frontend/streamlit_app.py`
Upload → index → ask. Show answer, citations (doc + page + section), retrieved
chunks, retrieval method, and a debug expander with the `QueryTrace`. Talk to
the API over `API_URL`. Keep it plain.

### 5.6 `Dockerfile` + `docker-compose.yml`
`python:3.11-slim`, install `requirements.txt`, two services (api 8000, ui 8501),
volume for `data/`. Consider baking the HF models into the image at build time
so first query is not a download. Never bake keys.

### 5.7 `README.md`
Phase-20 outline: overview, problem, Mermaid architecture, pipeline, why hybrid,
dense vs BM25, why rerank, evaluation methodology, **measured** results, example
query, API example, install, Docker, config, testing, limitations, future work.

Limitations that must be stated honestly: token counts are a
`chars/4` heuristic, not a real tokenizer; the corpus is synthetic and small
(44 chunks at the 500-token default) so Recall@5 saturates and **Recall@1 / MRR
are the discriminative metrics**; the no-answer threshold is tuned on one
corpus; the LLM judge is approximate and is not ground truth.

### 5.8 Then actually run Phase 22
Install, run tests, ingest, index, evaluate ≥30 questions, all four configs,
serve the API, hit real queries, verify citations resolve to correct pages,
verify an unanswerable question abstains, fix what breaks, and only then write
the numbers into the README.

---

## 6. Known rough edges

* `scripts/demo_corpus_part*.py` should eventually be merged into
  `make_demo_corpus.py`; they are split only because they were written
  incrementally.
* `app/indexing/vector_store.py::NumpyVectorStore.upsert` does an `O(n)`
  `vstack` per chunk — fine for tests, wrong for bulk. Batch it if it ever
  becomes a real backend.
* `data/processed/` (Chroma + the fitted embedder pickle) is gitignored; it is
  rebuilt by `scripts/ingest.py`.
* The `tfidf_svd` embedding dimension is capped by corpus size
  (44 chunks → 43 dims). It will jump to 384 under `all-MiniLM-L6-v2`.
