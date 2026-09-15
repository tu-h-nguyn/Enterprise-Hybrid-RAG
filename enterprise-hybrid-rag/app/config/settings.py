"""Central application configuration.

Every tunable behaviour of the RAG system is declared here in a single Pydantic
Settings object. Nothing else in the pipeline reads ``os.environ`` directly,
which keeps configuration auditable and makes experiments (which override
settings programmatically) reproducible.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    # ------------------------------------------------------------------ paths
    app_name: str = "Enterprise Hybrid RAG"
    data_dir: Path = PROJECT_ROOT / "data"
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    evaluation_dir: Path = PROJECT_ROOT / "data" / "evaluation"

    # -------------------------------------------------------------- chunking
    # 256 rather than 500, decided by measurement rather than by convention:
    # `scripts/scale_experiment.py --extra-chunk-sizes 500` runs both sizes over
    # corpora from 44 to ~8500 chunks, and 256 wins Recall@1, MRR and nDCG@5 at
    # every size while halving reranking latency. See the corpus-scale section
    # of the README for the trade-off it loses (Recall@5) and why that one is
    # partly a context-budget artefact.
    chunk_size_tokens: int = Field(256, ge=64, le=4096)
    chunk_overlap_tokens: int = Field(51, ge=0, le=2048)   # 20% of the chunk size
    min_chunk_tokens: int = Field(24, ge=1)
    merge_small_sections: bool = True
    chars_per_token: float = Field(
        4.0, gt=0,
        description="Heuristic used by the token estimator when no model tokenizer is available.",
    )

    # -------------------------------------------------------------- embedding
    embedding_backend: Literal["auto", "sentence_transformers", "tfidf_svd"] = "auto"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = Field(32, ge=1)
    tfidf_svd_dim: int = Field(384, ge=16)

    # ----------------------------------------------------------- vector store
    vector_backend: Literal["chroma", "numpy"] = "chroma"
    chroma_dir: Path = PROJECT_ROOT / "data" / "processed" / "chroma"
    chroma_collection: str = "rag_chunks"

    # ------------------------------------------------------------------ bm25
    bm25_dir: Path = PROJECT_ROOT / "data" / "processed" / "bm25"
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # ------------------------------------------------------------- retrieval
    retrieval_method: Literal["dense", "sparse", "hybrid"] = "hybrid"
    dense_top_k: int = Field(20, ge=1)
    sparse_top_k: int = Field(20, ge=1)
    fusion_top_k: int = Field(20, ge=1)
    rrf_k: int = Field(60, ge=1)
    rrf_dense_weight: float = 1.0
    rrf_sparse_weight: float = 1.0

    # ------------------------------------------------------------ second hop
    # Off by default: it helps one question type and costs a second retrieval
    # on every query, and that trade is measured rather than assumed. See the
    # corpus README's per-type table and app/retrieval/second_hop.py.
    second_hop_enabled: bool = False
    second_hop_feedback_chunks: int = Field(2, ge=1)
    second_hop_terms: int = Field(8, ge=1)
    second_hop_weight: float = Field(0.5, ge=0.0)

    # ------------------------------------------------------------- reranking
    rerank_enabled: bool = True
    reranker_backend: Literal["auto", "cross_encoder", "lexical", "none"] = "auto"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = Field(5, ge=1)
    rerank_batch_size: int = Field(32, ge=1)

    # --------------------------------------------------------------- context
    context_token_budget: int = Field(2400, ge=256)
    context_dedup_threshold: float = Field(0.92, ge=0.0, le=1.0)

    # ------------------------------------------------------------- no-answer
    no_answer_enabled: bool = True
    # Score scales differ per stage, so each stage owns its own threshold.
    # cross-encoder logits are roughly [-11, +11]; the lexical fallback is
    # roughly [0, 5]; RRF is ~[0, 0.04]; cosine is [-1, 1].
    min_rerank_score: float = -6.0
    min_lexical_rerank_score: float = 1.25
    min_fusion_score: float = 0.0
    min_dense_similarity: float = 0.15
    min_sparse_score: float = 0.5
    no_answer_message: str = (
        "I could not find enough information in the indexed documents to answer "
        "this question confidently."
    )

    # ------------------------------------------------------------------- llm
    llm_provider: Literal["auto", "openai_compatible", "anthropic", "extractive"] = "auto"
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_temperature: float = Field(0.0, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(700, ge=32)
    llm_timeout_s: float = Field(60.0, gt=0)

    anthropic_api_key: str = ""
    anthropic_base_url: str = "https://api.anthropic.com/v1"

    # --------------------------------------------------------------- serving
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_url: str = "http://localhost:8000"
    max_upload_mb: int = Field(25, ge=1)

    # --------------------------------------------------------- observability
    log_level: str = "INFO"
    log_json: bool = True
    log_chunk_text: bool = False

    @field_validator("chunk_overlap_tokens")
    @classmethod
    def _overlap_smaller_than_chunk(cls, v: int, info) -> int:
        size = info.data.get("chunk_size_tokens")
        if size is not None and v >= size:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")
        return v

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.raw_dir, self.processed_dir,
                     self.evaluation_dir, self.chroma_dir, self.bm25_dir):
            Path(path).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
