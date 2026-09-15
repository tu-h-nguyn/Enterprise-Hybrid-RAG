"""Embedding backends.

Two backends implement the same interface:

``SentenceTransformerEmbedder``
    The production default (``all-MiniLM-L6-v2``). Requires the model weights,
    which are downloaded from the Hugging Face hub on first use.

``TfidfSvdEmbedder``
    A fully offline, dependency-light fallback: TF-IDF over word *and*
    character n-grams, reduced with truncated SVD (i.e. LSA) and L2-normalised
    so that an inner product is a cosine similarity. It is genuinely dense and
    genuinely learned, but it is a *lexical-semantic* model: it captures term
    co-occurrence, not the sentence-level semantics of a neural encoder.

Why keep both? The system must remain runnable (and evaluable) in restricted
environments — air-gapped machines, CI, or a container with no hub access. The
selection is explicit in every evaluation artefact so results are never
attributed to the wrong model.
"""

from __future__ import annotations

import abc
import logging
import pickle
from pathlib import Path

import numpy as np

from app.config.settings import Settings

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    pass


class BaseEmbedder(abc.ABC):
    name: str = "base"
    requires_fit: bool = False

    @property
    @abc.abstractmethod
    def dimension(self) -> int: ...

    @property
    def is_fitted(self) -> bool:
        return True

    def fit(self, corpus: list[str]) -> BaseEmbedder:
        return self

    @abc.abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]

    def save(self, directory: Path) -> None:
        return None

    def load(self, directory: Path) -> bool:
        return True

    def describe(self) -> dict[str, object]:
        return {"backend": self.name, "dimension": self.dimension if self.is_fitted else None}


class SentenceTransformerEmbedder(BaseEmbedder):
    name = "sentence_transformers"

    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise EmbeddingError("sentence-transformers is not installed") from exc
        try:
            self._model = SentenceTransformer(model_name)
        except Exception as exc:  # network / missing weights
            raise EmbeddingError(f"Could not load embedding model '{model_name}': {exc}") from exc
        self.model_name = model_name
        self.batch_size = batch_size
        dimension = self._model.get_sentence_embedding_dimension()
        if dimension is None:
            raise EmbeddingError(
                f"Model '{model_name}' did not report an embedding dimension")
        self._dim = int(dimension)

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        vectors = self._model.encode(
            texts, batch_size=self.batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def describe(self) -> dict[str, object]:
        return {"backend": self.name, "model": self.model_name, "dimension": self._dim}


class TfidfSvdEmbedder(BaseEmbedder):
    name = "tfidf_svd"
    requires_fit = True

    def __init__(self, dim: int = 384, random_state: int = 42) -> None:
        self.target_dim = dim
        self.random_state = random_state
        self._pipeline = None
        self._dim = 0

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def is_fitted(self) -> bool:
        return self._pipeline is not None

    def fit(self, corpus: list[str]) -> TfidfSvdEmbedder:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import FeatureUnion, Pipeline
        from sklearn.preprocessing import Normalizer

        if not corpus:
            raise EmbeddingError("Cannot fit TfidfSvdEmbedder on an empty corpus")

        # Word n-grams capture terminology; character n-grams give robustness to
        # morphology and typos ("reimbursement" vs "reimburse").
        features = FeatureUnion([
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                     sublinear_tf=True, min_df=1, max_df=0.9)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                     sublinear_tf=True, min_df=2)),
        ])
        matrix = features.fit_transform(corpus)
        n_components = int(min(self.target_dim, max(2, min(matrix.shape) - 1)))
        svd = TruncatedSVD(n_components=n_components, random_state=self.random_state)
        svd.fit(matrix)
        self._pipeline = Pipeline([("features", features), ("svd", svd),
                                   ("norm", Normalizer(copy=False))])
        self._dim = n_components
        logger.info("tfidf_svd_fitted", extra={"documents": len(corpus), "dim": n_components})
        return self

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if self._pipeline is None:
            raise EmbeddingError("TfidfSvdEmbedder.fit() must be called before embedding")
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        vectors = self._pipeline.transform(texts)
        return np.asarray(vectors, dtype=np.float32)

    def save(self, directory: Path) -> None:
        if self._pipeline is None:
            return
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "tfidf_svd.pkl").open("wb") as fh:
            pickle.dump({"pipeline": self._pipeline, "dim": self._dim}, fh)

    def load(self, directory: Path) -> bool:
        path = Path(directory) / "tfidf_svd.pkl"
        if not path.exists():
            return False
        try:
            with path.open("rb") as fh:
                state = pickle.load(fh)
        except Exception as exc:  # pragma: no cover - corrupt artefact
            logger.warning("tfidf_svd_load_failed", extra={"error": str(exc)})
            return False
        self._pipeline = state["pipeline"]
        self._dim = int(state["dim"])
        return True

    def describe(self) -> dict[str, object]:
        return {"backend": self.name, "model": "tfidf(word1-2 + char3-5)+svd",
                "dimension": self._dim}


def build_embedder(settings: Settings) -> BaseEmbedder:
    """Instantiate the configured embedder.

    ``auto`` prefers the neural encoder and degrades to the offline backend if
    the weights cannot be obtained, logging loudly so the substitution is never
    silent in results.
    """
    backend = settings.embedding_backend
    if backend in {"auto", "sentence_transformers"}:
        try:
            return SentenceTransformerEmbedder(settings.embedding_model,
                                               settings.embedding_batch_size)
        except EmbeddingError as exc:
            if backend == "sentence_transformers":
                raise
            logger.warning(
                "embedding_backend_fallback",
                extra={"requested": settings.embedding_model, "error": str(exc),
                       "fallback": "tfidf_svd"},
            )
    return TfidfSvdEmbedder(dim=settings.tfidf_svd_dim)
