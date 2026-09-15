# ---------------------------------------------------------------- base image
# One image serves both the API and the UI; they differ only by command, which
# keeps the dependency set identical between them and halves build time.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models \
    SENTENCE_TRANSFORMERS_HOME=/models

WORKDIR /app

# Dependencies first: this layer stays cached until requirements.txt changes.
COPY requirements.txt .
# torch arrives as a dependency of sentence-transformers, and on Linux the PyPI
# wheel is the CUDA build: roughly 3 GB of wheels for a container that has no
# GPU and never asks for one. PyTorch's own index serves `X.Y.Z+cpu`, which
# sorts *above* the plain `X.Y.Z` on PyPI under PEP 440, so adding it as an
# extra index is enough to select the CPU build without pinning a version that
# would then have to be chased every time sentence-transformers moves.
RUN pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      -r requirements.txt \
 && python -c "import torch; print('torch', torch.__version__)"

# ------------------------------------------------------------- model warmup
# Bake the neural weights into the image so the first query is not a download,
# and so a container without hub access still gets the real models instead of
# silently degrading to the offline backends.
#
# Best-effort on purpose: where the build host cannot reach the Hugging Face
# hub the build still succeeds and the application falls back at runtime — a
# fallback it already reports in /health and in every evaluation artefact.
# Build with --build-arg MODEL_WARMUP=false to skip it deliberately.
ARG MODEL_WARMUP=true
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ARG RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
ENV EMBEDDING_MODEL=${EMBEDDING_MODEL} \
    RERANKER_MODEL=${RERANKER_MODEL}
COPY scripts/warm_models.py /tmp/warm_models.py
RUN if [ "$MODEL_WARMUP" = "true" ]; then python /tmp/warm_models.py; fi \
 && rm -f /tmp/warm_models.py

# ----------------------------------------------------------- application code
COPY app ./app
COPY frontend ./frontend
COPY scripts ./scripts
COPY tests ./tests
COPY pytest.ini ./

# data/ is a mounted volume at runtime; create the tree so a fresh container
# starts cleanly even against an empty mount.
RUN mkdir -p /app/data/raw /app/data/processed /app/data/evaluation

# Never run as root. Ownership covers the writable paths: index and model cache.
RUN useradd --create-home --uid 10001 raguser \
 && mkdir -p /models \
 && chown -R raguser:raguser /app /models
USER raguser

# ------------------------------------------------------------------ services
FROM base AS api
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://localhost:8000/health',timeout=4).status_code==200 else 1)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS ui
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://localhost:8501/_stcore/health',timeout=4).status_code==200 else 1)"
CMD ["streamlit", "run", "frontend/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
