"""Streamlit UI: upload -> index -> ask.

Deliberately plain. This is a thin client over the API: it holds no retrieval
logic of its own, so what it shows is exactly what the service returned. All
state lives in the API, which means the UI can be restarted without losing the
index, and the API can be exercised without the UI.

    API_URL=http://localhost:8000 streamlit run frontend/streamlit_app.py
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT_S = float(os.environ.get("UI_TIMEOUT_S", "120"))

st.set_page_config(page_title="Enterprise Hybrid RAG", page_icon="📄", layout="wide")


# ----------------------------------------------------------------- transport
def _get(path: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT_S) as client:
            response = client.get(f"{API_URL}{path}")
            response.raise_for_status()
            return response.json(), None
    except httpx.HTTPStatusError as exc:
        return None, f"HTTP {exc.response.status_code}: {exc.response.text[:300]}"
    except httpx.HTTPError as exc:
        return None, f"Could not reach the API at {API_URL}: {exc}"


def _post(path: str, *, json: dict | None = None,
          files: dict | None = None) -> tuple[dict[str, Any] | None, str | None]:
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT_S) as client:
            response = client.post(f"{API_URL}{path}", json=json, files=files)
            response.raise_for_status()
            return response.json(), None
    except httpx.HTTPStatusError as exc:
        return None, f"HTTP {exc.response.status_code}: {exc.response.text[:300]}"
    except httpx.HTTPError as exc:
        return None, f"Could not reach the API at {API_URL}: {exc}"


def _citation_label(citation: dict[str, Any]) -> str:
    parts = [str(citation.get("source", "unknown"))]
    if citation.get("page") is not None:
        parts.append(f"p.{citation['page']}")
    if citation.get("section"):
        parts.append(str(citation["section"]))
    return " — ".join(parts)


# -------------------------------------------------------------------- sidebar
st.sidebar.title("Enterprise Hybrid RAG")
st.sidebar.caption(f"API: `{API_URL}`")

health, health_error = _get("/health")
if health_error:
    st.sidebar.error(health_error)
    st.sidebar.info("Start the API with `uvicorn app.main:app --reload`.")
elif health:
    if health.get("index_ready"):
        st.sidebar.success(f"Index ready — {health['n_chunks']} chunks")
    else:
        st.sidebar.warning("No index yet. Upload a document and build the index.")
    st.sidebar.markdown(
        f"**Embedding** `{health.get('embedding_backend')}`  \n"
        f"**Reranker** `{health.get('reranker')}`  \n"
        f"**LLM** `{health.get('llm_provider')}`  \n"
        f"**API version** `{health.get('version')}`"
    )
    # An offline fallback changes what the answers mean, so say so in the UI
    # rather than only in the logs.
    if "extractive" in str(health.get("llm_provider", "")):
        st.sidebar.info("Running the offline **extractive** backend: answers are "
                        "sentences selected from the sources, not generated text.")
    if str(health.get("reranker")) == "lexical":
        st.sidebar.info("Reranker is the **lexical** fallback, not a cross-encoder.")

st.sidebar.divider()
st.sidebar.subheader("Retrieval settings")
method = st.sidebar.selectbox("Method", ["hybrid", "dense", "sparse"], index=0,
                              help="hybrid fuses dense and BM25 with Reciprocal Rank Fusion")
rerank = st.sidebar.checkbox("Rerank", value=True,
                             help="Reorder candidates with the reranker before answering")
top_k = st.sidebar.slider("Contexts passed to the LLM (top_k)", 1, 20, 5)

st.sidebar.divider()
st.sidebar.subheader("Documents")
uploaded = st.sidebar.file_uploader("Upload PDF, DOCX or Markdown",
                                    type=["pdf", "docx", "md", "markdown", "txt"])
if uploaded is not None and st.sidebar.button("Upload", use_container_width=True):
    payload, error = _post("/documents/upload",
                           files={"file": (uploaded.name, uploaded.getvalue())})
    if error:
        st.sidebar.error(error)
    else:
        st.sidebar.success(f"Uploaded {payload['filename']} — now rebuild the index.")

if st.sidebar.button("Rebuild index", use_container_width=True):
    with st.spinner("Re-ingesting and rebuilding both indexes…"):
        payload, error = _post("/documents/index", json={"rebuild": True})
    if error:
        st.sidebar.error(error)
    else:
        st.sidebar.success(
            f"{payload['n_documents']} documents, {payload['n_chunks']} chunks "
            f"in {payload['elapsed_ms']:.0f} ms")
        st.rerun()

documents, documents_error = _get("/documents")
if documents and documents.get("documents"):
    with st.sidebar.expander(f"Indexed ({documents['n_documents']})"):
        for entry in documents["documents"]:
            st.caption(f"{entry['source']} — {entry['n_chunks']} chunks")
    if documents.get("pending_files"):
        st.sidebar.warning("Not yet indexed: " + ", ".join(documents["pending_files"]))


# ----------------------------------------------------------------------- main
st.title("Ask the documents")
st.caption("Answers are restricted to the indexed corpus. When retrieval confidence "
           "is too low the service abstains instead of guessing.")

question = st.text_input("Question",
                         placeholder="How many days of paid annual leave do employees get?")
ask = st.button("Ask", type="primary")

if ask and question.strip():
    with st.spinner("Retrieving and answering…"):
        result, error = _post("/query", json={
            "question": question, "top_k": top_k,
            "retrieval_method": method, "rerank": rerank, "include_chunks": True})

    if error:
        st.error(error)
    elif result:
        metadata = result.get("metadata", {})

        if metadata.get("no_answer"):
            st.warning(result["answer"])
            st.caption(f"Abstained: `{metadata.get('no_answer_reason')}` "
                       f"(top score {metadata.get('top_score')})")
        else:
            st.subheader("Answer")
            st.write(result["answer"])

        citations = result.get("citations") or []
        if citations:
            st.subheader("Citations")
            for citation in citations:
                with st.expander(f"[Source {citation['source_index']}] {_citation_label(citation)}"):
                    st.caption(f"chunk `{citation['chunk_id']}` — score {citation.get('score')}")
                    if citation.get("snippet"):
                        st.write(citation["snippet"])

        columns = st.columns(4)
        columns[0].metric("Method", metadata.get("retrieval_method", "-"))
        columns[1].metric("Reranker", metadata.get("reranker", "none"))
        columns[2].metric("Contexts", metadata.get("n_final_contexts", 0))
        columns[3].metric("Latency", f"{metadata.get('latency_ms', 0):.0f} ms")

        chunks = result.get("retrieved_chunks") or []
        if chunks:
            st.subheader(f"Retrieved chunks ({len(chunks)})")
            for chunk in chunks:
                header = (f"#{chunk['rank']} · {chunk['source']}"
                          + (f" · p.{chunk['page']}" if chunk.get("page") else "")
                          + f" · score {chunk['score']:.4f}")
                with st.expander(header):
                    if chunk.get("section"):
                        st.caption(f"Section: {chunk['section']}")
                    st.caption(f"retriever `{chunk['retriever']}` · "
                               f"components `{chunk.get('component_scores', {})}`")
                    st.write(chunk["text"])

        with st.expander("Debug — QueryTrace"):
            st.json(metadata)
            if result.get("extra"):
                st.caption("Extra")
                st.json(result["extra"])
elif ask:
    st.info("Enter a question first.")
