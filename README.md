# Enterprise Hybrid RAG

The project lives in [`enterprise-hybrid-rag/`](enterprise-hybrid-rag/) — see
**[its README](enterprise-hybrid-rag/README.md)** for the architecture, the
measured benchmark results and instructions for running it.

```bash
cd enterprise-hybrid-rag
pip install -r requirements.txt
python scripts/ingest.py
uvicorn app.main:app --reload
```
