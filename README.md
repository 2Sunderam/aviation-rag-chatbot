# FAA Handbooks RAG Chatbot

Question answering over 11 FAA aviation handbooks (3,831 pages) with cited sources (book + page),
figure retrieval, and a 1-2 s response time.

- **Embeddings:** EmbeddingGemma 2 (`embeddinggemma-2:440m`, open, local via Ollama), text + images
- **Vector DB:** Qdrant (open source), hybrid search: dense HNSW + BM25 fused with RRF
- **LLM:** Mercury 3 (Inception) for planning and answers, Groq models as automatic fallback
- **Stack:** Python 3.12, uv, PyMuPDF + pymupdf4llm, RapidOCR, LangChain, FastAPI, plain HTML/JS UI

How everything fits together: [docs/architecture.md](docs/architecture.md).

---

## 1. Prerequisites

| Tool | Install | Check |
|---|---|---|
| uv | `curl -LsSf https://astral.sh/uv/install.sh \| sh` | `uv --version` |
| Docker | Docker Desktop | `docker info` |
| Ollama | `brew install ollama` (macOS) | `ollama --version` |

## 2. Setup

```bash
uv sync                                  # app dependencies
uv sync --group eval                     # + Ragas, only needed for the evals
ollama pull embeddinggemma-2:440m        # local embedding model (~700 MB)
```

Create `.env` in the project root:

```bash
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=<your-qdrant-api-key>
INCEPTION_API_KEY=<mercury-3-key>        # answers + query planner
GROQ_API_KEY=<groq-key>                  # optional fallback LLMs
# optional
# QDRANT_HYBRID_COLLECTION=faa_hybrid
# OLLAMA_URL=http://localhost:11434
# EMBED_MODEL=embeddinggemma-2:440m
# ANSWER_MODEL=mercury-3
# ANSWER_REASONING=instant               # or "low" (slightly more careful, ~0.5 s slower)
```

Start Qdrant (same API key as in `.env`):

```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -e QDRANT__SERVICE__API_KEY=<your-qdrant-api-key> \
  -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant:v1.15.4
curl localhost:6333/healthz              # -> healthz check passed
```

Qdrant must be **v1.15.3 or newer** (server-side BM25).

## 3. Data

The PDFs and processed data are not in git (two PDFs exceed GitHub's 100 MB file limit).

Download the data from **[Google Drive](https://drive.google.com/drive/folders/1j5tXmtN4bHrlV_OSb0Tz6n32vqZU1Ofo?usp=drive_link)** and unzip into `data/`:

```
data/
  pdfs/                              11 source PDFs (sources: docs/data.md)
  processed/<book>/pages.jsonl       extracted pages      (optional: skips ingestion)
  processed/<book>/figures/*.png     figure crops         (optional: skips ingestion)
  chunks/<book>.jsonl                chunks               (optional: skips chunking)
```

Then build the vector database (your own Qdrant):

```bash
uv run python -m rag.ingestion.run --workers 2      # PDF -> per-page markdown + figures   (~20 min, skip if processed/ exists)
uv run python -m rag.chunking.run                   # pages -> chunks                       (seconds)
uv run python -m rag.embedding.run --texts          # chunks -> Qdrant (dense + BM25)       (~15-20 min)
uv run python -m rag.embedding.run --figures --all-figures   # figures -> Qdrant           (~40 min)
```

Every step is resumable: finished books / stored points are skipped on re-run.
Only `data/pdfs/` is required; `processed/` and `chunks/` just save time.

## 4. Run the chatbot

```bash
uv run uvicorn rag.server.app:app --port 8000
```

Open **http://localhost:8000**: chat on the left, retrieval visualization on the right
(chunks with dense/BM25 ranks, figures, timing bar). Click the pill in the header for the ingestion pipeline stats.

Terminal-only retrieval check:

```bash
uv run python -m rag.retrieval.search "What do VASI lights indicate?"
```

## 5. Evaluation

```bash
uv run python -m tests.eval.build_dataset                 # once; dataset is frozen (use --force to rebuild)
uv run python -m tests.eval.eval_retrieval                # Recall@k, Precision@k, MRR       (~2 min)
uv run python -m tests.eval.eval_latency                  # p50 / p95 per stage              (~3 min)
uv run --group eval python -m tests.eval.eval_generation  # Ragas + citations + refusals     (~30 min)
uv run --group eval python -m tests.eval.eval_generation --skip-ragas   # citations + refusals only
```

Results: `tests/eval/results/*_latest.md`. Summary and method: [docs/evaluation.md](docs/evaluation.md).

| Metric | Result |
|---|---|
| MRR / Recall@5 (hybrid + planner) | 0.835 / 0.944 |
| p95 end-to-end latency | 1.69 s |
| Ragas faithfulness | 0.982 |
| Hallucination rate (answers with >= 1 unsupported claim) | 9.9 % (0 % severe) |
| Citation accuracy (cited chunk supports its sentence) | 0.79 |
| Unanswerable questions refused | 100 % |

## 6. Project layout

```
rag/
  config.py          paths + book titles
  ingestion/         PDF -> per-page markdown, OCR when needed, figure crops   docs/ingestion.md
  chunking/          structure-aware recursive chunking with page metadata     docs/chunking.md
  embedding/         EmbeddingGemma 2 + Qdrant hybrid collection               docs/embedding.md
  retrieval/         hybrid dense + BM25 + RRF search                          docs/retrieval.md
  generation/        query planner, memory, Mercury 3 answers, citations       docs/generation.md
  server/            FastAPI + static UI                                       docs/server.md
tests/eval/          frozen dataset + retrieval / latency / generation evals   docs/evaluation.md
docs/                documentation
```
