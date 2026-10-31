# Architecture

Two pipelines share one Qdrant collection (`faa_hybrid`): an offline **ingestion pipeline** that runs once,
and an online **query pipeline** that runs per chat message.

## Offline: PDF -> vectors (runs once)

```
data/pdfs/*.pdf  (11 books, 3,831 pages)
   │  rag/ingestion     PyMuPDF + pymupdf4llm: per-page markdown, headers/footers dropped
   │                    quality check per page -> RapidOCR only where text is missing/garbage
   │                    figure regions cropped to PNG with captions
   ▼
data/processed/<book>/pages.jsonl  +  figures/*.png
   │  rag/chunking      LangChain markdown-aware recursive splitter, ~800 tokens, 15 % overlap
   │                    page / page_end / section metadata, TOC & noise chunks dropped
   ▼
data/chunks/<book>.jsonl  (4,106 chunks)
   │  rag/embedding     EmbeddingGemma 2 via Ollama (768-dim) for chunks and figure images
   │                    Qdrant computes BM25 sparse vectors server-side
   ▼
Qdrant collection faa_hybrid  (4,106 text points + 2,761 figure points)
   each point: dense vector + bm25 sparse vector + payload (book, page, section, text / caption, path)
```

## Online: question -> cited answer (per message, ~1.2 s median)

```
user question  ──►  memory: last 6 messages of this session (sliding window)
       │
       ▼
Query planner (Mercury 3, structured output)          ~0.5 s
   question + memory -> SearchPlan{queries: 1..3}
   resolves follow-ups ("it" -> "carburetor icing"), splits multi-part questions
       │
       ▼
Embed queries (EmbeddingGemma 2, local)                ~50 ms
       │
       ▼
Qdrant, ONE call per type (text, figures)              ~35 ms
   prefetch per query: dense top-30 + BM25 top-30
   FusionQuery(RRF) -> top 5/7/8 chunks (1/2/3 queries) + top 3 figures
       │
       ▼
Answer (Mercury 3, streamed)                           ~0.6 s
   original question + memory + numbered sources [1..k], [F1..F3]
   rule: answer only from sources, cite every fact, refuse if not found
       │
       ▼
Citations: [n] -> "Book title, p. X" from payload metadata (never from the LLM)
       │
       ▼
FastAPI SSE stream -> UI: answer + citation chips, chunk cards (dense/BM25 ranks), figures, timings
```

If Mercury 3 errors or is rate-limited, every LLM call falls back to Groq
(`gpt-oss-120b` -> `gpt-oss-20b` -> `qwen3.8-27b`).

## Which file does what

| Stage | File | Key function |
|---|---|---|
| Paths, book titles | `rag/config.py` | `TITLES` |
| PDF extraction + OCR + figures | `rag/ingestion/extract.py` | `process_pdf` |
| OCR decision | `rag/ingestion/quality.py` | `needs_ocr`, `word_stats` |
| Text cleanup | `rag/ingestion/clean.py` | `clean_text` |
| Chunking | `rag/chunking/chunker.py` | `chunk_book` |
| Embedding model | `rag/embedding/gemma.py` | `GemmaEmbeddings` |
| Qdrant collection | `rag/embedding/store.py` | `ensure_collection`, `upsert` |
| Embedding runner | `rag/embedding/run.py` | `embed_texts`, `embed_figures` |
| Hybrid search | `rag/retrieval/hybrid.py` | `HybridSearcher.search` |
| Planner, memory, answer | `rag/generation/chat.py` | `ChatEngine.plan`, `ChatEngine.stream` |
| Prompts | `rag/generation/prompts.py` | `PLANNER_PROMPT`, `ANSWER_PROMPT` |
| API + UI | `rag/server/app.py`, `rag/server/static/` | `/api/chat` |
| Evals | `tests/eval/` | see [evaluation.md](evaluation.md) |

## Stage docs

[data](data.md) · [ingestion](ingestion.md) · [chunking](chunking.md) · [embedding](embedding.md) ·
[retrieval](retrieval.md) · [generation](generation.md) · [server](server.md) · [evaluation](evaluation.md)
