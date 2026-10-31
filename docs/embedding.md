# Embedding + Qdrant collection

Code: `rag/embedding/` · Run:

```bash
uv run python -m rag.embedding.run --texts                    # all chunks
uv run python -m rag.embedding.run --figures --all-figures    # all figures
uv run python -m rag.embedding.run --texts --only 02 --limit 20   # quick test
```

## Model: EmbeddingGemma 2 (`embeddinggemma-2:440m`, via Ollama)

- Open model, runs locally (Apple GPU), ~680 MB in memory. Text **and** images in one 768-dim space.
- Prompts from Google's docs:
  - chunks: `title: <Book> > <Section> | text: <chunk>`
  - queries: `task: question answering | query: <question>`
- Images are sent as `input=[{"image": <base64>}]` (Ollama's `images` field does not work for embed).
- Speed on M1: ~5 chunks/s, ~0.9 s per image (size does not matter), ~25 ms per query.
- `GemmaEmbeddings` implements LangChain's `Embeddings` interface (`gemma.py`).

## Collection `faa_hybrid` (`store.py`)

| Named vector | Type | Purpose |
|---|---|---|
| `dense` | 768-dim, cosine, HNSW (m=16, ef_construct=128) | semantic search |
| `bm25` | sparse, IDF modifier | keyword search; vectors computed **inside Qdrant** from text (`models.Document(model="Qdrant/bm25")`, needs Qdrant >= 1.15.3) |

- BM25 indexes `section + text` for chunks and the caption for figures (`avg_len` = 480 words).
- Payload indexes: `pdf_id`, `type` (keyword), `page` (integer).
- Point id = `uuid5(chunk_id or fig_id)`: re-runs overwrite instead of duplicating, finished points are skipped.

Payload (text point): every chunk field from `chunks/*.jsonl` + `"type": "text"`.
Payload (figure point): `fig_id, pdf_id, filename, title, page, caption, path, width, height, "type": "figure", "text": caption`.

Counts: 4,106 text + 2,761 figure points = 6,867.

## Notes

- `migrate.py` copied the first dense-only run (`faa_handbooks`) into the hybrid collection without
  re-embedding. Not needed for a fresh setup: `run.py` writes straight to `faa_hybrid`.
- Mercury/Groq are not used here; embeddings are 100 % local and open.
