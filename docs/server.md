# Server + UI

Code: `rag/server/app.py`, `rag/server/static/{index.html, app.js, style.css}`
Run: `uv run uvicorn rag.server.app:app --port 8000` -> http://localhost:8000

On startup the chat engine is created and the embedding model is warmed up.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | chat UI |
| POST | `/api/chat` | chat, streamed as Server-Sent Events |
| POST | `/api/reset/{session_id}` | clear a session's memory |
| GET | `/api/stats` | ingestion numbers per book + vector counts (UI header) |
| GET | `/figures/{pdf_id}/{name}.png` | figure image (path-checked to stay inside `data/processed`) |

`POST /api/chat` body:

```json
{"session_id": "abc", "message": "What do VASI lights indicate?", "k": null, "fig_k": 3, "pdf_ids": null}
```

`k` = null lets the planner decide (5 / 7 / 8 chunks). The response is a stream of
`data: {json}\n\n` lines with event types `retrieval`, `token`, `done`, `error` (see [generation.md](generation.md)).

```bash
curl -N -X POST localhost:8000/api/chat -H 'Content-Type: application/json' \
  -d '{"session_id":"t1","message":"What causes carburetor icing?"}'
```

## UI (plain HTML/JS, no framework)

- **Chat:** streamed answer, citation chips `[1]` / `[F1]` that scroll to and highlight the source,
  "Sources" line, end-to-end time and model. "New chat" clears memory.
- **Retrieval panel:**
  - timing bar: planner, embed, search, LLM first token, generation, end-to-end
  - planned search queries (Q1..Q3)
  - figure thumbnails with caption and citation, click to zoom
  - chunk cards: RRF score bar, dense rank + cosine, BM25 rank + score, sub-query badge,
    colour = found by both / semantic only / keyword only, expandable text
- **Header pill:** ingestion pipeline (PDFs -> pages -> chunks -> vectors) and per-book table.
- Session id is kept in `localStorage`.
