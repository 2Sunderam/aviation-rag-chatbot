"""FastAPI server: chat API (SSE streaming), figure files, corpus stats, static UI.

    uv run uvicorn rag.server.app:app --port 8000
    open http://localhost:8000
"""

import glob
import json
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from qdrant_client import models

from rag.config import CHUNKS_DIR, PROCESSED_DIR, TITLES
from rag.embedding.store import COLLECTION
from rag.generation.chat import ChatEngine
from rag.retrieval.hybrid import Hit

STATIC = Path(__file__).parent / "static"
engine: ChatEngine | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global engine
    engine = ChatEngine()
    engine.searcher.emb.embed_query("warm up")  # load the embedding model before the first user query
    yield


app = FastAPI(title="FAA Handbooks RAG", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=2000)
    k: int = Field(5, ge=1, le=12)
    fig_k: int = Field(3, ge=0, le=6)
    pdf_ids: list[str] | None = None


def _hit_json(h: Hit) -> dict:
    p = dict(h.payload)
    if p.get("type") == "figure":
        p["url"] = f"/figures/{p['pdf_id']}/{Path(p['path']).name}"
        p.pop("path", None)
    return {**{k: v for k, v in asdict(h).items() if k != "payload"}, "citation": h.citation, "payload": p}


def _sse(event: dict) -> str:
    if event["type"] == "retrieval":
        event = {**event, "chunks": [_hit_json(h) for h in event["chunks"]],
                 "figures": [_hit_json(h) for h in event["figures"]]}
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.post("/api/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    async def events():
        try:
            async for ev in engine.stream(req.session_id, req.message, req.k, req.fig_k, req.pdf_ids):
                yield _sse(ev)
        except Exception as e:  # surface errors to the UI instead of a silent broken stream
            yield _sse({"type": "error", "message": str(e)})

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.post("/api/reset/{session_id}")
def reset(session_id: str) -> dict:
    engine.reset(session_id)
    return {"ok": True}


@app.get("/figures/{pdf_id}/{name}")
def figure(pdf_id: str, name: str) -> FileResponse:
    path = (PROCESSED_DIR / pdf_id / "figures" / name).resolve()
    if not path.is_relative_to(PROCESSED_DIR.resolve()) or path.suffix != ".png" or not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/png")


@app.get("/api/stats")
def stats() -> dict:
    """Ingestion pipeline numbers for the UI header: PDFs -> pages -> chunks -> vectors."""
    books = []
    for pages_file in sorted(glob.glob(str(PROCESSED_DIR / "*/pages.jsonl"))):
        pdf_id = Path(pages_file).parent.name
        pages = [json.loads(l) for l in open(pages_file)]
        chunk_file = CHUNKS_DIR / f"{pdf_id}.jsonl"
        books.append({
            "pdf_id": pdf_id, "title": TITLES.get(pdf_id, pdf_id), "pages": len(pages),
            "ocr_pages": sum(p["ocr"] for p in pages), "figures": sum(len(p["figures"]) for p in pages),
            "chunks": sum(1 for _ in open(chunk_file)) if chunk_file.exists() else 0,
        })
    client = engine.searcher.client
    count = lambda kind: client.count(COLLECTION, count_filter=models.Filter(must=[
        models.FieldCondition(key="type", match=models.MatchValue(value=kind))])).count
    return {"books": books, "collection": COLLECTION,
            "vectors": {"text": count("text"), "figure": count("figure")},
            "embedding_model": engine.searcher.emb.model, "llm": engine.answer_model}
