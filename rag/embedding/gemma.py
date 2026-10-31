"""EmbeddingGemma 2 (local, via Ollama) as a LangChain Embeddings class, plus image embedding.

Prompt format from Google's EmbeddingGemma docs:
    documents: "title: {title} | text: {text}"
    queries:   "task: question answering | query: {question}"
Images go to Ollama as input=[{"image": <base64>}] (the "images" field does not work).
"""

import base64
import os
import time
from pathlib import Path

import httpx
from langchain_core.embeddings import Embeddings

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "embeddinggemma-2:440m")


def doc_prompt(text: str, title: str = "none") -> str:
    return f"title: {title} | text: {text}"


def query_prompt(question: str) -> str:
    return f"task: question answering | query: {question}"


class GemmaEmbeddings(Embeddings):
    def __init__(self, model: str = EMBED_MODEL, url: str = OLLAMA_URL, timeout: float = 120.0):
        self.model = model
        self._client = httpx.Client(base_url=url, timeout=timeout)

    def embed_raw(self, inputs: list, retries: int = 3) -> list[list[float]]:
        """Embed inputs as given (prompts already applied, or {"image": b64} dicts)."""
        for attempt in range(retries):
            try:
                r = self._client.post("/api/embed", json={"model": self.model, "input": inputs})
                r.raise_for_status()
                return r.json()["embeddings"]
            except (httpx.HTTPError, KeyError):
                if attempt == retries - 1:
                    raise
                time.sleep(2**attempt)
        return []

    # LangChain interface -------------------------------------------------
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Texts already formatted with doc_prompt() are sent as-is."""
        return self.embed_raw([t if t.startswith("title: ") else doc_prompt(t) for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self.embed_raw([query_prompt(text)])[0]

    # images ----------------------------------------------------------------
    def embed_images(self, paths: list[str | Path]) -> list[list[float]]:
        # one request per image: batching gave no speedup on M1 (0.86 s/image either way)
        out = []
        for p in paths:
            b64 = base64.b64encode(Path(p).read_bytes()).decode()
            out.extend(self.embed_raw([{"image": b64}]))
        return out
