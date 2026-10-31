"""Qdrant hybrid collection: dense (EmbeddingGemma 2) + sparse (BM25, computed by Qdrant).

Each point has two named vectors:
    "dense": 768-dim cosine, HNSW index      -> semantic search
    "bm25":  sparse, IDF modifier            -> keyword search, built server-side by
             models.Document(model="Qdrant/bm25") (needs Qdrant >= 1.15.3)
Text chunks and figures share the collection, told apart by payload "type".
Point ids are uuid5(chunk_id / fig_id): re-runs overwrite instead of duplicating.
"""

import os
import uuid

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()

COLLECTION = os.getenv("QDRANT_HYBRID_COLLECTION", "faa_hybrid")
DENSE, SPARSE = "dense", "bm25"
BM25_MODEL = "Qdrant/bm25"
BM25_AVG_LEN = 480.0  # average words per chunk (default 256 assumes shorter docs)
_NS = uuid.UUID("6f1c2b8e-3d4a-4e5f-9a7b-1c2d3e4f5a6b")


def point_id(key: str) -> str:
    return str(uuid.uuid5(_NS, key))


def get_client() -> QdrantClient:
    return QdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY"),
        check_compatibility=False,  # client 1.20 vs server 1.15 is fine for these calls
        cloud_inference=True,  # send models.Document to the server (BM25 runs in Qdrant)
    )


def bm25_doc(text: str) -> models.Document:
    return models.Document(text=text, model=BM25_MODEL, options={"avg_len": BM25_AVG_LEN})


def ensure_collection(client: QdrantClient, dim: int, name: str = COLLECTION) -> None:
    if client.collection_exists(name):
        return
    client.create_collection(
        name,
        vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
        sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=128),
    )
    for field in ("pdf_id", "type"):
        client.create_payload_index(name, field, models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(name, "page", models.PayloadSchemaType.INTEGER)


def existing_ids(client: QdrantClient, ids: list[str], name: str = COLLECTION) -> set[str]:
    if not client.collection_exists(name):
        return set()
    return {str(p.id) for p in client.retrieve(name, ids, with_payload=False, with_vectors=False)}


def upsert(client: QdrantClient, ids: list[str], vectors: list[list[float]], bm25_texts: list[str],
           payloads: list[dict], name: str = COLLECTION) -> None:
    points = [
        models.PointStruct(id=i, vector={DENSE: v, SPARSE: bm25_doc(t)}, payload=p)
        for i, v, t, p in zip(ids, vectors, bm25_texts, payloads, strict=True)
    ]
    client.upsert(name, points=points, wait=True)


def bm25_text(payload: dict) -> str:
    """What BM25 indexes: section heading + text (figures: their caption)."""
    return f"{payload.get('section', '')}\n{payload.get('text', '')}".strip()
