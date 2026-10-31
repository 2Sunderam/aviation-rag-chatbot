"""Hybrid retrieval: dense (EmbeddingGemma 2, cosine) + BM25 (Qdrant server-side), fused with RRF.

All fusion happens inside Qdrant in one Query API call:
    prefetch dense top-N  +  prefetch BM25 top-N  ->  FusionQuery(RRF)  ->  top-k
Text chunks and figures are searched separately (payload filter on "type") so the
UI can show "retrieved chunks" and "related figures" side by side.
"""

import time
from dataclasses import dataclass, field

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from qdrant_client import QdrantClient, models

from rag.embedding.gemma import GemmaEmbeddings
from rag.embedding.store import COLLECTION, DENSE, SPARSE, bm25_doc, get_client


@dataclass
class Hit:
    id: str
    rrf_score: float
    dense_rank: int | None  # 1-based rank in the dense list (None = not in top-N)
    bm25_rank: int | None  # 1-based rank in the BM25 list
    dense_score: float | None
    bm25_score: float | None
    payload: dict

    @property
    def citation(self) -> str:
        p = self.payload
        pages = f"p. {p['page']}" if p.get("page_end", p["page"]) == p["page"] else f"pp. {p['page']}-{p['page_end']}"
        return f"{p['title']}, {pages}"


@dataclass
class RetrievalResult:
    query: str
    chunks: list[Hit]
    figures: list[Hit]
    timings_ms: dict = field(default_factory=dict)


def _filter(kind: str, pdf_ids: list[str] | None) -> models.Filter:
    must = [models.FieldCondition(key="type", match=models.MatchValue(value=kind))]
    if pdf_ids:
        must.append(models.FieldCondition(key="pdf_id", match=models.MatchAny(any=pdf_ids)))
    return models.Filter(must=must)


class HybridSearcher:
    def __init__(self, client: QdrantClient | None = None, embeddings: GemmaEmbeddings | None = None,
                 collection: str = COLLECTION):
        self.client = client or get_client()
        self.emb = embeddings or GemmaEmbeddings()
        self.collection = collection

    def _search(self, qvec: list[float], query: str, kind: str, k: int, prefetch_k: int,
                pdf_ids: list[str] | None, explain: bool) -> list[Hit]:
        flt = _filter(kind, pdf_ids)
        dense = models.Prefetch(query=qvec, using=DENSE, limit=prefetch_k, filter=flt)
        sparse = models.Prefetch(query=bm25_doc(query), using=SPARSE, limit=prefetch_k, filter=flt)
        fused = self.client.query_points(
            self.collection, prefetch=[dense, sparse], query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=k, with_payload=True,
        ).points

        ranks: dict[str, dict[str, tuple[int, float]]] = {"dense": {}, "bm25": {}}
        if explain:  # each retriever's own ranking, for the retrieval visualization
            d, s = self.client.query_batch_points(self.collection, requests=[
                models.QueryRequest(query=qvec, using=DENSE, limit=prefetch_k, filter=flt),
                models.QueryRequest(query=bm25_doc(query), using=SPARSE, limit=prefetch_k, filter=flt),
            ])
            ranks["dense"] = {str(p.id): (r, p.score) for r, p in enumerate(d.points, 1)}
            ranks["bm25"] = {str(p.id): (r, p.score) for r, p in enumerate(s.points, 1)}

        hits = []
        for p in fused:
            dr, br = ranks["dense"].get(str(p.id)), ranks["bm25"].get(str(p.id))
            hits.append(Hit(str(p.id), p.score, dr and dr[0], br and br[0], dr and dr[1], br and br[1], p.payload))
        return hits

    def search(self, query: str, k: int = 6, fig_k: int = 3, prefetch_k: int = 50,
               pdf_ids: list[str] | None = None, explain: bool = True) -> RetrievalResult:
        t0 = time.perf_counter()
        qvec = self.emb.embed_query(query)
        t1 = time.perf_counter()
        chunks = self._search(qvec, query, "text", k, prefetch_k, pdf_ids, explain)
        t2 = time.perf_counter()
        figures = self._search(qvec, query, "figure", fig_k, prefetch_k, pdf_ids, explain) if fig_k else []
        t3 = time.perf_counter()
        return RetrievalResult(query, chunks, figures, {
            "embed": round((t1 - t0) * 1000), "search_text": round((t2 - t1) * 1000),
            "search_figures": round((t3 - t2) * 1000), "total": round((t3 - t0) * 1000),
        })


class FAARetriever(BaseRetriever):
    """LangChain retriever wrapper (text chunks only) for use in chains."""

    searcher: HybridSearcher
    k: int = 6

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        res = self.searcher.search(query, k=self.k, fig_k=0, explain=False)
        return [Document(page_content=h.payload["text"],
                         metadata={**{k: v for k, v in h.payload.items() if k != "text"},
                                   "citation": h.citation, "rrf_score": h.rrf_score})
                for h in res.chunks]
