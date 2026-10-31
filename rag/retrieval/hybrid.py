"""Hybrid retrieval: dense (EmbeddingGemma 2, cosine) + BM25 (Qdrant server-side), fused with RRF.

Supports several sub-queries (from the query planner) in ONE Qdrant call:
    for each query: prefetch dense top-N + prefetch BM25 top-N
    -> FusionQuery(RRF) over all prefetches -> top-k
Text chunks and figures are searched separately (payload filter on "type") so the
UI can show "retrieved chunks" and "related figures" side by side.
"""

import time
from dataclasses import dataclass, field

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from qdrant_client import QdrantClient, models

from rag.embedding.gemma import GemmaEmbeddings, query_prompt
from rag.embedding.store import COLLECTION, DENSE, SPARSE, bm25_doc, get_client

PREFETCH_K = 30  # candidates per retriever per sub-query


@dataclass
class Hit:
    id: str
    rrf_score: float
    dense_rank: int | None  # best 1-based rank in any sub-query's dense list (None = not in top-N)
    bm25_rank: int | None  # best rank in any sub-query's BM25 list
    dense_score: float | None
    bm25_score: float | None
    query_idx: int | None  # which sub-query ranked it best (index into RetrievalResult.queries)
    payload: dict

    @property
    def citation(self) -> str:
        p = self.payload
        pages = f"p. {p['page']}" if p.get("page_end", p["page"]) == p["page"] else f"pp. {p['page']}-{p['page_end']}"
        return f"{p['title']}, {pages}"


@dataclass
class RetrievalResult:
    queries: list[str]
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

    def _search(self, qvecs: list[list[float]], queries: list[str], kind: str, k: int,
                pdf_ids: list[str] | None, explain: bool) -> list[Hit]:
        flt = _filter(kind, pdf_ids)
        prefetch = []
        for vec, q in zip(qvecs, queries):
            prefetch += [models.Prefetch(query=vec, using=DENSE, limit=PREFETCH_K, filter=flt),
                         models.Prefetch(query=bm25_doc(q), using=SPARSE, limit=PREFETCH_K, filter=flt)]
        fused = self.client.query_points(
            self.collection, prefetch=prefetch, query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=k, with_payload=True,
        ).points

        # best rank per retriever across sub-queries, for the retrieval visualization
        best: dict[str, dict[str, tuple[int, float, int]]] = {"dense": {}, "bm25": {}}
        if explain:
            reqs = [models.QueryRequest(query=p.query, using=p.using, limit=PREFETCH_K, filter=flt) for p in prefetch]
            for i, resp in enumerate(self.client.query_batch_points(self.collection, requests=reqs)):
                which = "dense" if i % 2 == 0 else "bm25"
                for rank, p in enumerate(resp.points, 1):
                    prev = best[which].get(str(p.id))
                    if prev is None or rank < prev[0]:
                        best[which][str(p.id)] = (rank, p.score, i // 2)

        hits = []
        for p in fused:
            d, b = best["dense"].get(str(p.id)), best["bm25"].get(str(p.id))
            q_idx = min((x for x in (d, b) if x), key=lambda x: x[0])[2] if (d or b) else None
            hits.append(Hit(str(p.id), p.score, d and d[0], b and b[0], d and d[1], b and b[1], q_idx, p.payload))
        return hits

    def search(self, queries: str | list[str], k: int = 5, fig_k: int = 3,
               pdf_ids: list[str] | None = None, explain: bool = True) -> RetrievalResult:
        queries = [queries] if isinstance(queries, str) else queries
        t0 = time.perf_counter()
        qvecs = self.emb._embed([query_prompt(q) for q in queries])  # one batched embedding call
        t1 = time.perf_counter()
        chunks = self._search(qvecs, queries, "text", k, pdf_ids, explain)
        t2 = time.perf_counter()
        figures = self._search(qvecs, queries, "figure", fig_k, pdf_ids, explain) if fig_k else []
        t3 = time.perf_counter()
        return RetrievalResult(queries, chunks, figures, {
            "embed": round((t1 - t0) * 1000), "search_text": round((t2 - t1) * 1000),
            "search_figures": round((t3 - t2) * 1000), "total": round((t3 - t0) * 1000),
        })


class FAARetriever(BaseRetriever):
    """LangChain retriever wrapper (text chunks only) for use in chains."""

    searcher: HybridSearcher
    k: int = 5

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        res = self.searcher.search(query, k=self.k, fig_k=0, explain=False)
        return [Document(page_content=h.payload["text"],
                         metadata={**{k: v for k, v in h.payload.items() if k != "text"},
                                   "citation": h.citation, "rrf_score": h.rrf_score})
                for h in res.chunks]
