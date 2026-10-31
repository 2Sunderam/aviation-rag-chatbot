# Retrieval: hybrid dense + BM25 + RRF

Code: `rag/retrieval/hybrid.py` · Try: `uv run python -m rag.retrieval.search "What do VASI lights indicate?" [--k 5] [--only 04_instrument_procedures_handbook]`

## One Qdrant call per type

`HybridSearcher.search(queries, k, fig_k)` accepts 1-3 queries (from the planner):

```
for each query:
    Prefetch(dense query vector,  using="dense", limit=30, filter=type)
    Prefetch(BM25 query document, using="bm25",  limit=30, filter=type)
FusionQuery(fusion=RRF)  ->  top k
```

- Text and figures are searched separately (`type` = `text` / `figure`) so the UI can show both.
- All fusion happens inside Qdrant (Reciprocal Rank Fusion uses only ranks, so cosine scores and
  BM25 scores never need to be on the same scale).
- Query embeddings for all sub-queries are computed in one Ollama call.
- Optional filter by book (`pdf_ids`).

## Explain mode (retrieval visualization)

With `explain=True`, each dense and BM25 prefetch is also run on its own (one batch call) to record,
for every fused result, its **best dense rank + cosine score**, **best BM25 rank + score**, and which
sub-query found it (`query_idx`). The UI shows these as "found by both / semantic only / keyword only".

## Result objects

```python
Hit(id, rrf_score, dense_rank, bm25_rank, dense_score, bm25_score, query_idx, payload)
Hit.citation  # "Pilot's Handbook of Aeronautical Knowledge, pp. 169-170"
RetrievalResult(queries, chunks: list[Hit], figures: list[Hit], timings_ms)
```

`FAARetriever` wraps the searcher as a LangChain `BaseRetriever` (returns `Document`s with citation metadata).

## Measured

Search ~20 ms (text) + ~13 ms (figures). Hybrid beats either retriever alone
(MRR 0.83 vs 0.75), see [evaluation.md](evaluation.md).
