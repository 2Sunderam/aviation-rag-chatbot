"""Retrieval quality on the frozen dataset: Hit@k, Recall@k, Precision@k, MRR for 4 setups.

    uv run python -m tests.eval.eval_retrieval

Setups (same queries, same collection):
    dense            EmbeddingGemma 2 cosine only
    bm25             Qdrant BM25 only
    hybrid           dense + BM25 fused with RRF (one query)
    hybrid+planner   query planner (1-3 sub-queries) -> hybrid RRF  [what the app uses]

Metrics per question (unanswerable rows excluded, they have no gold):
    Hit@k        1 if any gold chunk is in the top k
    Recall@k     share of gold chunks found in the top k (multi-part rows have 2 gold chunks)
    Precision@k  share of the top k that are relevant
    MRR          1 / rank of the first relevant result (0 if none in top 10)
"""

import asyncio
import time

from qdrant_client import models

from rag.embedding.gemma import query_prompt
from rag.embedding.store import COLLECTION, DENSE, SPARSE, bm25_doc
from rag.generation.chat import ChatEngine

from .common import is_relevant, load_dataset, mean, write_result

KS = (1, 3, 5, 10)
MAX_K = max(KS)
TEXT_ONLY = models.Filter(must=[models.FieldCondition(key="type", match=models.MatchValue(value="text"))])


def score(ranked: list[dict], golds: list[dict]) -> dict:
    rel = [any(is_relevant(p, g) for g in golds) for p in ranked]
    first = next((i for i, r in enumerate(rel, 1) if r), None)
    out = {"mrr": 1 / first if first else 0.0, "first_rank": first}
    for k in KS:
        top = ranked[:k]
        out[f"hit@{k}"] = float(any(rel[:k]))
        out[f"recall@{k}"] = sum(any(is_relevant(p, g) for p in top) for g in golds) / len(golds)
        out[f"precision@{k}"] = sum(rel[:k]) / k
    return out


def main() -> None:
    rows = load_dataset(("single", "multi"))
    engine = ChatEngine()
    client, emb = engine.searcher.client, engine.searcher.emb

    def dense(q: str) -> list[dict]:
        vec = emb.embed_raw([query_prompt(q)])[0]
        return [p.payload for p in client.query_points(COLLECTION, query=vec, using=DENSE, limit=MAX_K,
                                                       query_filter=TEXT_ONLY, with_payload=True).points]

    def bm25(q: str) -> list[dict]:
        return [p.payload for p in client.query_points(COLLECTION, query=bm25_doc(q), using=SPARSE, limit=MAX_K,
                                                       query_filter=TEXT_ONLY, with_payload=True).points]

    def hybrid(q: str | list[str]) -> list[dict]:
        return [h.payload for h in engine.searcher.search(q, k=MAX_K, fig_k=0, explain=False).chunks]

    def planned(q: str) -> list[dict]:
        return hybrid(asyncio.run(engine.plan([], q)))

    setups = {"dense": dense, "bm25": bm25, "hybrid": hybrid, "hybrid+planner": planned}
    per_q: dict[str, list[dict]] = {name: [] for name in setups}
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        for name, fn in setups.items():
            s = score(fn(r["question"]), r["gold"])
            per_q[name].append({"id": r["id"], "type": r["type"], **s})
        if i % 10 == 0:
            print(f"{i}/{len(rows)} questions ({time.time() - t0:.0f}s)", flush=True)

    def summarize(items: list[dict]) -> dict:
        keys = ["mrr"] + [f"{m}@{k}" for m in ("hit", "recall", "precision") for k in KS]
        return {k: round(mean([x[k] for x in items]), 3) for k in keys} | {"n": len(items)}

    summary = {name: {"all": summarize(items),
                      "single": summarize([x for x in items if x["type"] == "single"]),
                      "multi": summarize([x for x in items if x["type"] == "multi"])}
               for name, items in per_q.items()}

    cols = ["mrr", "hit@1", "hit@5", "recall@5", "recall@10", "precision@5"]
    lines = [f"# Retrieval eval ({len(rows)} questions: "
             f"{sum(r['type'] == 'single' for r in rows)} single, {sum(r['type'] == 'multi' for r in rows)} multi-part)\n",
             "Relevant = same book + overlapping pages as a gold chunk. Multi-part rows have 2 gold chunks.\n"]
    for subset in ("all", "single", "multi"):
        lines += [f"\n## {subset}\n", "| setup | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        lines += [f"| {name} | " + " | ".join(f"{summary[name][subset][c]:.3f}" for c in cols) + " |"
                  for name in setups]
    md = "\n".join(lines)
    path = write_result("retrieval", {"summary": summary, "per_question": per_q}, md)
    print(md)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
