"""Try retrieval from the terminal.

    uv run python -m rag.retrieval.search "What causes carburetor icing?"
    uv run python -m rag.retrieval.search "VASI" --k 5 --only 04
"""

import argparse

from .hybrid import HybridSearcher


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--figs", type=int, default=3)
    ap.add_argument("--only", nargs="*", help="restrict to pdf_ids")
    a = ap.parse_args()

    res = HybridSearcher().search(a.query, k=a.k, fig_k=a.figs, pdf_ids=a.only)
    print(f"Q: {res.queries}\ntimings ms: {res.timings_ms}\n")
    for i, h in enumerate(res.chunks, 1):
        print(f"{i}. rrf={h.rrf_score:.3f} dense#{h.dense_rank} bm25#{h.bm25_rank} | {h.citation} | {h.payload['section'][:40]}")
        print(f"   {h.payload['text'][:160]!r}")
    for h in res.figures:
        print(f"[fig] rrf={h.rrf_score:.3f} dense#{h.dense_rank} bm25#{h.bm25_rank} | {h.citation} | {h.payload['caption'][:70]}")


if __name__ == "__main__":
    main()
