"""Copy the first dense-only run (faa_handbooks) into the hybrid collection (faa_hybrid).

Reuses the stored dense vectors (no re-embedding); Qdrant computes BM25 for each point.

    uv run python -m rag.embedding.migrate
"""

import time

from .store import COLLECTION, LEGACY_COLLECTION, bm25_text, ensure_collection, get_client, upsert


def main(batch: int = 256) -> None:
    client = get_client()
    total = client.count(LEGACY_COLLECTION).count
    print(f"copying {total} points {LEGACY_COLLECTION} -> {COLLECTION}")
    t0, done, offset = time.time(), 0, None
    while True:
        points, offset = client.scroll(LEGACY_COLLECTION, limit=batch, offset=offset,
                                       with_payload=True, with_vectors=True)
        if not points:
            break
        ensure_collection(client, len(points[0].vector))
        upsert(client, [str(p.id) for p in points], [p.vector for p in points],
               [bm25_text(p.payload) for p in points], [p.payload for p in points])
        done += len(points)
        print(f"{done}/{total} ({time.time() - t0:.0f}s)", flush=True)
        if offset is None:
            break
    print("hybrid collection size:", client.count(COLLECTION).count)


if __name__ == "__main__":
    main()
