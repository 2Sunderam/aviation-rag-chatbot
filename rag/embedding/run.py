"""Embed chunks and figures into Qdrant.

    uv run python -m rag.embedding.run --texts               # all text chunks (~15 min)
    uv run python -m rag.embedding.run --figures             # captioned figures (~20 min)
    uv run python -m rag.embedding.run --texts --only 02 --limit 20   # quick test

Resumable: points already in Qdrant are skipped.
"""

import argparse
import glob
import json
import time

from rag.config import CHUNKS_DIR, PROCESSED_DIR, TITLES

from .gemma import GemmaEmbeddings, doc_prompt
from .store import COLLECTION, bm25_text, ensure_collection, existing_ids, get_client, point_id, upsert


def load_chunks(only: str | None) -> list[dict]:
    files = sorted(glob.glob(str(CHUNKS_DIR / "*.jsonl")))
    return [json.loads(l) for f in files if not only or f.split("/")[-1].startswith(only) for l in open(f)]


def load_figures(only: str | None, captioned_only: bool) -> list[dict]:
    figs = []
    for f in sorted(glob.glob(str(PROCESSED_DIR / "*/pages.jsonl"))):
        if only and not f.split("/")[-2].startswith(only):
            continue
        for line in open(f):
            page = json.loads(line)
            for fig in page["figures"]:
                if captioned_only and not fig["caption"].lower().startswith("figure"):
                    continue
                figs.append({**fig, "pdf_id": page["pdf_id"], "filename": page["filename"]})
    return figs


def _batches(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def embed_texts(emb: GemmaEmbeddings, client, chunks: list[dict], batch: int) -> int:
    done = 0
    t0 = time.time()
    for group in _batches(chunks, batch):
        ids = [point_id(c["chunk_id"]) for c in group]
        have = existing_ids(client, ids)
        todo = [(i, c) for i, c in zip(ids, group) if i not in have]
        if todo:
            # title + section give each chunk context, e.g. "Pilot's Handbook ... > Oxygen Systems"
            inputs = [doc_prompt(c["text"], f"{c['title']} > {c['section']}" if c["section"] else c["title"])
                      for _, c in todo]
            vectors = emb.embed_documents(inputs)
            ensure_collection(client, len(vectors[0]))
            payloads = [{**c, "type": "text"} for _, c in todo]
            upsert(client, [i for i, _ in todo], vectors, [bm25_text(p) for p in payloads], payloads)
        done += len(group)
        rate = done / (time.time() - t0)
        print(f"texts {done}/{len(chunks)} | {rate:.1f}/s | eta {(len(chunks) - done) / rate / 60:.1f} min", flush=True)
    return done


def embed_figures(emb: GemmaEmbeddings, client, figs: list[dict], batch: int) -> int:
    done = 0
    t0 = time.time()
    for group in _batches(figs, batch):
        ids = [point_id(f["fig_id"]) for f in group]
        have = existing_ids(client, ids)
        todo = [(i, f) for i, f in zip(ids, group) if i not in have]
        if todo:
            vectors = emb.embed_images([f["path"] for _, f in todo])
            ensure_collection(client, len(vectors[0]))
            payloads = [{**f, "type": "figure", "title": TITLES.get(f["pdf_id"], f["pdf_id"]),
                         "text": f["caption"]} for _, f in todo]
            upsert(client, [i for i, _ in todo], vectors, [bm25_text(p) for p in payloads], payloads)
        done += len(group)
        rate = done / (time.time() - t0)
        print(f"figures {done}/{len(figs)} | {rate:.2f}/s | eta {(len(figs) - done) / rate / 60:.1f} min", flush=True)
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--all-figures", action="store_true", help="include uncaptioned figures")
    ap.add_argument("--only", help="book prefix, e.g. 08")
    ap.add_argument("--limit", type=int, help="max items (testing)")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    if not (args.texts or args.figures):
        ap.error("choose --texts and/or --figures")

    emb, client = GemmaEmbeddings(), get_client()
    print(f"collection={COLLECTION} model={emb.model}")
    if args.texts:
        chunks = load_chunks(args.only)[: args.limit]
        embed_texts(emb, client, chunks, args.batch)
    if args.figures:
        figs = load_figures(args.only, captioned_only=not args.all_figures)[: args.limit]
        embed_figures(emb, client, figs, args.batch)
    print("collection size:", client.count(COLLECTION).count)


if __name__ == "__main__":
    main()
