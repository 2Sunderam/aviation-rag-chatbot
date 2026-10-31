"""Chunk every extracted book.

    uv run python -m rag.chunking.run            # all books with pages.jsonl
    uv run python -m rag.chunking.run --only 02  # one book

Writes data/chunks/<pdf_id>.jsonl. Fast (seconds), single process.
"""

import argparse
import statistics

from rag.config import CHUNKS_DIR, PROCESSED_DIR

from .chunker import chunk_book, write_chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="book prefix, e.g. 08")
    args = ap.parse_args()

    books = sorted(PROCESSED_DIR.glob("*/pages.jsonl"))
    if args.only:
        books = [b for b in books if b.parent.name.startswith(args.only)]

    total, sizes = 0, []
    for pages_jsonl in books:
        chunks = chunk_book(pages_jsonl)
        write_chunks(chunks, CHUNKS_DIR / f"{pages_jsonl.parent.name}.jsonl")
        cross = sum(c.page != c.page_end for c in chunks)
        print(f"{pages_jsonl.parent.name:45s} chunks={len(chunks):5d} cross-page={cross}")
        total += len(chunks)
        sizes += [c.n_tokens_est for c in chunks]

    if sizes:
        print(f"TOTAL chunks={total} | est tokens/chunk: median={statistics.median(sizes):.0f} "
              f"min={min(sizes)} max={max(sizes)}")


if __name__ == "__main__":
    main()
