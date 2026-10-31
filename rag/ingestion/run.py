"""Run extraction for all PDFs.

    uv run python -m rag.ingestion.run                 # all books, 4 in parallel
    uv run python -m rag.ingestion.run --only 08 --max-pages 30   # quick test

Books that already have pages.jsonl are skipped (resume after a crash).
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from rag.config import PDF_DIR, PROCESSED_DIR

from .extract import process_pdf


def _job(pdf: str, max_pages: int | None) -> dict:
    t = time.time()
    stats = process_pdf(Path(pdf), PROCESSED_DIR / Path(pdf).stem, max_pages)
    stats["seconds"] = round(time.time() - t, 1)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="process PDFs whose name starts with this prefix")
    ap.add_argument("--max-pages", type=int, help="limit pages per PDF (testing)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="re-process finished books")
    args = ap.parse_args()

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if args.only:
        pdfs = [p for p in pdfs if p.name.startswith(args.only)]
    todo = [p for p in pdfs if args.force or not (PROCESSED_DIR / p.stem / "pages.jsonl").exists()]
    print(f"{len(todo)} of {len(pdfs)} PDFs to process")

    t0, results = time.time(), []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_job, str(p), args.max_pages): p for p in todo}
        for fut in as_completed(futures):
            try:
                stats = fut.result()
                results.append(stats)
                print(json.dumps(stats), flush=True)
            except Exception as e:  # one bad book must not kill the run
                print(f"FAILED {futures[fut].name}: {e!r}", flush=True)

    print(f"done in {time.time() - t0:.0f}s | pages={sum(r['pages'] for r in results)} "
          f"ocr={sum(r['ocr_pages'] for r in results)} figures={sum(r['figures'] for r in results)}")


if __name__ == "__main__":
    main()
