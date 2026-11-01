"""Shared helpers for the evals: load the frozen dataset, relevance rule, result writing."""

import json
import statistics
from datetime import datetime
from pathlib import Path

EVAL_DIR = Path(__file__).parent
DATASET = EVAL_DIR / "dataset" / "qa.jsonl"
RESULTS = EVAL_DIR / "results"


def load_dataset(types: tuple[str, ...] | None = None) -> list[dict]:
    if not DATASET.exists():
        raise SystemExit(f"{DATASET} missing: run `uv run python -m tests.eval.build_dataset` first")
    rows = [json.loads(l) for l in DATASET.open()]
    return [r for r in rows if not types or r["type"] in types]


def is_relevant(payload: dict, gold: dict) -> bool:
    """Same book and overlapping pages. Chunks overlap by 15%, so a neighbour chunk on the
    same page carries the same facts; an exact chunk-id match would be unfairly strict."""
    return (payload["pdf_id"] == gold["pdf_id"]
            and payload["page"] <= gold["page_end"] and gold["page"] <= payload.get("page_end", payload["page"]))


def pct(values: list[float], p: float) -> float:
    """p-th percentile (nearest rank): the value that p% of the values are at or below.
    Used for latency: pct(times, 50) = median (p50), pct(times, 95) = slow-tail p95.
    e.g. 20 timings, p=95 -> index round(0.95 * 19) = 18 -> 2nd slowest."""
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, max(0, round(p / 100 * (len(s) - 1))))]


def mean(values: list[float]) -> float:
    """Average, 0.0 for an empty list instead of raising. Averaging 0/1 flags gives a rate,
    e.g. mean([1, 0, 1, 1]) = 0.75 = 75% of questions refused / cited correctly."""
    return statistics.fmean(values) if values else 0.0


def write_result(name: str, data: dict, markdown: str) -> Path:
    """Save full numbers to results/<name>_<timestamp>.json (history, never overwritten)
    and a readable summary to results/<name>_latest.md (overwritten each run)."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = RESULTS / f"{name}_{stamp}.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    (RESULTS / f"{name}_latest.md").write_text(markdown)
    return path
