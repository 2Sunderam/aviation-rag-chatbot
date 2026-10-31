"""End-to-end latency on the frozen dataset through the full chat pipeline.

    uv run python -m tests.eval.eval_latency            # all questions
    uv run python -m tests.eval.eval_latency --n 30     # first 30

Each question: planner -> embed -> Qdrant hybrid search -> Mercury 3 answer (fresh session, sequential).
Reports p50 / p95 / max per stage and checks the 2-5 s target (p95 <= 5 s). One warm-up call first.
"""

import argparse
import asyncio

from rag.generation.chat import ChatEngine

from .common import load_dataset, mean, pct, write_result

STAGES = ["plan", "embed", "search_text", "search_figures", "llm_first_token", "llm_total", "end_to_end"]
TARGET_P95_MS = 5000


async def run_one(engine: ChatEngine, q: str, sid: str) -> dict:
    timings = {}
    async for ev in engine.stream(sid, q):
        if ev["type"] == "done":
            timings = ev["timings_ms"]
    return timings


async def main_async(n: int | None) -> None:
    rows = load_dataset()[:n] if n else load_dataset()
    engine = ChatEngine()
    await run_one(engine, "What is a stall?", "warmup")

    results = []
    for i, r in enumerate(rows, 1):
        t = await run_one(engine, r["question"], f"lat-{i}")
        results.append({"id": r["id"], "type": r["type"], **{s: t.get(s) or 0 for s in STAGES}})
        if i % 10 == 0:
            print(f"{i}/{len(rows)}  last end-to-end {t['end_to_end']} ms", flush=True)

    stats = {s: {"p50": pct([x[s] for x in results], 50), "p95": pct([x[s] for x in results], 95),
                 "max": max(x[s] for x in results), "mean": round(mean([x[s] for x in results]))} for s in STAGES}
    e2e = [x["end_to_end"] for x in results]
    within = sum(2000 <= v <= 5000 for v in e2e) / len(e2e)
    under5 = sum(v <= 5000 for v in e2e) / len(e2e)
    passed = stats["end_to_end"]["p95"] <= TARGET_P95_MS

    lines = [f"# Latency eval ({len(results)} questions, sequential, Mercury 3 answers)\n",
             "| stage | p50 ms | p95 ms | max ms | mean ms |", "|---|---|---|---|---|"]
    lines += [f"| {s} | {v['p50']} | {v['p95']} | {v['max']} | {v['mean']} |" for s, v in stats.items()]
    lines += [f"\n- **p95 end-to-end: {stats['end_to_end']['p95'] / 1000:.2f} s** (target <= 5 s): "
              f"{'PASS' if passed else 'FAIL'}",
              f"- answers within 5 s: {under5:.0%} | within the 2-5 s band: {within:.0%} (faster than 2 s counts as fine)"]
    md = "\n".join(lines)
    path = write_result("latency", {"stats": stats, "pass": passed, "per_question": results}, md)
    print(md)
    print(f"\nsaved {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, help="only the first n questions")
    asyncio.run(main_async(ap.parse_args().n))


if __name__ == "__main__":
    main()
