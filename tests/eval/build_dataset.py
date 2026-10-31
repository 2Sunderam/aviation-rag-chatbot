"""Build the frozen evaluation set: tests/eval/dataset/qa.jsonl

    uv run python -m tests.eval.build_dataset            # builds once, refuses to overwrite
    uv run python -m tests.eval.build_dataset --force    # regenerate on purpose

Three kinds of rows (all with gold = the chunks that should be retrieved):
    single        80  one question per sampled chunk, ~7 per book (fixed seed)
    multi         10  comparison question over two related chunks from different books
    unanswerable  10  hand-written questions the handbooks cannot answer (gold = none)

Questions are written by Groq models (NOT Mercury 3, which answers in the app) to avoid a
model grading its own questions. Groq free tier = 8k tokens/min per model, so calls rotate
across three models and only wait when all are rate-limited. Progress is saved after every
row, so an interrupted run resumes.
"""

import argparse
import json
import random
import re
import time
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from rag.config import CHUNKS_DIR
from rag.embedding.store import COLLECTION, get_client, point_id
from qdrant_client import models

load_dotenv()

OUT_DIR = Path(__file__).parent / "dataset"
OUT = OUT_DIR / "qa.jsonl"
PARTIAL = OUT_DIR / "qa.partial.jsonl"
SEED = 42
N_SINGLE, N_MULTI = 80, 10
GEN_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
SKIP_SECTIONS = re.compile(r"glossary|chapter summary|preface|acknowledg|end of work package|contents|index|\bsee\b",
                           re.IGNORECASE)
VAGUE = re.compile(r"\b(this|the) (passage|text|section|excerpt|chunk|document|figure above)\b", re.IGNORECASE)

UNANSWERABLE = [
    "What is the capital of France?",
    "What is the takeoff checklist for a Boeing 787 Dreamliner?",
    "How do I program the flight management computer on an Airbus A320?",
    "What is the current price of a private pilot checkride in the United States?",
    "Who won the 2022 FIFA World Cup?",
    "What is the maximum cruise speed of the Cessna Citation Longitude business jet?",
    "How do I bake sourdough bread at high altitude?",
    "What are the EASA licensing requirements for a commercial pilot in Germany?",
    "Which airline has the best on-time performance this year?",
    "How do I reset the cabin pressurization controller on a Bombardier CRJ900?",
]


class QAItem(BaseModel):
    question: str = Field(description="A specific question a student pilot might ask, answerable from the excerpt")
    answer: str = Field(description="Short reference answer (1-3 sentences) using only the excerpt")
    answerable: bool = Field(description="False if the excerpt has no clear, specific fact worth asking about")


SINGLE_PROMPT = """You write evaluation questions for a search system over FAA aviation handbooks.

Excerpt (from "{title}", section "{section}"):
\"\"\"
{text}
\"\"\"

Write ONE question that this excerpt clearly answers.
- Ask about a specific fact, procedure, cause, or definition in the excerpt.
- Paraphrase: do NOT copy long phrases from the excerpt (the search must find it by meaning).
- The question must stand alone: never say "this passage", "the text", "the section", or "the figure".
- If the excerpt is only a list of references, a table of contents, or has no clear fact, set answerable=false."""

MULTI_PROMPT = """You write evaluation questions for a search system over FAA aviation handbooks.

Excerpt A (from "{title_a}", section "{section_a}"):
\"\"\"
{text_a}
\"\"\"

Excerpt B (from "{title_b}", section "{section_b}"):
\"\"\"
{text_b}
\"\"\"

Write ONE comparison or two-part question that needs BOTH excerpts to answer fully
(e.g. "How does X differ between airplanes and helicopters?" or "What is X and how does it relate to Y?").
- Paraphrase; do not copy phrases. Never say "excerpt", "passage" or "text".
- If the two excerpts have nothing meaningful in common, set answerable=false."""


class RotatingGroq:
    """Structured-output calls rotated across Groq models; switches model on 429, waits only if all are limited."""

    def __init__(self):
        self.llms = [(m, ChatGroq(model=m, temperature=0.3, max_tokens=1200, max_retries=0,
                                  **({"reasoning_effort": "low", "reasoning_format": "hidden"} if m.startswith("openai/")
                                     else {"reasoning_format": "hidden"})).with_structured_output(QAItem, method="function_calling"))
                     for m in GEN_MODELS]
        self.i = 0

    def __call__(self, prompt: str) -> tuple[QAItem | None, str]:
        for attempt in range(len(self.llms) * 6):
            name, llm = self.llms[self.i % len(self.llms)]
            try:
                return llm.invoke(prompt), name
            except Exception as e:
                msg = str(e)
                self.i += 1  # next model
                if "429" in msg or "rate" in msg.lower():
                    if attempt % len(self.llms) == len(self.llms) - 1:
                        time.sleep(15)  # every model limited: wait for the per-minute window
                    continue
                if attempt >= len(self.llms):  # non-rate-limit error on several models: give up on this item
                    print(f"   generation failed: {msg[:120]}")
                    return None, name
        return None, ""


def load_chunks() -> list[dict]:
    return [json.loads(l) for f in sorted(CHUNKS_DIR.glob("*.jsonl")) for l in open(f)]


def eligible(c: dict) -> bool:
    return c["n_chars"] >= 1200 and not SKIP_SECTIONS.search(c["section"] or "") and c["lang"] in ("en", "unknown")


def sample_single(chunks: list[dict], rng: random.Random) -> list[dict]:
    by_book: dict[str, list[dict]] = {}
    for c in chunks:
        if eligible(c):
            by_book.setdefault(c["pdf_id"], []).append(c)
    books = sorted(by_book)
    per_book = {b: N_SINGLE // len(books) for b in books}
    for b in books[: N_SINGLE - sum(per_book.values())]:
        per_book[b] += 1
    # oversample x2 so rows rejected by the quality filter can be replaced
    return [c for b in books for c in rng.sample(by_book[b], min(len(by_book[b]), per_book[b] * 2))], per_book


def related_pair(client, chunk: dict) -> dict | None:
    """Nearest text chunk from a DIFFERENT book (dense similarity), for comparison questions."""
    hits = client.query_points(
        COLLECTION, query=point_id(chunk["chunk_id"]), using="dense", limit=10, with_payload=True,
        query_filter=models.Filter(
            must=[models.FieldCondition(key="type", match=models.MatchValue(value="text"))],
            must_not=[models.FieldCondition(key="pdf_id", match=models.MatchValue(value=chunk["pdf_id"]))]),
    ).points
    good = [h for h in hits if 0.70 <= h.score <= 0.95 and eligible(h.payload)]
    return good[0].payload if good else None


def gold(c: dict) -> dict:
    return {"chunk_id": c["chunk_id"], "pdf_id": c["pdf_id"], "page": c["page"], "page_end": c["page_end"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="regenerate even if qa.jsonl exists")
    args = ap.parse_args()
    if OUT.exists() and not args.force:
        raise SystemExit(f"{OUT} exists (frozen). Use --force to regenerate on purpose.")
    if args.force:
        PARTIAL.unlink(missing_ok=True)

    rng = random.Random(SEED)
    chunks = load_chunks()
    gen = RotatingGroq()
    done = [json.loads(l) for l in PARTIAL.open()] if PARTIAL.exists() else []
    seen = {r["id"] for r in done}
    out = PARTIAL.open("a")

    def save(row: dict) -> None:
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        out.flush()
        done.append(row)
        seen.add(row["id"])

    # --- single-chunk questions ---
    candidates, per_book = sample_single(chunks, rng)
    have = {b: sum(1 for r in done if r["type"] == "single" and r["gold"][0]["pdf_id"] == b) for b in per_book}
    for c in candidates:
        rid = f"single:{c['chunk_id']}"
        if rid in seen or have[c["pdf_id"]] >= per_book[c["pdf_id"]]:
            continue
        item, model = gen(SINGLE_PROMPT.format(title=c["title"], section=c["section"], text=c["text"]))
        if not item or not item.answerable or VAGUE.search(item.question) or len(item.question) < 15:
            continue
        save({"id": rid, "type": "single", "question": item.question.strip(), "reference_answer": item.answer.strip(),
              "gold": [gold(c)], "generator": model})
        have[c["pdf_id"]] += 1
        print(f"single {sum(have.values())}/{N_SINGLE} [{model}] {item.question[:90]}", flush=True)

    # --- multi-part questions over two related chunks from different books ---
    client = get_client()
    pool = [c for c in chunks if eligible(c)]
    rng.shuffle(pool)
    n_multi = sum(1 for r in done if r["type"] == "multi")
    for a in pool:
        if n_multi >= N_MULTI:
            break
        rid = f"multi:{a['chunk_id']}"
        if rid in seen:
            continue
        b = related_pair(client, a)
        if not b:
            continue
        item, model = gen(MULTI_PROMPT.format(title_a=a["title"], section_a=a["section"], text_a=a["text"],
                                              title_b=b["title"], section_b=b["section"], text_b=b["text"]))
        if not item or not item.answerable or VAGUE.search(item.question):
            continue
        save({"id": rid, "type": "multi", "question": item.question.strip(), "reference_answer": item.answer.strip(),
              "gold": [gold(a), gold(b)], "generator": model})
        n_multi += 1
        print(f"multi {n_multi}/{N_MULTI} [{model}] {item.question[:90]}", flush=True)

    # --- unanswerable (hand-written) ---
    for i, q in enumerate(UNANSWERABLE):
        if f"unanswerable:{i}" not in seen:
            save({"id": f"unanswerable:{i}", "type": "unanswerable", "question": q,
                  "reference_answer": "I couldn't find this in the handbooks.", "gold": [], "generator": "hand-written"})

    out.close()
    PARTIAL.rename(OUT)  # freeze
    counts = {t: sum(r["type"] == t for r in done) for t in ("single", "multi", "unanswerable")}
    (OUT_DIR / "meta.json").write_text(json.dumps({
        "created": date.today().isoformat(), "seed": SEED, "generator_models": GEN_MODELS,
        "counts": counts, "relevance_rule": "retrieved chunk is relevant if same pdf_id and page range overlaps a gold chunk",
    }, indent=2))
    print(f"frozen -> {OUT} {counts}")


if __name__ == "__main__":
    main()
