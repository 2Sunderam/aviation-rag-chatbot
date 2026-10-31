"""Answer quality on the frozen dataset: Ragas + citation accuracy + refusals (LLM as judge = Mercury 3).

    uv run --group eval python -m tests.eval.eval_generation            # all rows
    uv run --group eval python -m tests.eval.eval_generation --n 20     # quick run

1. Every question goes through the real pipeline (planner -> hybrid retrieval -> Mercury 3 answer).
2. Ragas on answered, answerable rows:
       faithfulness        claims in the answer supported by the retrieved chunks (hallucination check)
       answer_relevancy    does the answer address the question
       context_precision   are the retrieved chunks useful for the reference answer
       context_recall      do the retrieved chunks contain what the reference answer needs
   Hallucination rate = share of answers with faithfulness < 1.0 (at least one unsupported claim).
3. Citation accuracy (custom judge): for each sentence with [n], does chunk n support it?
   Also: invalid citation rate (cited numbers that do not exist).
4. Refusals: unanswerable rows should say "I couldn't find this"; answerable rows should not.
"""

import argparse
import asyncio
import os
import re

from . import ragas_compat  # noqa: F401  (must come before ragas)

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy

from rag.embedding.gemma import GemmaEmbeddings
from rag.generation.chat import INCEPTION_URL, NOT_FOUND, ChatEngine

from .common import load_dataset, mean, write_result

_CITE = re.compile(r"\[(F?\d+)\]")
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")


def judge_llm(max_tokens: int = 4000) -> ChatOpenAI:
    return ChatOpenAI(model="mercury-3", base_url=INCEPTION_URL, api_key=os.getenv("INCEPTION_API_KEY"),
                      temperature=0, max_tokens=max_tokens, timeout=60, max_retries=2,
                      extra_body={"reasoning_effort": "low"})


class Verdict(BaseModel):
    claim_index: int
    supported: bool = Field(description="True only if the source text directly supports the claim")


class Verdicts(BaseModel):
    verdicts: list[Verdict]


CITE_JUDGE = """For each numbered claim, decide if its cited SOURCE text directly supports it.
Be strict: partial or unrelated support = false.

{pairs}

Return one verdict per claim_index."""


async def run_pipeline(engine: ChatEngine, row: dict, i: int) -> dict:
    answer, chunks, sources, invalid = "", [], [], []
    async for ev in engine.stream(f"gen-{i}", row["question"]):
        if ev["type"] == "retrieval":
            chunks = ev["chunks"]
        elif ev["type"] == "token":
            answer += ev["text"]
        elif ev["type"] == "done":
            sources, invalid = ev["sources"], ev["invalid_citations"]
    return {"row": row, "answer": answer.strip(), "chunks": chunks, "sources": sources, "invalid": invalid,
            "refused": answer.strip().startswith(NOT_FOUND[:25])}


JUDGE_CONCURRENCY = 2  # Mercury rate-limits per minute (~100 req/min); window clears in ~20 s


async def judge_citations(judge, res: dict, sem: asyncio.Semaphore) -> list[bool]:
    """One judge call per answer over all (sentence, cited chunk) pairs."""
    pairs, n = [], 0
    for sent in _SENT.split(res["answer"]):
        for ref in dict.fromkeys(_CITE.findall(sent)):
            if ref.startswith("F") or not (1 <= int(ref) <= len(res["chunks"])):
                continue
            claim = _CITE.sub("", sent).strip()
            src = res["chunks"][int(ref) - 1].payload["text"][:2500]
            pairs.append(f"[{n}] CLAIM: {claim}\nSOURCE [{ref}]: {src}")
            n += 1
    if not pairs:
        return []
    for attempt in range(6):
        try:
            async with sem:
                out: Verdicts = await judge.with_structured_output(Verdicts, method="function_calling").ainvoke(
                    CITE_JUDGE.format(pairs="\n\n".join(pairs)))
            got = {v.claim_index: v.supported for v in out.verdicts}
            return [got.get(i, False) for i in range(n)]
        except Exception as e:
            if "429" in str(e) and attempt < 5:
                await asyncio.sleep(min(30 * (attempt + 1), 90))  # back off past the rate window, then retry
                continue
            print(f"   citation judge failed: {str(e)[:100]}")
            return []
    return []


async def main_async(n: int | None, skip_ragas: bool = False) -> None:
    rows = load_dataset()
    rows = rows[:n] if n else rows
    engine = ChatEngine()

    print(f"1/3 answering {len(rows)} questions through the pipeline...", flush=True)
    results = []
    for i, r in enumerate(rows, 1):
        results.append(await run_pipeline(engine, r, i))
        if i % 10 == 0:
            print(f"   {i}/{len(rows)}", flush=True)

    answerable = [x for x in results if x["row"]["type"] != "unanswerable"]
    answered = [x for x in answerable if not x["refused"]]
    unanswerable = [x for x in results if x["row"]["type"] == "unanswerable"]

    df, metric_cols, faith = None, [], []
    if not skip_ragas:
        print(f"2/3 ragas on {len(answered)} answered questions (judge: mercury-3)...", flush=True)
        ds = EvaluationDataset.from_list([{
            "user_input": x["row"]["question"], "response": x["answer"],
            "retrieved_contexts": [c.payload["text"] for c in x["chunks"]], "reference": x["row"]["reference_answer"],
        } for x in answered])
        llm = LangchainLLMWrapper(judge_llm())
        emb = LangchainEmbeddingsWrapper(GemmaEmbeddings())
        ragas_res = evaluate(ds, metrics=[Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(),
                                          LLMContextRecall()],
                             llm=llm, embeddings=emb, run_config=RunConfig(max_workers=4, timeout=120),
                             show_progress=True)
        df = ragas_res.to_pandas()
        metric_cols = [c for c in df.columns if c in ("faithfulness", "answer_relevancy", "llm_context_precision_with_reference",
                                                       "context_recall")]

    print("3/3 citation judge...", flush=True)
    judge = judge_llm(3000)
    sem = asyncio.Semaphore(JUDGE_CONCURRENCY)
    cite_verdicts = await asyncio.gather(*[judge_citations(judge, x, sem) for x in answered])
    judged_answers = sum(bool(v) for v, x in zip(cite_verdicts, answered, strict=True) if x["sources"])

    if df is not None:
        faith = [v for v in df["faithfulness"].tolist() if v == v]  # drop NaN
    all_cites = [v for vs in cite_verdicts for v in vs]
    n_cited = sum(len(x["sources"]) for x in answered)

    summary = {
        "n_questions": len(results),
        "ragas": {c: round(float(df[c].mean()), 3) for c in metric_cols} if df is not None else "skipped",
        "hallucination_rate": round(sum(v < 1.0 for v in faith) / len(faith), 3) if faith else None,
        "severe_hallucination_rate": round(sum(v < 0.5 for v in faith) / len(faith), 3) if faith else None,
        "citation_accuracy": round(mean([float(v) for v in all_cites]), 3),
        "citation_pairs_judged": len(all_cites),
        "answers_citation_judged": f"{judged_answers}/{sum(bool(x['sources']) for x in answered)}",
        "invalid_citation_rate": round(sum(len(x["invalid"]) for x in answered) / max(n_cited + sum(len(x["invalid"]) for x in answered), 1), 3),
        "answers_without_citation": round(sum(not x["sources"] for x in answered) / max(len(answered), 1), 3),
        "unanswerable_correctly_refused": round(mean([float(x["refused"]) for x in unanswerable]), 3) if unanswerable else None,
        "answerable_wrongly_refused": round(mean([float(x["refused"]) for x in answerable]), 3),
    }

    md = "\n".join([
        f"# Generation eval ({len(results)} questions, judge: mercury-3)\n",
        "| metric | value | meaning |", "|---|---|---|",
        *([f"| ragas {k} | {v:.3f} | |" for k, v in summary["ragas"].items()] if df is not None else ["| ragas | skipped | |"]),
        f"| hallucination rate | {summary['hallucination_rate']} | answers with >= 1 unsupported claim (faithfulness < 1) |",
        f"| severe hallucination rate | {summary['severe_hallucination_rate']} | faithfulness < 0.5 |",
        f"| citation accuracy | {summary['citation_accuracy']} | cited chunk supports the sentence ({len(all_cites)} pairs, {summary['answers_citation_judged']} answers) |",
        f"| invalid citation rate | {summary['invalid_citation_rate']} | cited numbers that do not exist |",
        f"| answers without citation | {summary['answers_without_citation']} | |",
        f"| unanswerable correctly refused | {summary['unanswerable_correctly_refused']} | |",
        f"| answerable wrongly refused | {summary['answerable_wrongly_refused']} | |",
    ])
    per_q = [{"id": x["row"]["id"], "question": x["row"]["question"], "answer": x["answer"], "refused": x["refused"],
              "sources": x["sources"], "invalid": x["invalid"]} for x in results]
    path = write_result("generation" if df is not None else "citations", {"summary": summary,
                        "ragas_rows": df.to_dict(orient="records") if df is not None else [],
                                       "per_question": per_q}, md)
    print(md)
    print(f"\nsaved {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, help="only the first n rows")
    ap.add_argument("--skip-ragas", action="store_true", help="only citation accuracy + refusals (~3 min)")
    a = ap.parse_args()
    asyncio.run(main_async(a.n, a.skip_ragas))


if __name__ == "__main__":
    main()
