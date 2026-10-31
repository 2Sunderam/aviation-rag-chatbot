"""RAG chat: query planner -> hybrid retrieval -> Mercury 3 answer (LangChain, Groq as backup).

Flow per message:
    1. memory: last 6 messages of this session (sliding window, langchain trim_messages)
    2. planner: question + memory -> SearchPlan{queries: 1..3} (structured output, validated by Pydantic)
       resolves follow-ups ("it" -> "carburetor icing") and splits multi-part questions
    3. retrieval: ONE Qdrant call, dense + BM25 prefetch per sub-query, fused with RRF
    4. answer: original question + memory + numbered sources [1..k], [F1..], streamed
    5. citations: cited numbers -> "Title, p. X" from metadata (never from the LLM)
"""

import asyncio
import os
import re
import time
from collections.abc import AsyncIterator

from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, trim_messages
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from rag.retrieval.hybrid import Hit, HybridSearcher

from .prompts import ANSWER_PROMPT, PLANNER_PROMPT

load_dotenv()

ANSWER_MODEL = os.getenv("ANSWER_MODEL", "mercury-3")  # Inception diffusion LLM, 1000+ tok/s
INCEPTION_URL = "https://api.inceptionlabs.ai/v1"
WINDOW_MESSAGES = 6  # sliding window: last 3 user/assistant turns
K_BY_QUERIES = {1: 5, 2: 7, 3: 8}  # text chunks sent to the LLM, by number of sub-queries
NOT_FOUND = "I couldn't find this in the handbooks."
# Backup only: Groq free tier allows 8,000 tokens/minute PER MODEL, so we chain several.
GROQ_BACKUPS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]

_CITE = re.compile(r"\[(F?\d+)\]")


class SearchPlan(BaseModel):
    """Search queries for the handbook retriever."""

    queries: list[str] = Field(min_length=1, max_length=3,
                               description="1-3 standalone search queries; 1 for simple questions")


def _mercury(max_tokens: int, effort: str, retries: int = 1) -> ChatOpenAI:
    # max_tokens includes hidden reasoning tokens: keep it generous or the visible answer comes back empty
    return ChatOpenAI(model=ANSWER_MODEL, base_url=INCEPTION_URL, api_key=os.getenv("INCEPTION_API_KEY"),
                      temperature=0, max_tokens=max_tokens, max_retries=retries, timeout=30,
                      extra_body={"reasoning_effort": effort})


def _groq(model: str, max_tokens: int, retries: int = 0) -> ChatGroq:
    kw = dict(reasoning_effort="low", reasoning_format="hidden") if model.startswith("openai/") else \
        dict(reasoning_format="hidden")
    return ChatGroq(model=model, temperature=0, max_tokens=max_tokens, max_retries=retries, **kw)


def _with_backups(max_tokens: int, effort: str, structured: type[BaseModel] | None = None):
    """Mercury 3 first; on error fall through the Groq models (no waiting on 429s).

    effort: "instant" = no hidden reasoning (fast, never truncated), "low" = brief reasoning.
    """
    def wrap(llm):
        return llm.with_structured_output(structured, method="function_calling") if structured else llm

    primary = wrap(_mercury(max_tokens, effort))
    backups = [wrap(_groq(m, max_tokens)) for m in GROQ_BACKUPS] if os.getenv("GROQ_API_KEY") else []
    return primary.with_fallbacks(backups) if backups else primary


def format_context(chunks: list[Hit], figures: list[Hit]) -> str:
    parts = [f"[{i}] ({h.citation}; section: {h.payload.get('section', '')})\n{h.payload['text']}"
             for i, h in enumerate(chunks, 1)]
    parts += [f"[F{i}] Figure ({h.citation}): {h.payload.get('caption') or 'no caption'}"
              for i, h in enumerate(figures, 1)]
    return "\n\n".join(parts)


class ChatEngine:
    def __init__(self, searcher: HybridSearcher | None = None):
        self.searcher = searcher or HybridSearcher()
        self.answer_model = ANSWER_MODEL
        # Set ANSWER_REASONING=low for slightly more careful answers (larger token budget, ~0.5 s slower).
        effort = os.getenv("ANSWER_REASONING", "instant")
        self.answer_llm = _with_backups(2500 if effort != "instant" else 1000, effort)
        self.answer_retry_llm = _with_backups(1000, "instant")  # used if the answer comes back empty
        self.planner = PLANNER_PROMPT | _with_backups(300, "instant", structured=SearchPlan)
        self.sessions: dict[str, InMemoryChatMessageHistory] = {}

    def history(self, session_id: str) -> list:
        hist = self.sessions.setdefault(session_id, InMemoryChatMessageHistory())
        return trim_messages(hist.messages, strategy="last", token_counter=len,
                             max_tokens=WINDOW_MESSAGES, start_on="human")

    def reset(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)

    async def plan(self, window: list, question: str) -> list[str]:
        """Question + memory -> 1-3 standalone search queries. On any failure: search the question as-is."""
        convo = "\n".join(f"{'User' if m.type == 'human' else 'Assistant'}: {m.content[:300]}"
                          for m in window) or "(none)"
        try:
            plan: SearchPlan = await self.planner.ainvoke({"conversation": convo, "question": question})
            queries = [q.strip() for q in plan.queries if q.strip()][:3]
        except Exception:
            queries = []
        return queries or [question]

    async def stream(self, session_id: str, question: str, k: int | None = None, fig_k: int = 3,
                     pdf_ids: list[str] | None = None) -> AsyncIterator[dict]:
        t0 = time.perf_counter()
        window = self.history(session_id)

        queries = await self.plan(window, question)
        plan_ms = round((time.perf_counter() - t0) * 1000)

        k = k or K_BY_QUERIES[len(queries)]
        res = await asyncio.to_thread(self.searcher.search, queries, k, fig_k, pdf_ids, True)
        yield {"type": "retrieval", "queries": queries, "timings_ms": res.timings_ms,
               "chunks": res.chunks, "figures": res.figures}

        t_llm = time.perf_counter()
        first_token_ms, answer, model_used = None, "", None
        inputs = {"history": window, "question": question, "context": format_context(res.chunks, res.figures)}
        async for chunk in (ANSWER_PROMPT | self.answer_llm).astream(inputs):
            model_used = chunk.response_metadata.get("model_name") or model_used
            if chunk.content:
                if first_token_ms is None:
                    first_token_ms = round((time.perf_counter() - t_llm) * 1000)
                answer += chunk.content
                yield {"type": "token", "text": chunk.content}

        if not answer.strip():  # reasoning used the whole token budget: retry without reasoning
            msg = await (ANSWER_PROMPT | self.answer_retry_llm).ainvoke(inputs)
            answer, model_used = msg.content, msg.response_metadata.get("model_name", model_used)
            first_token_ms = round((time.perf_counter() - t_llm) * 1000)
            yield {"type": "token", "text": answer}

        hist = self.sessions.setdefault(session_id, InMemoryChatMessageHistory())
        hist.add_messages([HumanMessage(question), AIMessage(answer)])

        cited = list(dict.fromkeys(_CITE.findall(answer)))
        sources = []
        for c in cited:
            pool, idx = (res.figures, int(c[1:])) if c.startswith("F") else (res.chunks, int(c))
            if 1 <= idx <= len(pool):
                sources.append({"ref": c, "citation": pool[idx - 1].citation, "id": pool[idx - 1].id})
        invalid = [c for c in cited if c not in {s["ref"] for s in sources}]

        yield {"type": "done", "sources": sources, "invalid_citations": invalid, "model": model_used,
               "not_found": answer.strip() == NOT_FOUND,
               "timings_ms": {**res.timings_ms, "plan": plan_ms, "llm_first_token": first_token_ms,
                              "llm_total": round((time.perf_counter() - t_llm) * 1000),
                              "end_to_end": round((time.perf_counter() - t0) * 1000)}}
