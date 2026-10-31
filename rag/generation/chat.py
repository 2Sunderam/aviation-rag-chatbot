"""RAG chat: hybrid retrieval -> Mercury 3 LLM (LangChain, Groq as backup) with sliding-window memory.

Flow per message:
    1. last N messages of this session (sliding window, langchain trim_messages)
    2. follow-ups only: rewrite into a standalone query
    3. hybrid retrieval (dense + BM25 + RRF in Qdrant): top-k chunks + top figures
    4. stream the answer with numbered sources [1..k], [F1..]
    5. map cited numbers -> "Title, p. X" (provenance comes from metadata, not the LLM)
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

from rag.retrieval.hybrid import Hit, HybridSearcher

from .prompts import ANSWER_PROMPT, REWRITE_PROMPT

load_dotenv()

ANSWER_MODEL = os.getenv("ANSWER_MODEL", "mercury-3")  # Inception diffusion LLM, 1000+ tok/s
INCEPTION_URL = "https://api.inceptionlabs.ai/v1"
WINDOW_MESSAGES = 6  # sliding window: last 3 user/assistant turns
NOT_FOUND = "I couldn't find this in the handbooks."

_CITE = re.compile(r"\[(F?\d+)\]")


# Backup only: Groq free tier allows 8,000 tokens/minute PER MODEL, so we chain several.
GROQ_BACKUPS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]


def _mercury(max_tokens: int, effort: str, retries: int = 1) -> ChatOpenAI:
    # max_tokens includes hidden reasoning tokens: keep it generous or the visible answer comes back empty
    return ChatOpenAI(model=ANSWER_MODEL, base_url=INCEPTION_URL, api_key=os.getenv("INCEPTION_API_KEY"),
                      temperature=0, max_tokens=max_tokens, max_retries=retries, timeout=30,
                      extra_body={"reasoning_effort": effort})


def _groq(model: str, max_tokens: int, retries: int = 0) -> ChatGroq:
    kw = dict(reasoning_effort="low", reasoning_format="hidden") if model.startswith("openai/") else \
        dict(reasoning_format="hidden")
    return ChatGroq(model=model, temperature=0, max_tokens=max_tokens, max_retries=retries, **kw)


def _with_backups(max_tokens: int, effort: str = "low"):
    """Mercury 3 first; if it errors, fall through the Groq models (no waiting on 429s).

    effort: "instant" = no reasoning (~0.5 s, used for query rewrite), "low" = brief reasoning (answers).
    """
    primary = _mercury(max_tokens, effort)
    backups = [_groq(m, max_tokens) for m in GROQ_BACKUPS] if os.getenv("GROQ_API_KEY") else []
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
        # "instant" = no hidden reasoning: fast and never truncated by a reasoning budget.
        # Set ANSWER_REASONING=low for slightly more careful answers (needs a larger token budget).
        effort = os.getenv("ANSWER_REASONING", "instant")
        self.answer_llm = _with_backups(2500 if effort != "instant" else 1000, effort)
        self.answer_retry_llm = _with_backups(1000, "instant")  # used if the answer comes back empty
        self.rewrite_llm = _with_backups(100, "instant")
        self.sessions: dict[str, InMemoryChatMessageHistory] = {}

    def history(self, session_id: str) -> list:
        hist = self.sessions.setdefault(session_id, InMemoryChatMessageHistory())
        return trim_messages(hist.messages, strategy="last", token_counter=len,
                             max_tokens=WINDOW_MESSAGES, start_on="human")

    def reset(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)

    async def _rewrite(self, window: list, question: str) -> str:
        """Standalone search query for a follow-up. Falls back to 'previous question + follow-up'."""
        convo = "\n".join(f"{'User' if m.type == 'human' else 'Assistant'}: {m.content[:300]}" for m in window)
        last_user = next((m.content for m in reversed(window) if m.type == "human"), "")
        try:
            msg = await (REWRITE_PROMPT | self.rewrite_llm).ainvoke({"conversation": convo, "question": question})
            query = msg.content.strip().splitlines()[0].strip(" \"'") if msg.content.strip() else ""
        except Exception:
            query = ""
        if not query or len(query) > 200 or "[" in query:  # empty, or it answered instead of rewriting
            query = f"{last_user} {question}"
        return query

    async def stream(self, session_id: str, question: str, k: int = 5, fig_k: int = 3,
                     pdf_ids: list[str] | None = None) -> AsyncIterator[dict]:
        t0 = time.perf_counter()
        window = self.history(session_id)

        query, rewrite_ms = question, 0
        if window:  # follow-up: make it standalone so retrieval works
            t = time.perf_counter()
            query = await self._rewrite(window, question)
            rewrite_ms = round((time.perf_counter() - t) * 1000)

        res = await asyncio.to_thread(self.searcher.search, query, k, fig_k, 50, pdf_ids, True)
        yield {"type": "retrieval", "query": query, "rewrite_ms": rewrite_ms,
               "timings_ms": res.timings_ms, "chunks": res.chunks, "figures": res.figures}

        t_llm = time.perf_counter()
        first_token_ms, answer, model_used = None, "", None
        chain = ANSWER_PROMPT | self.answer_llm
        async for chunk in chain.astream({"history": window, "question": question,
                                          "context": format_context(res.chunks, res.figures)}):
            model_used = chunk.response_metadata.get("model_name") or model_used
            if chunk.content:
                if first_token_ms is None:
                    first_token_ms = round((time.perf_counter() - t_llm) * 1000)
                answer += chunk.content
                yield {"type": "token", "text": chunk.content}

        if not answer.strip():  # reasoning used the whole token budget: retry without reasoning
            msg = await (ANSWER_PROMPT | self.answer_retry_llm).ainvoke(
                {"history": window, "question": question, "context": format_context(res.chunks, res.figures)})
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
               "timings_ms": {**res.timings_ms, "rewrite": rewrite_ms, "llm_first_token": first_token_ms,
                              "llm_total": round((time.perf_counter() - t_llm) * 1000),
                              "end_to_end": round((time.perf_counter() - t0) * 1000)}}
