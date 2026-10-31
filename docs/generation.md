# Generation: planner, memory, answer, citations

Code: `rag/generation/chat.py`, `rag/generation/prompts.py`

## Per message (`ChatEngine.stream`)

1. **Memory.** Sliding window: last 6 messages (3 turns) of the session
   (`InMemoryChatMessageHistory` + LangChain `trim_messages`). In-memory, per `session_id`.
2. **Query planner.** Mercury 3 (`reasoning_effort="instant"`) with LangChain
   `with_structured_output(SearchPlan, method="function_calling")`:

   ```python
   class SearchPlan(BaseModel):
       queries: list[str]   # 1..3, validated by Pydantic
   ```

   The conversation is passed as plain text. The planner resolves follow-ups and splits multi-part questions:
   - "How do pilots fix it?" -> `["how do pilots fix carburetor icing"]`
   - "What is VASI and what is PAPI?" -> `["What is VASI ...", "What is PAPI ..."]`

   On any failure the original question is searched as-is.
3. **Retrieval.** One hybrid call ([retrieval.md](retrieval.md)). Chunks sent to the LLM:
   5 / 7 / 8 for 1 / 2 / 3 queries, plus 3 figures (caption + citation).
4. **Answer.** Mercury 3, streamed. Input: original question + memory + numbered sources `[1..k]`, `[F1..F3]`.
   Prompt rules: use only the sources, cite every fact, mention relevant figures,
   reply exactly "I couldn't find this in the handbooks." if the sources do not contain the answer.
5. **Citations.** Cited numbers are mapped to "Book title, p. X" from the payload, so page numbers
   always come from metadata, never from the LLM. Unknown numbers are reported as invalid.

## Models

| Use | Model | Setting |
|---|---|---|
| Planner | Mercury 3 (Inception, OpenAI-compatible API) | `instant`, 300 tokens |
| Answer | Mercury 3 | `instant` (default) or `low` via `ANSWER_REASONING` |
| Fallback (both) | Groq `gpt-oss-120b` -> `gpt-oss-20b` -> `qwen3.8-27b` | LangChain `with_fallbacks` |

Notes:
- Mercury's `max_tokens` includes hidden reasoning tokens; with `low` the budget is raised to 2,500,
  and an empty answer is retried once with `instant`.
- Groq free tier: 8,000 tokens/min per model, so it is only a fallback.
- Mercury rate limit: roughly 100 requests/min, clears in ~20 s.

## Events yielded (consumed by the API)

| type | content |
|---|---|
| `retrieval` | planned queries, chunks, figures, retrieval timings |
| `token` | answer text pieces |
| `done` | sources, invalid citations, model used, `not_found`, all timings |
