# Chunking: pages -> chunks

Code: `rag/chunking/chunker.py` · Run: `uv run python -m rag.chunking.run [--only 08]` (seconds)

## Strategy: structure-aware recursive chunking

- **Splitter:** LangChain `RecursiveCharacterTextSplitter.from_language(Language.MARKDOWN)`.
  Splits at markdown headings first, then paragraphs, lines, words, so sections stay intact.
- **Size:** 3,200 characters (~800 tokens at ~4 chars/token, measured 4.1 with the Gemma tokenizer),
  **15 % overlap** (480 characters). Median chunk ~650 tokens.
- **Whole book at once:** pages are joined so sections and overlap can cross page breaks.
  Character offsets map every chunk back to its start page (`page`) and end page (`page_end`).
- **Section:** nearest markdown heading above the chunk.
- **Deterministic:** same input -> same chunks and ids.

## Cleanup and filters (applied at chunk time)

| Problem | Fix |
|---|---|
| Ligatures extracted with a space ("fl ightpath", "suffi cient") | rejoin if either half is not a dictionary word ("off the" stays) |
| HTML tags (`<mark>`, `<sup>`) | stripped |
| Table-of-contents dot leaders ("Release ......12-7") | collapsed |
| TOC / index chunks | dropped (> 50 % of lines end in a page reference, or section is "Contents"/"Index") |
| Chart / screenshot label noise | dropped if < 50 % real words |
| Near-empty chunks | dropped if < 200 characters |

## Output

`data/chunks/<book>.jsonl`, one line per chunk:

```json
{"chunk_id": "08_pilots_handbook_aeronautical_knowledge_c00300", "pdf_id": "08_pilots_handbook_aeronautical_knowledge",
 "filename": "08_pilots_handbook_aeronautical_knowledge.pdf", "title": "Pilot's Handbook of Aeronautical Knowledge",
 "page": 197, "page_end": 197, "section": "Oxygen Systems", "text": "...", "n_chars": 1152,
 "n_tokens_est": 288, "ocr": false, "lang": "en"}
```

Total: 4,106 chunks over 11 books.
