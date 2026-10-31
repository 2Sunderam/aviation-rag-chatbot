# Ingestion: PDF -> per-page markdown + figures

Code: `rag/ingestion/` · Run: `uv run python -m rag.ingestion.run [--only 08] [--max-pages 30] [--workers 2] [--force]`

## Steps (per book, `extract.process_pdf`)

1. **Fast text pass.** `pymupdf4llm.to_markdown(page_chunks=True, use_ocr=NEVER, header=False, footer=False)`
   reads the PDF's own text layer and returns markdown per page with its page number.
   The layout model labels regions (heading, text, table, picture, caption, page-header, page-footer);
   headers and footers are dropped. ~5 pages/s on an M1.
2. **Quality check per page** (`quality.py`):
   - word count and share of real English words (macOS dictionary `/usr/share/dict/words`)
   - old OCR layer present? (invisible text, PDF render mode 3)
   - a page is sent to OCR if it **has an old OCR layer and < 85 % real words** (bad scan text),
     or **has < 15 words and contains an image** (image-only page).
3. **OCR only for those pages.** `to_markdown(..., use_ocr=FORCE_DROP_OLD)` with **RapidOCR**.
   The OCR text is kept only if it is better (more words or higher real-word ratio).
   OCR errors (e.g. an undecodable JPX image) are logged and the page keeps its native text.
4. **Cleanup** (`clean.py`): NFKC unicode, join hyphenated line breaks, drop stray page numbers
   ("3-12") and picture-text markers, collapse whitespace, language detection (pages with >= 50 words).
5. **Figures.** Layout `picture` regions are grouped by nearest caption (one figure is often split
   into several boxes), regions smaller than 120 pt (~1.7 in) are skipped, the rest are cropped to PNG
   at 110 dpi, de-duplicated by hash. Scanned books have no caption boxes, so the caption falls back
   to the "Figure x.y ..." line on the page.

## Output

`data/processed/<book>/pages.jsonl`, one line per page:

```json
{"pdf_id": "02_glider_flying_handbook", "filename": "02_glider_flying_handbook.pdf", "page": 40,
 "text": "<page markdown>", "n_words": 215, "real_word_ratio": 0.907, "ocr": false, "lang": "en",
 "figures": [{"fig_id": "02_glider_flying_handbook_p40_f0", "page": 40, "path": ".../figures/....png",
              "caption": "Figure 4-2. Pressure inside the diaphragm as a function of airspeed.",
              "width": 398, "height": 254}]}
```

The file is written to a temp name and renamed at the end: a book is either complete or absent,
so a crashed run resumes at the next unfinished book.

## Why PyMuPDF and not Docling

Docling's layout model runs on every page (~1 page/s, ~1 h for the corpus) and needs PyTorch.
pymupdf4llm gives the same per-page markdown, headers/footers removal and figure regions at ~5 pages/s,
and OCR runs only where needed. Full run: ~20 min with 2-4 workers.
