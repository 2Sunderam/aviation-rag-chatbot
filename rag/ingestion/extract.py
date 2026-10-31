"""PDF -> per-page markdown (PyMuPDF + pymupdf4llm), OCR only when needed, figure crops.

pymupdf4llm's layout model drops page headers/footers and labels picture/caption
regions. OCR is off for the bulk pass (5 pages/s on M1) and runs per page
(RapidOCR) only for pages that fail the quality check in quality.py.
"""

import hashlib
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pymupdf
import pymupdf4llm
from pymupdf4llm.ocr import OCRMode

from .clean import clean_text, detect_language
from .quality import needs_ocr, word_stats

log = logging.getLogger(__name__)

MD_OPTS = dict(page_chunks=True, header=False, footer=False, show_progress=False)
FIG_MIN_SIDE_PT = 120  # skip icons/logos smaller than ~1.7 inch
FIG_DPI = 110


@dataclass
class Figure:
    fig_id: str
    page: int
    path: str
    caption: str
    width: int
    height: int


@dataclass
class PageRecord:
    pdf_id: str
    filename: str
    page: int  # 1-based, as printed in citations
    text: str
    n_words: int
    real_word_ratio: float
    ocr: bool
    lang: str
    figures: list[dict] = field(default_factory=list)


def _has_old_ocr(page: pymupdf.Page) -> bool:
    # render mode 3 = invisible text, the signature of a scanned page's OCR layer
    return any(span["type"] == 3 for span in page.get_texttrace())


def _ocr_page(pdf_path: Path, index: int) -> str:
    """OCR one page. Returns "" on failure (e.g. undecodable JPX image) so the caller keeps native text."""
    try:
        out = pymupdf4llm.to_markdown(str(pdf_path), pages=[index], use_ocr=OCRMode.FORCE_DROP_OLD, **MD_OPTS)
        return out[0]["text"] if out else ""
    except Exception as e:
        log.warning("OCR failed on %s page %d: %r", pdf_path.name, index + 1, e)
        return ""


_FIG_LINE = re.compile(r"Figure\s+\d+[.\-–]\d+[^\n]{0,120}")


def _figures(page: pymupdf.Page, boxes: list[dict], text: str, pdf_id: str, fig_dir: Path,
             seen: set[str]) -> list[Figure]:
    captions = [pymupdf.Rect(b["bbox"]) for b in boxes if b["class"] == "caption"]
    # group picture boxes by their nearest caption: one figure is often split into several boxes
    groups: dict[int, pymupdf.Rect] = {}
    for b in boxes:
        if b["class"] != "picture":
            continue
        rect = pymupdf.Rect(b["bbox"])
        dists = [min(abs(c.y0 - rect.y1), abs(rect.y0 - c.y1)) for c in captions]
        key = dists.index(min(dists)) if dists else -1 - len(groups)  # no caption -> own group
        groups[key] = groups[key] | rect if key in groups else rect

    text_captions = [clean_text(m.group(0)) for m in _FIG_LINE.finditer(text)]
    figs = []
    for key, rect in groups.items():
        if rect.width < FIG_MIN_SIDE_PT or rect.height < FIG_MIN_SIDE_PT:
            continue
        try:
            pix = page.get_pixmap(clip=rect, dpi=FIG_DPI)
            png = pix.tobytes("png")
        except Exception as e:  # one broken image must not stop the book
            log.warning("figure render failed on page %d: %r", page.number + 1, e)
            continue
        digest = hashlib.sha1(png).hexdigest()
        if digest in seen:  # repeated decorative image
            continue
        seen.add(digest)
        if key >= 0:
            caption = clean_text(page.get_textbox(captions[key]))
        else:  # scans have no caption boxes: fall back to a "Figure x-y ..." line on the page
            caption = text_captions[len(figs)] if len(figs) < len(text_captions) else ""
        fig_id = f"{pdf_id}_p{page.number + 1}_f{len(figs)}"
        path = fig_dir / f"{fig_id}.png"
        path.write_bytes(png)
        figs.append(Figure(fig_id, page.number + 1, str(path), caption, pix.width, pix.height))
    return figs


def process_pdf(pdf_path: Path, out_dir: Path, max_pages: int | None = None) -> dict:
    """Extract one PDF into out_dir/pages.jsonl + out_dir/figures/*.png. Returns stats."""
    import json

    pdf_id = pdf_path.stem
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(pdf_path)
    pages = list(range(min(doc.page_count, max_pages or doc.page_count)))
    chunks = pymupdf4llm.to_markdown(str(pdf_path), pages=pages, use_ocr=OCRMode.NEVER, **MD_OPTS)

    seen: set[str] = set()
    records, n_ocr, n_figs = [], 0, 0
    for chunk in chunks:
        page_no = chunk["metadata"]["page_number"]
        page = doc[page_no - 1]
        boxes = chunk.get("page_boxes", [])
        raw = clean_text(chunk["text"])
        n_words, ratio = word_stats(raw)
        has_image = any(b["class"] == "picture" for b in boxes) or bool(page.get_images())
        did_ocr = False
        if needs_ocr(n_words, ratio, _has_old_ocr(page), has_image):
            ocr_text = clean_text(_ocr_page(pdf_path, page_no - 1))
            ocr_words, ocr_ratio = word_stats(ocr_text)
            if ocr_words >= n_words or ocr_ratio > ratio:  # keep OCR only if it is better
                raw, n_words, ratio, did_ocr = ocr_text, ocr_words, ocr_ratio, True
                n_ocr += 1
        figs = _figures(page, boxes, raw, pdf_id, fig_dir, seen)
        n_figs += len(figs)
        records.append(
            PageRecord(pdf_id, pdf_path.name, page_no, raw, n_words, round(ratio, 3), did_ocr,
                       detect_language(raw), [asdict(f) for f in figs])
        )

    tmp = out_dir / "pages.jsonl.tmp"
    with tmp.open("w") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    tmp.rename(out_dir / "pages.jsonl")  # atomic: a finished book is never half-written
    return {"pdf_id": pdf_id, "pages": len(records), "ocr_pages": n_ocr, "figures": n_figs}
