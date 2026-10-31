"""Structure-aware recursive chunking of extracted page markdown.

Each book's pages are joined into one text so sections and overlap can cross
page breaks. The splitter prefers markdown headings, then paragraphs, lines,
words. Character offsets map every chunk back to its start/end page, so
citations stay exact. Deterministic: same input -> same chunks and ids.

Size: ~800 tokens, 15% overlap. Tokens are estimated at 4 chars/token
(English prose), so 3200 chars per chunk and 480 chars overlap.
"""

import bisect
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from rag.config import TITLES
from rag.ingestion.quality import _is_word, word_stats

CHARS_PER_TOKEN = 4
CHUNK_TOKENS = 800
OVERLAP = 0.15
MIN_CHUNK_CHARS = 200  # drop near-empty chunks (covers, blank pages)

CHUNK_SIZE = CHUNK_TOKENS * CHARS_PER_TOKEN
CHUNK_OVERLAP = int(CHUNK_SIZE * OVERLAP)

MIN_REAL_WORD_RATIO = 0.5  # below this a chunk is chart/screenshot label noise
SKIP_SECTIONS = ("table of contents", "contents", "index")

_HEADING = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_DOT_LEADER = re.compile(r"(\s?\.){4,}")
_INLINE_TAGS = re.compile(r"</?(mark|sup|sub|u|span|small)[^>]*>", re.IGNORECASE)
# PDF ligatures extracted with a space: "fl ightpath", "suffi cient"
_LIGATURE_BREAK = re.compile(r"\b([A-Za-z]*(?:ffi|ffl|fi|fl|ff)) ([a-z]{2,})")


def _join_ligature(m: re.Match) -> str:
    left, right = m.group(1), m.group(2)
    # "off the" stays (both real words); "fl ight" / "suffi cient" get joined
    if _is_word(left.lower()) and _is_word(right):
        return m.group(0)
    return left + right


def _normalize(text: str) -> str:
    text = _INLINE_TAGS.sub("", text)
    text = _LIGATURE_BREAK.sub(_join_ligature, text)
    return _DOT_LEADER.sub(" ", text)  # "Release ......12-7" -> "Release 12-7"
# TOC / index lines end in a page reference: "Hypoxia ... 13-4", "Lift, 4-2, 5-7"
_PAGE_REF_LINE = re.compile(r"(\d+[-–]\d+|\b\d{1,3})\W*$")


def _is_toc_or_index(body: str) -> bool:
    lines = [l for l in body.splitlines() if l.strip() and l.strip("|-: ")]
    if len(lines) < 5:
        return False
    refs = sum(bool(_PAGE_REF_LINE.search(l.strip(" |*"))) for l in lines)
    return refs / len(lines) > 0.5


@dataclass
class Chunk:
    chunk_id: str
    pdf_id: str
    filename: str
    title: str
    page: int  # page where the chunk starts (used in citations)
    page_end: int  # page where it ends (differs when the chunk crosses a page break)
    section: str  # nearest heading above the chunk
    text: str
    n_chars: int
    n_tokens_est: int
    ocr: bool  # any OCR'd page involved
    lang: str


def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter.from_language(
        Language.MARKDOWN,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
        strip_whitespace=True,
    )


def _clean_heading(h: str) -> str:
    return re.sub(r"[*_`]+", "", h).strip()


def chunk_book(pages_jsonl: Path) -> list[Chunk]:
    pages = [json.loads(line) for line in pages_jsonl.open()]
    pages.sort(key=lambda p: p["page"])
    pdf_id, filename = pages[0]["pdf_id"], pages[0]["filename"]

    # join pages, remember where each page starts in the combined text
    parts, starts, offset = [], [], 0
    for p in pages:
        starts.append(offset)
        page_text = _normalize(p["text"])
        parts.append(page_text)
        offset += len(page_text) + 2  # "\n\n" joiner
    text = "\n\n".join(parts)

    headings = [(m.start(), _clean_heading(m.group(2))) for m in _HEADING.finditer(text)]
    heading_pos = [h[0] for h in headings]

    def page_at(pos: int) -> int:
        return bisect.bisect_right(starts, pos) - 1  # index into pages

    chunks = []
    for doc in _splitter().create_documents([text]):
        body = doc.page_content
        if len(body) < MIN_CHUNK_CHARS or _is_toc_or_index(body):
            continue
        start = doc.metadata["start_index"]
        end = start + len(body) - 1
        first, last = page_at(start), page_at(end)
        h = bisect.bisect_right(heading_pos, start) - 1
        section = headings[h][1] if h >= 0 else ""
        if section.lower().strip(" :") in SKIP_SECTIONS:
            continue
        if word_stats(body)[1] < MIN_REAL_WORD_RATIO:
            continue
        span = pages[first : last + 1]
        langs = [p["lang"] for p in span if p["lang"] != "unknown"]
        chunks.append(
            Chunk(
                chunk_id=f"{pdf_id}_c{len(chunks):05d}",
                pdf_id=pdf_id,
                filename=filename,
                title=TITLES.get(pdf_id, pdf_id),
                page=pages[first]["page"],
                page_end=pages[last]["page"],
                section=section,
                text=body,
                n_chars=len(body),
                n_tokens_est=len(body) // CHARS_PER_TOKEN,
                ocr=any(p["ocr"] for p in span),
                lang=max(set(langs), key=langs.count) if langs else "unknown",
            )
        )
    return chunks


def write_chunks(chunks: list[Chunk], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".jsonl.tmp")
    with tmp.open("w") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    tmp.rename(out)
