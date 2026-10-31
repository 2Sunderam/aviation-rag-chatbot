"""Text normalization for extracted page markdown.

Headers and footers are already dropped by the layout model
(header=False, footer=False in extract.py); this handles what is left.
"""

import re
import unicodedata

from langdetect import DetectorFactory, LangDetectException, detect

DetectorFactory.seed = 0  # deterministic language detection

_PICTURE_TEXT = re.compile(r"<!-- (Start|End) of picture text -->")
_IMAGE_PLACEHOLDER = re.compile(r"\*\*==> picture .*? intentionally omitted <==\*\*")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_PAGE_NUMBER_LINE = re.compile(r"^\s*(\d+[-–]\d+|\d+|[ivxlc]+)\s*$", re.IGNORECASE | re.MULTILINE)


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _PICTURE_TEXT.sub("", text)
    text = _IMAGE_PLACEHOLDER.sub("", text)
    text = text.replace("<br>", " ")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)  # "aero-\ndynamic" -> "aerodynamic"
    text = _PAGE_NUMBER_LINE.sub("", text)  # stray "3-12" page numbers
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_language(text: str) -> str:
    if len(text.split()) < 50:  # short figure-label pages give random guesses
        return "unknown"
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"
