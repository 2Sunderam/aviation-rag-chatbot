"""Text-quality checks used to decide which pages need OCR."""

import re
from functools import lru_cache

DICT_PATH = "/usr/share/dict/words"  # ships with macOS
SUFFIXES = ("s", "es", "ed", "d", "ing", "ly", "er", "al")

# Thresholds calibrated on book 08 (clean) vs book 09 (old scan):
# clean text pages score 0.90-0.98, garbage OCR on figure pages 0.63-0.79.
MIN_REAL_WORD_RATIO = 0.85
MIN_WORDS = 15


@lru_cache(maxsize=1)
def _words() -> frozenset[str]:
    with open(DICT_PATH) as f:
        return frozenset(w.strip().lower() for w in f)


def _is_word(w: str) -> bool:
    words = _words()
    return w in words or any(w.endswith(s) and w[: -len(s)] in words for s in SUFFIXES)


def word_stats(text: str) -> tuple[int, float]:
    """Return (word count, share of words found in the English dictionary)."""
    text = re.sub(r"<[^>]+>", " ", text)
    tokens = re.findall(r"[A-Za-z]{3,}", text)
    if not tokens:
        return 0, 0.0
    real = sum(_is_word(t.lower()) for t in tokens)
    return len(tokens), real / len(tokens)


def needs_ocr(n_words: int, real_ratio: float, has_old_ocr: bool, has_image: bool) -> bool:
    """OCR only when the text layer is missing or garbage."""
    if has_old_ocr and real_ratio < MIN_REAL_WORD_RATIO:
        return True  # scanned page with a bad old OCR layer
    if n_words < MIN_WORDS and has_image:
        return True  # image-only page (scan or full-page figure)
    return False
