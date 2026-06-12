"""Orthography normalization for Akkadian transliteration.

Two jobs:

1. ``normalize_translit`` — bring every source file (train.csv with Unicode
   subscripts, published_texts.csv with ASCII sign indices, raw test input)
   to one canonical orthography.

2. ``corrupt_like_test`` — reproduce the deterministic character cipher found
   in the competition test set (font/OCR artifact of the source publications,
   derived in notebooks/01_eda.ipynb by aligning the 4 visible test rows with
   their clean counterpart, tablet AKT 5 1):

       š→a  ṭ→m  ḫ→+  4→„  5→…  {→(  }→)

   The cipher is lossy (test 'a' is both real 'a' and former 'š'), so we do
   NOT try to invert it on test input; instead training sources are corrupted
   with this map so the model learns the test distribution.
"""

from __future__ import annotations

import re
import unicodedata

_SUBSCRIPT_TO_ASCII = str.maketrans("₀₁₂₃₄₅₆₇₈₉ₓ", "0123456789x")

# half brackets and editorial damage marks around partially broken signs
_DAMAGE_MARKS = re.compile(r"[⸢⸣\[\]!?#*]")
# <gap>, <<...>>, <...> editorial insertions/omissions
_ANGLE_MARKUP = re.compile(r"<+[^<>]*>+")
_MULTISPACE = re.compile(r"\s+")

GAP_TOKEN = "…"  # the test set marks lacunae with the ellipsis character

_CIPHER = str.maketrans({
    "š": "a", "Š": "A",
    "ṭ": "m", "Ṭ": "M",
    "ḫ": "+", "Ḫ": "+",
    "4": "„",
    "5": "…",
    "{": "(",
    "}": ")",
})


def normalize_translit(text: str, keep_damage_marks: bool = False) -> str:
    """Canonical orthography: NFC, ASCII sign indices, unified gap marking,
    no editorial damage markup, collapsed whitespace. Diacritics are kept —
    ByT5 is byte-level and handles them natively."""
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_SUBSCRIPT_TO_ASCII)
    text = _ANGLE_MARKUP.sub(GAP_TOKEN, text)
    if not keep_damage_marks:
        text = _DAMAGE_MARKS.sub("", text)
    # collapse runs of gap marks ("… … …" -> "…")
    text = re.sub(rf"(?:{GAP_TOKEN}\s*)+", GAP_TOKEN + " ", text)
    text = _MULTISPACE.sub(" ", text).strip()
    return text


def corrupt_like_test(text: str) -> str:
    """Apply the test-set cipher to a *normalized* transliteration.

    Note: sign-index digits in the visible test rows appear as „ (4) and … (5);
    other digits (quantities like '14', '0.3333') pass through unchanged in the
    cipher map only for 4/5 — this matches the observed data, where '4' inside
    numbers is also replaced (e.g. u„-mì-im), so we replace globally."""
    return text.translate(_CIPHER)


def normalize_target(text: str) -> str:
    """Light cleanup of the English side: NFC, unified whitespace, strip
    <gap>-style markup that leaked into translations."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("<gap>", GAP_TOKEN)
    text = _MULTISPACE.sub(" ", text).strip()
    return text
