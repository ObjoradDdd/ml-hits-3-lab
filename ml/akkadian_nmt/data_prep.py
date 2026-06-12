"""Build the processed parallel corpus for the Deep Past competition.

Sources (all shipped inside the Kaggle competition dataset):

- ``train.csv``                          — 1561 document-level pairs (oare_id, transliteration, translation)
- ``published_texts.csv``                — 7953 transliterations (no translations)
- ``Sentences_Oare_FirstWord_LinNum.csv``— 9782 sentence-level English translations
  with the spelling of the sentence's first word; joined to transliterations via
  text_uuid == oare_id and anchored by monotone first-word search (see 01_eda.ipynb:
  85% of texts anchor fully, 95% of sentences).

The Kaggle test items are *line ranges* of a tablet (~1-10 lines), so anchored
sentences are grouped into chunks of 1-5 consecutive sentences to mimic the
test length distribution. Each output row carries the clean source and its
test-style corrupted variant (see normalize.corrupt_like_test).

Output: <out_dir>/train.csv, <out_dir>/dev.csv with columns
    text_id, origin, src_raw, src_clean, src_test_style, tgt
plus <out_dir>/stats.json. ``src_raw`` is the verbatim transliteration
(whitespace-collapsed only) — used by the no-normalization baseline.

Leakage protection:
- the tablet behind the 4 visible test rows (AKT 5 1) is excluded entirely,
  located by content match rather than a hard-coded id;
- dev split is by text id (no tablet straddles train/dev);
- train rows whose source also occurs in dev are dropped.
"""

from __future__ import annotations

import csv
import json
import random
import re
import zlib
from itertools import groupby
from pathlib import Path

from .logging_utils import get_logger
from .normalize import corrupt_like_test, normalize_target, normalize_translit

log = get_logger(__name__)

csv.field_size_limit(10**9)

_SUBSCRIPT = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

MIN_SRC_CHARS = 5
MIN_TGT_CHARS = 3
LEN_RATIO_RANGE = (0.2, 6.0)
CHUNK_SIZES = (1, 2, 3, 4, 5)


def _read(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _anchor_key(word: str) -> str:
    """Loose word form used to match Sentences first_word_spelling against
    transliteration tokens across orthography variants."""
    word = word.translate(_SUBSCRIPT).lower()
    word = re.sub(r"[<>\[\]⸢⸣!?#*…]", "", word)
    return re.sub(r"\d+$", "", word)


def _find_test_tablet_ids(test_rows: list[dict], texts: dict[str, str]) -> set[str]:
    """Ids of corpus texts that are (fragments of) visible test tablets.

    The test source is corrupted with a 1:1 cipher that leaves most characters
    intact, so a clean text whose cipher image contains a test chunk is a leak."""
    leaked = set()
    chunks = [normalize_translit(r["transliteration"]) for r in test_rows]
    for tid, text in texts.items():
        image = corrupt_like_test(normalize_translit(text))
        if any(c in image or image in c for c in chunks):
            leaked.add(tid)
    return leaked


def _anchor_sentences(words: list[str], sentences: list[dict]) -> list[int] | None:
    """Monotone first-word anchoring; None unless every sentence anchors."""
    keys = [_anchor_key(w) for w in words]
    anchors, pos = [], 0
    for sent in sentences:
        spelling = _anchor_key(sent["first_word_spelling"]) if sent["first_word_spelling"] else ""
        if not spelling:
            return None
        found = next((j for j in range(pos, len(keys)) if keys[j] == spelling), -1)
        if found < 0:
            return None
        anchors.append(found)
        pos = found + 1
    return anchors


def _chunk_pairs(text_id: str, words: list[str], anchors: list[int],
                 translations: list[str]) -> list[tuple[str, str]]:
    """Group consecutive sentences into chunks of 1-5 to mimic test items.
    Deterministic per text (seeded by text id)."""
    rng = random.Random(zlib.crc32(text_id.encode()))
    bounds = anchors + [len(words)]
    pairs = []
    i = 0
    while i < len(anchors):
        k = min(rng.choice(CHUNK_SIZES), len(anchors) - i)
        src = " ".join(words[bounds[i]:bounds[i + k]])
        tgt = " ".join(t for t in translations[i:i + k] if t)
        pairs.append((src, tgt))
        i += k
    return pairs


def _keep(src: str, tgt: str) -> bool:
    if len(src) < MIN_SRC_CHARS or len(tgt) < MIN_TGT_CHARS:
        return False
    ratio = len(tgt) / max(len(src), 1)
    return LEN_RATIO_RANGE[0] <= ratio <= LEN_RATIO_RANGE[1]


def build_corpus(data_dir: str = "./data", out_dir: str = "./data/processed",
                 dev_frac: float = 0.1, seed: int = 13) -> dict:
    data, out = Path(data_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_rows = _read(data / "train.csv")
    test_rows = _read(data / "test.csv")
    published = _read(data / "published_texts.csv")
    sentences = _read(data / "Sentences_Oare_FirstWord_LinNum.csv")
    log.info("loaded: train=%d test=%d published=%d sentences=%d",
             len(train_rows), len(test_rows), len(published), len(sentences))

    texts: dict[str, str] = {r["oare_id"]: r["transliteration"]
                             for r in published if r["transliteration"]}
    texts.update({r["oare_id"]: r["transliteration"] for r in train_rows})

    leaked_ids = _find_test_tablet_ids(test_rows, texts)
    log.info("excluding %d text(s) overlapping visible test tablets: %s",
             len(leaked_ids), sorted(leaked_ids))

    pairs: list[dict] = []  # text_id, origin, src_clean, tgt

    # 1) document-level pairs from train.csv
    for r in train_rows:
        if r["oare_id"] in leaked_ids:
            continue
        src = normalize_translit(r["transliteration"])
        tgt = normalize_target(r["translation"])
        if _keep(src, tgt):
            pairs.append({"text_id": r["oare_id"], "origin": "doc",
                          "src_raw": " ".join(r["transliteration"].split()),
                          "src_clean": src, "tgt": tgt})

    # 2) sentence-chunk pairs via first-word anchoring
    sentences.sort(key=lambda r: (r["text_uuid"], int(r["sentence_obj_in_text"] or 0)))
    anchored_texts = skipped_texts = 0
    for tid, grp_iter in groupby(sentences, key=lambda r: r["text_uuid"]):
        grp = [s for s in grp_iter if s["translation"].strip()]
        raw = texts.get(tid)
        if raw is None or not grp or tid in leaked_ids:
            continue
        words = raw.split()
        anchors = _anchor_sentences(words, grp)
        if anchors is None:
            skipped_texts += 1
            continue
        anchored_texts += 1
        translations = [normalize_target(s["translation"]) for s in grp]
        for src_raw, tgt in _chunk_pairs(tid, words, anchors, translations):
            src = normalize_translit(src_raw)
            if _keep(src, tgt):
                pairs.append({"text_id": tid, "origin": "chunk", "src_raw": src_raw,
                              "src_clean": src, "tgt": tgt})
    log.info("sentence anchoring: %d texts anchored, %d skipped", anchored_texts, skipped_texts)

    # 3) exact dedup on (src, tgt)
    seen: set[tuple[str, str]] = set()
    deduped = []
    for p in pairs:
        key = (p["src_clean"].lower(), p["tgt"].lower())
        if key not in seen:
            seen.add(key)
            deduped.append(p)
    log.info("dedup: %d -> %d pairs", len(pairs), len(deduped))

    # 4) corrupted source variant
    for p in deduped:
        p["src_test_style"] = corrupt_like_test(p["src_clean"])

    # 5) split by text id (stable hash, independent of insertion order)
    def is_dev(tid: str) -> bool:
        return (zlib.crc32(f"{seed}:{tid}".encode()) % 10_000) / 10_000 < dev_frac

    train_set = [p for p in deduped if not is_dev(p["text_id"])]
    dev_set = [p for p in deduped if is_dev(p["text_id"])]

    # 6) cross-split source leakage: drop train rows whose source occurs in dev
    dev_srcs = {p["src_clean"].lower() for p in dev_set}
    before = len(train_set)
    train_set = [p for p in train_set if p["src_clean"].lower() not in dev_srcs]
    log.info("cross-split dedup dropped %d train rows", before - len(train_set))

    fields = ["text_id", "origin", "src_raw", "src_clean", "src_test_style", "tgt"]
    for name, subset in (("train", train_set), ("dev", dev_set)):
        with open(out / f"{name}.csv", "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(subset)

    stats = {
        "train_pairs": len(train_set),
        "dev_pairs": len(dev_set),
        "train_docs": sum(p["origin"] == "doc" for p in train_set),
        "train_chunks": sum(p["origin"] == "chunk" for p in train_set),
        "dev_docs": sum(p["origin"] == "doc" for p in dev_set),
        "dev_chunks": sum(p["origin"] == "chunk" for p in dev_set),
        "anchored_texts": anchored_texts,
        "skipped_texts": skipped_texts,
        "excluded_test_tablets": sorted(leaked_ids),
        "dev_frac": dev_frac,
        "seed": seed,
    }
    with open(out / "stats.json", "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    log.info("corpus built: %s", stats)
    return stats


if __name__ == "__main__":
    import fire

    fire.Fire(build_corpus)
