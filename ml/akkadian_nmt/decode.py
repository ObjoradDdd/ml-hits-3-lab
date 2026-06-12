"""Decoding: greedy / beam search / multi-candidate generation with an
MBR (Minimum Bayes Risk) selector over chrF, usable as a reference-free
ensemble combiner (candidates pooled from several checkpoints).
"""

from __future__ import annotations

from pathlib import Path

import torch

from .logging_utils import get_logger
from .normalize import normalize_translit

log = get_logger(__name__)


def load_model(model_dir: str, device: str | None = None):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device).eval()
    log.info("loaded %s on %s", model_dir, device)
    return model, tokenizer


@torch.inference_mode()
def generate(model, tokenizer, texts: list[str], num_beams: int = 4,
             num_return_sequences: int = 1, batch_size: int = 8,
             max_src_len: int = 512, max_new_tokens: int = 512,
             length_penalty: float = 1.0, no_repeat_ngram_size: int = 0,
             normalize: bool = True) -> list[list[str]]:
    """Translate a list of sources; returns num_return_sequences candidates each."""
    device = next(model.parameters()).device
    out: list[list[str]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        if normalize:
            batch = [normalize_translit(t) for t in batch]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_src_len).to(device)
        gen = model.generate(
            **enc,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            max_new_tokens=max_new_tokens,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        for j in range(len(batch)):
            out.append(decoded[j * num_return_sequences:(j + 1) * num_return_sequences])
        if (i // batch_size) % 10 == 0:
            log.info("decoded %d/%d", min(i + batch_size, len(texts)), len(texts))
    return out


def mbr_select(candidates: list[str]) -> str:
    """Pick the candidate with max average chrF against the other candidates
    (consensus translation). Reference-free — works on the hidden test set."""
    import sacrebleu

    if len(candidates) == 1:
        return candidates[0]
    best, best_score = candidates[0], float("-inf")
    for hyp in candidates:
        others = [c for c in candidates if c is not hyp]
        score = sum(sacrebleu.sentence_chrf(hyp, [o]).score for o in others) / len(others)
        if score > best_score:
            best, best_score = hyp, score
    return best


def translate(texts: list[str], model_dirs: list[str] | str, num_beams: int = 4,
              candidates_per_model: int = 1, use_mbr: bool = False,
              **gen_kwargs) -> list[str]:
    """Full pipeline: one or more checkpoints -> pooled candidates -> MBR pick.

    Single model + candidates_per_model=1 degenerates to plain beam search.
    """
    if isinstance(model_dirs, str):
        model_dirs = [model_dirs]
    pooled: list[list[str]] = [[] for _ in texts]
    for md in model_dirs:
        model, tokenizer = load_model(md)
        cands = generate(model, tokenizer, texts, num_beams=num_beams,
                         num_return_sequences=candidates_per_model, **gen_kwargs)
        for pool, c in zip(pooled, cands):
            pool.extend(c)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if use_mbr or any(len(p) > 1 for p in pooled):
        return [mbr_select(p) for p in pooled]
    return [p[0] for p in pooled]


def predict_kaggle(model_dirs: list[str] | str, test_csv: str,
                   out_csv: str = "./data/results.csv", num_beams: int = 4,
                   candidates_per_model: int = 1, **gen_kwargs) -> str:
    """Read Kaggle test format (id,...,transliteration), write id,translation."""
    import pandas as pd

    df = pd.read_csv(test_csv)
    texts = df["transliteration"].fillna("").tolist()
    hyps = translate(texts, model_dirs, num_beams=num_beams,
                     candidates_per_model=candidates_per_model, **gen_kwargs)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": df["id"], "translation": hyps}).to_csv(out_csv, index=False)
    log.info("wrote %d predictions to %s", len(hyps), out_csv)
    return out_csv
