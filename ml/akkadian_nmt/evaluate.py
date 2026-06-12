"""Dev-set evaluation: BLEU, chrF++, their geometric mean (the competition
metric), and optionally COMET (auxiliary — not trained on Akkadian).

Usage (CLI):
    python -m akkadian_nmt.evaluate run --model_dirs=model/byt5-full-s13/final \
        --dev_file=data/processed/dev.csv --source=src_test_style --num_beams=4
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .logging_utils import get_logger

log = get_logger(__name__)


def score(hyps: list[str], refs: list[str], comet: bool = False,
          srcs: list[str] | None = None) -> dict:
    import sacrebleu

    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score
    metrics = {
        "bleu": round(bleu, 2),
        "chrf++": round(chrf, 2),
        "geo_mean": round((bleu * chrf) ** 0.5, 2) if bleu > 0 else 0.0,
    }
    if comet:
        try:
            from comet import download_model, load_from_checkpoint

            ckpt = download_model("Unbabel/wmt22-comet-da")
            comet_model = load_from_checkpoint(ckpt)
            data = [{"src": s, "mt": h, "ref": r}
                    for s, h, r in zip(srcs or refs, hyps, refs)]
            metrics["comet"] = round(
                comet_model.predict(data, batch_size=16).system_score, 4)
        except Exception:
            log.exception("COMET scoring failed; reporting without it")
    return metrics


def run(model_dirs: str | list[str], dev_file: str = "data/processed/dev.csv",
        source: str = "src_test_style", num_beams: int = 4,
        candidates_per_model: int = 1, comet: bool = False,
        max_samples: int | None = None, out_file: str | None = None,
        **gen_kwargs) -> dict:
    """Generate translations for the dev split and report the metric suite."""
    from .decode import translate

    with open(dev_file, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if max_samples:
        rows = rows[:max_samples]
    srcs = [r[source] for r in rows]
    refs = [r["tgt"] for r in rows]

    hyps = translate(srcs, model_dirs, num_beams=num_beams,
                     candidates_per_model=candidates_per_model,
                     normalize=False,  # dev sources are already normalized
                     **gen_kwargs)
    metrics = score(hyps, refs, comet=comet, srcs=srcs)
    metrics.update({"n": len(rows), "source": source, "num_beams": num_beams,
                    "candidates_per_model": candidates_per_model,
                    "model_dirs": model_dirs})
    log.info("dev metrics: %s", metrics)
    if out_file:
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as fh:
            json.dump({"metrics": metrics,
                       "predictions": [{"src": s, "hyp": h, "ref": r}
                                       for s, h, r in zip(srcs, hyps, refs)]},
                      fh, ensure_ascii=False, indent=1)
    print(json.dumps(metrics, indent=2, default=str))
    return metrics


def beam_sweep(model_dirs: str | list[str], dev_file: str = "data/processed/dev.csv",
               source: str = "src_test_style", beams: tuple = (1, 4, 8),
               max_samples: int | None = None) -> dict:
    """Experiment 2: greedy vs beam ablation."""
    results = {}
    for b in beams:
        results[f"beam_{b}"] = run(model_dirs, dev_file, source, num_beams=b,
                                   max_samples=max_samples)
    print(json.dumps(results, indent=2, default=str))
    return results


if __name__ == "__main__":
    import fire

    fire.Fire({"run": run, "beam_sweep": beam_sweep, "score": score})
