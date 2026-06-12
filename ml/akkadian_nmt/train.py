"""Fine-tune a ByT5 checkpoint on the processed Akkadian-English corpus.

Driven by a YAML config (see configs/). Designed for Colab free tier (T4):
fp32 (T5 is numerically unstable in fp16), gradient checkpointing, Adafactor,
checkpoint resume across session disconnects, optional push_to_hub + W&B.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class TrainConfig:
    model_name: str = "google/byt5-base"
    train_file: str = "data/processed/train.csv"
    eval_file: str = "data/processed/dev.csv"
    # which column feeds the encoder: src_raw | src_clean | src_test_style | both
    # ("both" duplicates every row with src_clean and src_test_style variants)
    train_source: str = "both"
    eval_source: str = "src_test_style"
    target_field: str = "tgt"
    origin_filter: list[str] | None = None  # e.g. ["doc"] for the docs-only baseline

    max_src_len: int = 512
    max_tgt_len: int = 512
    learning_rate: float = 1e-3  # Adafactor + constant-ish schedule, standard for T5
    num_epochs: float = 10.0
    per_device_batch: int = 2
    eval_batch: int = 8  # generation needs less memory than training
    grad_accum: int = 32  # effective batch 64
    warmup_steps: int = 100
    seed: int = 13
    gradient_checkpointing: bool = True
    bf16: bool = False  # set true on A100/L4; keep false on T4

    eval_steps: int = 200
    save_steps: int = 200
    save_total_limit: int = 2
    eval_max_samples: int = 200
    generation_max_len: int = 512
    logging_steps: int = 20

    output_dir: str = "model/byt5-run"
    hub_model_id: str | None = None  # e.g. "username/byt5-base-akkadian"
    push_to_hub: bool = False
    wandb_project: str = "akkadian-nmt"
    run_name: str = "byt5-run"
    resume: bool = True

    extra: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        kwargs = {k: v for k, v in raw.items() if k in known}
        extra = {k: v for k, v in raw.items() if k not in known}
        if extra:
            log.warning("unknown config keys ignored: %s", sorted(extra))
        return cls(**kwargs, extra=extra)


def _load_split(path: str, source: str, target_field: str,
                origin_filter: list[str] | None):
    """Return a datasets.Dataset with columns src/tgt."""
    from datasets import load_dataset

    ds = load_dataset("csv", data_files=path, split="train")
    if origin_filter:
        keep = set(origin_filter)
        ds = ds.filter(lambda r: r["origin"] in keep)
    if source == "both":
        a = ds.map(lambda r: {"src": r["src_clean"], "tgt": r[target_field]})
        b = ds.map(lambda r: {"src": r["src_test_style"], "tgt": r[target_field]})
        from datasets import concatenate_datasets

        ds = concatenate_datasets([a, b])
    else:
        ds = ds.map(lambda r: {"src": r[source], "tgt": r[target_field]})
    return ds.select_columns(["src", "tgt"])


def train(config: str | TrainConfig, **overrides) -> str:
    """Run fine-tuning; returns the output directory with the final model.

    ``overrides`` lets the CLI tweak any config field without editing the yaml:
        python -m akkadian_nmt.train --config=configs/baseline.yaml \
            --push_to_hub=True --hub_model_id=user/byt5-akkadian-baseline
    """
    import numpy as np
    import sacrebleu
    import transformers
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    cfg = TrainConfig.from_yaml(config) if isinstance(config, str) else config
    for key, value in overrides.items():
        if key not in cfg.__dataclass_fields__:
            raise ValueError(f"unknown config override: {key}")
        setattr(cfg, key, value)
    log.info("train config: %s", cfg)
    transformers.set_seed(cfg.seed)
    os.environ.setdefault("WANDB_PROJECT", cfg.wandb_project)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name)
    if cfg.gradient_checkpointing:
        model.config.use_cache = False

    train_ds = _load_split(cfg.train_file, cfg.train_source, cfg.target_field, cfg.origin_filter)
    eval_ds = _load_split(cfg.eval_file, cfg.eval_source, cfg.target_field, None)
    if len(eval_ds) > cfg.eval_max_samples:
        eval_ds = eval_ds.shuffle(seed=cfg.seed).select(range(cfg.eval_max_samples))
    log.info("train pairs: %d | eval pairs: %d", len(train_ds), len(eval_ds))

    def tokenize(batch):
        enc = tokenizer(batch["src"], max_length=cfg.max_src_len, truncation=True)
        lab = tokenizer(text_target=batch["tgt"], max_length=cfg.max_tgt_len, truncation=True)
        enc["labels"] = lab["input_ids"]
        return enc

    train_tok = train_ds.map(tokenize, batched=True, remove_columns=["src", "tgt"])
    eval_tok = eval_ds.map(tokenize, batched=True, remove_columns=["src", "tgt"])

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        hyp = tokenizer.batch_decode(preds, skip_special_tokens=True)
        ref = tokenizer.batch_decode(labels, skip_special_tokens=True)
        chrf = sacrebleu.corpus_chrf(hyp, [ref], word_order=2).score
        bleu = sacrebleu.corpus_bleu(hyp, [ref]).score
        return {"chrf++": chrf, "bleu": bleu,
                "geo_mean": (chrf * bleu) ** 0.5 if bleu > 0 else 0.0}

    args = Seq2SeqTrainingArguments(
        output_dir=cfg.output_dir,
        run_name=cfg.run_name,
        seed=cfg.seed,
        num_train_epochs=cfg.num_epochs,
        learning_rate=cfg.learning_rate,
        optim="adafactor",
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=cfg.warmup_steps,
        per_device_train_batch_size=cfg.per_device_batch,
        per_device_eval_batch_size=cfg.eval_batch,
        gradient_accumulation_steps=cfg.grad_accum,
        gradient_checkpointing=cfg.gradient_checkpointing,
        bf16=cfg.bf16,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="chrf++",
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=cfg.generation_max_len,
        logging_steps=cfg.logging_steps,
        report_to=["wandb"],
        push_to_hub=cfg.push_to_hub,
        hub_model_id=cfg.hub_model_id,
        hub_strategy="checkpoint",  # resumable from the Hub after disconnect
        hub_private_repo=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    last_ckpt = None
    if cfg.resume and Path(cfg.output_dir).exists():
        from transformers.trainer_utils import get_last_checkpoint

        last_ckpt = get_last_checkpoint(cfg.output_dir)
    if cfg.resume and last_ckpt is None and cfg.push_to_hub and cfg.hub_model_id:
        # fresh VM after a Colab disconnect: pull the checkpoint pushed to the Hub
        # by hub_strategy="checkpoint" (stored under last-checkpoint/ in the repo)
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(repo_id=cfg.hub_model_id,
                              allow_patterns=["last-checkpoint/*"],
                              local_dir=cfg.output_dir)
            candidate = Path(cfg.output_dir) / "last-checkpoint"
            if (candidate / "trainer_state.json").exists():
                last_ckpt = str(candidate)
        except Exception as err:  # repo may not exist yet on the first run
            log.info("no resumable checkpoint on the Hub (%s)", err)
    if last_ckpt:
        log.info("resuming from checkpoint %s", last_ckpt)

    try:
        trainer.train(resume_from_checkpoint=last_ckpt)
    except Exception:
        log.exception("training failed")
        raise

    final_dir = str(Path(cfg.output_dir) / "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    if cfg.push_to_hub:
        trainer.push_to_hub()
    metrics = trainer.evaluate()
    log.info("final dev metrics: %s", metrics)
    return final_dir


if __name__ == "__main__":
    import fire

    fire.Fire(train)
