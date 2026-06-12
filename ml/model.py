"""Main software artifact (homework spec): My_Translator_Model + CLI.

    python model.py train --dataset=/path/to/train.csv
    python model.py predict --text="šarrum ana ālim illik"
    python model.py predict-file --dataset=/path/to/test.csv

Streaming behavior of ``predict``:
- ``stream=True`` — tokens are yielded in real time via TextIteratorStreamer
  (greedy decoding; HF streamers do not support beam search).
- ``stream=False`` — full quality path: beam search candidates, optionally
  pooled over several checkpoints, picked by a chrF-based MBR selector.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Iterator

from akkadian_nmt.logging_utils import get_logger

log = get_logger("model")

DEFAULT_MODEL_DIR = "./model"
DEFAULT_RESULTS = "./data/results.csv"


class My_Translator_Model:
    """Akkadian (transliteration) -> English translator, ByT5-based."""

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR,
                 extra_model_dirs: list[str] | None = None,
                 num_beams: int = 4, candidates_per_model: int = 4):
        self.model_dir = model_dir
        # extra checkpoints pooled into the MBR candidate set (mini-ensemble)
        self.extra_model_dirs = extra_model_dirs or []
        self.num_beams = num_beams
        self.candidates_per_model = candidates_per_model
        self._model = None
        self._tokenizer = None

    # ------------------------------------------------------------------ train
    def train(self, dataset_path: str,
              config: str = "configs/exp2_full_seed13.yaml") -> None:
        """Build the processed corpus from the competition data and fine-tune.

        ``dataset_path`` — the Kaggle ``train.csv`` (its directory must also
        contain ``test.csv``, ``published_texts.csv`` and
        ``Sentences_Oare_FirstWord_LinNum.csv``) or that directory itself.
        The final model is saved to ``./model/``.
        """
        from akkadian_nmt.data_prep import build_corpus
        from akkadian_nmt.train import TrainConfig, train as run_train

        data_dir = Path(dataset_path)
        if data_dir.is_file():
            data_dir = data_dir.parent
        processed = data_dir / "processed"
        build_corpus(data_dir=str(data_dir), out_dir=str(processed))

        cfg = TrainConfig.from_yaml(config)
        cfg.train_file = str(processed / "train.csv")
        cfg.eval_file = str(processed / "dev.csv")
        cfg.output_dir = self.model_dir
        final_dir = run_train(cfg)

        # expose weights directly under ./model/ as the spec requires
        import shutil

        for item in Path(final_dir).iterdir():
            shutil.copy2(item, Path(self.model_dir) / item.name)
        log.info("model saved to %s", self.model_dir)

    # ---------------------------------------------------------------- predict
    def _ensure_loaded(self):
        if self._model is None:
            from akkadian_nmt.decode import load_model

            self._model, self._tokenizer = load_model(self.model_dir)

    def predict(self, text: str, stream: bool = True) -> Iterator[str] | str:
        """Translate one string; yields tokens when ``stream=True``."""
        if stream:
            return self._predict_stream(text)
        from akkadian_nmt.decode import translate

        return translate([text], [self.model_dir, *self.extra_model_dirs],
                         num_beams=self.num_beams,
                         candidates_per_model=self.candidates_per_model)[0]

    def _predict_stream(self, text: str) -> Iterator[str]:
        from transformers import TextIteratorStreamer

        from akkadian_nmt.normalize import normalize_translit

        self._ensure_loaded()
        model, tokenizer = self._model, self._tokenizer
        enc = tokenizer(normalize_translit(text), return_tensors="pt",
                        truncation=True, max_length=512).to(model.device)
        streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
        worker = threading.Thread(
            target=model.generate,
            kwargs={**enc, "streamer": streamer, "max_new_tokens": 512},
        )
        worker.start()
        try:
            yield from streamer
        finally:
            worker.join()

    # ----------------------------------------------------------- predict_file
    def predict_file(self, dataset_path: str) -> None:
        """Load ./model/ and write Kaggle-format predictions to ./data/results.csv."""
        from akkadian_nmt.decode import predict_kaggle

        predict_kaggle([self.model_dir, *self.extra_model_dirs], dataset_path,
                       out_csv=DEFAULT_RESULTS, num_beams=self.num_beams,
                       candidates_per_model=self.candidates_per_model)


# ----------------------------------------------------------------------- CLI
def _cli_predict(text: str, stream: bool = True, model_dir: str = DEFAULT_MODEL_DIR):
    m = My_Translator_Model(model_dir=model_dir)
    if stream:
        for token in m.predict(text, stream=True):
            print(token, end="", flush=True)
        print()
    else:
        print(m.predict(text, stream=False))


def _cli_train(dataset: str, config: str = "configs/exp2_full_seed13.yaml",
               model_dir: str = DEFAULT_MODEL_DIR):
    My_Translator_Model(model_dir=model_dir).train(dataset, config=config)


def _cli_predict_file(dataset: str, model_dir: str = DEFAULT_MODEL_DIR):
    My_Translator_Model(model_dir=model_dir).predict_file(dataset)


if __name__ == "__main__":
    import fire

    fire.Fire({
        "train": _cli_train,
        "predict": _cli_predict,
        "predict-file": _cli_predict_file,
        "predict_file": _cli_predict_file,
    })
