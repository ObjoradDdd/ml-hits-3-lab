# Akkadian → English NMT — ML part (HW3, Deep Past Initiative)

**Автор ML-части:** Кирилл Малахов, группа `<номер группы>`
(веб-сервис / стриминг / Docker — см. `backend/` и `frontend/`, делает одногруппник)

Перевод древнеассирийских клинописных табличек (латинская транслитерация) на английский.
Соревнование: [Kaggle — Deep Past Initiative: Machine Translation](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation).

## Ключевые инсайты (см. `notebooks/01_eda.ipynb`)

1. **Тест испорчен детерминированным посимвольным шифром** (артефакт шрифта публикаций):
   `š→a`, `ṭ→m`, `ḫ→+`, `4→„`, `5→…`, `{}→()`. Шифр необратим, поэтому мы не чистим тест,
   а **аугментируем train-источник тем же шифром** (`akkadian_nmt/normalize.py`).
2. **+8k пар на уровне предложений** добываются из `Sentences_Oare_FirstWord_LinNum.csv`
   выравниванием по первому слову (85% текстов выравниваются полностью), источник
   транслитераций — `published_texts.csv`. Всё из официального датасета соревнования.
3. Тестовые примеры — диапазоны строк таблички, поэтому предложения группируются
   в чанки по 1–5, имитируя распределение длин теста.

Модель: **ByT5-base** (byte-level — диакритика `š/ṣ/ṭ/ā` и индексы знаков `bi₄`
обрабатываются без токенизатора). Обучение — Colab T4: fp32 (T5 нестабилен в fp16),
gradient checkpointing, Adafactor, возобновление с чекпоинта после обрыва сессии.

## Results (ablation)

| # | Техника | BLEU (dev) | chrF++ (dev) | geo-mean (dev) | Kaggle public LB | Примечания |
|---|---------|-----------:|-------------:|---------------:|-----------------:|------------|
| 0 | Baseline: ByT5-base, только train.csv (доки), raw орфография, greedy | — | — | — | — | `configs/baseline.yaml` |
| 1 | + нормализация орфографии и test-style аугментация | — | — | — | — | `configs/exp1_norm.yaml` |
| 2 | + sentence-chunk пары (≈+2.5k пар) | — | — | — | — | `configs/exp2_full_seed13.yaml` |
| 3 | + beam search (sweep 1/4/8) | — | — | — | — | лучший beam = `<N>` |
| 4 | + мини-ансамбль 2× seed (13, 42), MBR-chrF селектор | — | — | — | — | `configs/exp3_full_seed42.yaml` |

**Финальные метрики:** Kaggle public `—` / private `—` · dev: BLEU `—`, chrF++ `—`, COMET (`Unbabel/wmt22-comet-da`) `—`.

Скриншот лидерборда: `docs/leaderboard.png` *(добавить)*.

## Данные и лицензии

| Источник | Объём | Использование | Лицензия |
|---|---|---|---|
| Kaggle `train.csv` | 1561 док-пар | обучение | правила соревнования (Deep Past Initiative / OARE) |
| Kaggle `Sentences_Oare_FirstWord_LinNum.csv` | 9782 предложения | +2.5k чанк-пар после выравнивания | те же (входит в датасет соревнования) |
| Kaggle `published_texts.csv` | 7953 транслитерации | источник для выравнивания | те же |

Внешние корпуса не использовались. **Контроль утечки:** табличка видимого теста (AKT 5 1)
и все её дубликаты-копии (31 текст, циркулярное письмо существует во многих копиях)
исключены из обучения поиском по содержимому; train/dev разбиты по id текста;
точные дубликаты источника между train и dev удалены (`akkadian_nmt/data_prep.py`,
проверки в `data/processed/stats.json`).

## How-to

```bash
cd ml
poetry install                  # или: pip install -e .
# данные соревнования положить в ml/data/ (kaggle competitions download -c deep-past-initiative-machine-translation)
```

### Подготовка корпуса
```bash
poetry run python -m akkadian_nmt.data_prep --data_dir=./data --out_dir=./data/processed
```

### Обучение (Colab — рекомендуется, GPU)
Открыть `notebooks/colab_train.ipynb` в Colab, выбрать конфиг (`CONFIG = "configs/..."`),
добавить секреты `WANDB_API_KEY`, `KAGGLE_USERNAME/KEY`. Чекпоинты пишутся в Google Drive
и переживают обрыв сессии.

### Обучение (CLI, если есть GPU)
```bash
poetry run python model.py train --dataset=./data/train.csv          # = exp2-конфиг
poetry run python -m akkadian_nmt.train --config=configs/baseline.yaml
```

### Эксперименты
```bash
# beam sweep (эксперимент 2)
poetry run python -m akkadian_nmt.evaluate beam_sweep --model_dirs=model/byt5-full-s13/final
# полный метрический набор + COMET
poetry run python -m akkadian_nmt.evaluate run --model_dirs=model/byt5-full-s13/final --num_beams=4 --comet=True
# ансамбль (эксперимент 3): пул кандидатов двух чекпоинтов + MBR-chrF
poetry run python -m akkadian_nmt.evaluate run \
  --model_dirs='["model/byt5-full-s13/final","model/byt5-full-s42/final"]' \
  --num_beams=4 --candidates_per_model=4
```

### Перевод и сабмит
```bash
poetry run python model.py predict --text="um-ma kà-ru-um kà-ni-iš-ma" # стримит токены
poetry run python model.py predict-file --dataset=./data/test.csv      # -> ./data/results.csv
kaggle competitions submit -c deep-past-initiative-machine-translation -f data/results.csv -m "..."
```

## Конфигурация декодирования (финальная)

*(заполнить после экспериментов)* — N кандидатов: `—`, beam: `—`, селектор: MBR по chrF,
состав ансамбля: ByT5-base seed 13 + seed 42.

## Логи и трекинг

- Все стадии пишут в `./data/log_file.log` (singleton-логгер `akkadian_nmt/logging_utils.py`).
- Эксперименты: W&B, проект `akkadian-nmt` *(ссылка — добавить)*; веса — HF Hub *(ссылка — добавить)*.

## Ресурсы

- Gutherz et al., *Translating Akkadian to English with NMT*, PNAS Nexus 2023
- Xue et al., *ByT5: Towards a token-free future*, 2021
- Freitag et al., *MBR decoding*, 2022
- sacreBLEU: BLEU `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp`, chrF++ (`word_order=2`)
