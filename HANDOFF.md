# ML → Serving handoff (Akkadian → English)

Шпаргалка для софт-части (стриминговый переводчик). ML-часть живёт в `ml/`,
подробности — в `ml/README.md`.

## 1. Модель

- **Для сервинга бери лучшую одиночную модель:** HF Hub `kirmala/akkadian-byt5-full-seed13`
  (ByT5-base, fine-tuned). Это `seed13` — лучший одиночный чекпоинт (dev geo-mean 26.4).
- Ансамбль `seed13 + seed42` (dev 27.3) — только для офлайн Kaggle-сабмита, в реальном
  времени его стримить не нужно.
- ByT5 — **byte-level**, отдельного словаря/токенайзера качать не надо, всё в репозитории
  модели (`config.json`, `model.safetensors`, `tokenizer_config.json`).
- Доступ: репозиторий приватный — сделай public или дай read-токен.

## 2. 🔴 Нормализация входа — ОБЯЗАТЕЛЬНО

Модель обучена на нормализованной транслитерации. Текст пользователя **перед подачей в
модель** прогоняй через эту функцию (копия `ml/akkadian_nmt/normalize.py`). Без неё
качество заметно падает.

```python
import re, unicodedata
_SUB = str.maketrans("₀₁₂₃₄₅₆₇₈₉ₓ", "0123456789x")
_DAMAGE = re.compile(r"[⸢⸣\[\]!?#*]")
_ANGLE = re.compile(r"<+[^<>]*>+")
_WS = re.compile(r"\s+")
GAP = "…"

def normalize_translit(text: str) -> str:
    text = unicodedata.normalize("NFC", text).translate(_SUB)
    text = _ANGLE.sub(GAP, text)              # <gap>, <...> -> …
    text = _DAMAGE.sub("", text)              # убрать пометки повреждений
    text = re.sub(rf"(?:{GAP}\s*)+", GAP + " ", text)
    return _WS.sub(" ", text).strip()
```

Эталонные правила и почему так — в `ml/README.md` (раздел про орфографию/шифр теста).

## 3. Рекомендуемый путь сервинга — Path A: ByT5 + TGI

TGI нативно стримит encoder-decoder модели (T5/ByT5). Пример:

```bash
docker run --gpus all -p 8080:80 \
  -e MODEL_ID=kirmala/akkadian-byt5-full-seed13 \
  -e HF_TOKEN=... \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id kirmala/akkadian-byt5-full-seed13
```

Гейтвей: принимает текст → `normalize_translit` → шлёт в TGI с `stream=True` →
прокидывает токены в UI по **SSE** (или WebSocket).

Альтернатива — использовать готовый класс из `ml/model.py` (`My_Translator_Model`),
у него есть `predict(text, stream=True)`, который уже стримит токены через
`TextIteratorStreamer`. Можно поднять как свой инференс-сервис, если TGI не зайдёт.

## 4. Нюанс стриминга

Стрим идёт **greedy** (beam search по токенам не стримится). То есть стримящийся ответ —
уровня greedy-декодинга (dev geo ~25.8), это нормально для UX. Для максимального качества
без стрима используется beam search (см. `ml/akkadian_nmt/decode.py`).

## 5. Что требуется от софт-части (из задания)

- `GET /health`
- `POST /translate` — принимает `{text}`, отдаёт **SSE/WebSocket** поток токенов.
- Двухпанельный UI (слева аккадский, справа английский), debounce по вводу, видимый стрим.
- `Dockerfile` для инференса + для гейтвея/фронта, `docker-compose.yaml` (GPU и CPU профили).
- Логи всего пайплайна в `./data/log_file.log` (логгер уже есть: `ml/akkadian_nmt/logging_utils.py`).
- Замерить и записать в README: TTFT (медиана и p95) и tokens/sec.

## 6. Контакты артефактов

- Код ML + интерфейс: ветка `ml-dev`, папка `ml/`.
- Веса: HF Hub `kirmala/akkadian-byt5-full-seed13` (и `...-seed42`).
- Трекинг обучения: W&B проект `akkadian-nmt`.
