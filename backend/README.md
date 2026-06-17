# Backend: Akkadian Translation Service

Этот сервис предоставляет API для машинного перевода, реализованный на **FastAPI**, с использованием модели `transformers`.

## Структура

* `app.py` — основной код сервера.
* `akkadian-model/` — директория с весами модели.
* `Dockerfile` — инструкция для сборки контейнера.
* `requirements.txt` — зависимости проекта.

## Использование с Docker

### Сборка образа

```bash
docker build -t akkadian-backend .

```

### Запуск контейнера

Для запуска сервера используйте следующую команду:

```bash
docker run -d -p 8080:8080 --name akkadian-server akkadian-backend

```

Если веса модели находятся вне контекста сборки, используйте монтирование:

```bash
docker run -d -p 8080:8080 -v $(pwd)/akkadian-model:/app/akkadian-model akkadian-backend

```

Сервер будет доступен по адресу `http://localhost:8080`.

## API

Эндпоинт для перевода: `ws://localhost:8080/translate`.
Принимает JSON: `{"text": "ваш текст"}`. Возвращает переведенный текст.
