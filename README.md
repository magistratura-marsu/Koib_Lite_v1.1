# KOIB-V-4.5 — Оптимизированная RAG-система

> Экстремально оптимизированная Retrieval-Augmented Generation система для технической документации.
> Разработана для работы на слабом VPS: **1 vCPU / 2 ГБ ОЗУ**.

## Обзор

KOIB-V-4.5 — это система问答 по технической документации, построенная на гибридном поиске (векторный + BM25) с переранжированием и LLM-генерацией ответов. Система готова к интеграции с VK Callback API для работы в качестве чат-бота во время реальных выборов.

### Ключевые особенности

- **Лёгкие модели**: `intfloat/multilingual-e5-small` (33 МБ) с динамической квантизацией PyTorch (int8)
- **SQLite вместо JSON**: DocStore и кэш HyDE на SQLite для минимального потребления RAM
- **Только Tesseract OCR**: нулевое потребление RAM в Python (в отличие от EasyOCR)
- **Эвристические сводки**: таблицы и формулы описываются без LLM (мгновенная обработка)
- **Жёсткие таймауты**: 10с авторизация, 30с генерация, 700 токенов максимум
- **FastAPI сервер**: асинхронный эндпоинт для VK Callback API

## Структура проекта

```
koib-v4.5/
├── config.py               # Централизованная конфигурация
├── main.py                 # CLI-точка входа (--ingest, --query, --serve, --evaluate)
├── batch_ingest.py         # Пакетная индексация документов
├── requirements.txt        # Зависимости Python
├── .env.example            # Пример конфигурации окружения
├── README.md               # Этот файл
│
├── src/                    # Основной код системы
│   ├── __init__.py
│   ├── utils.py            # Утилиты (хэши, очистка, токены, детекция моделей)
│   ├── logging_module.py   # Настройка логирования
│   ├── parsing.py          # Парсинг PDF (PyMuPDF) и DOCX
│   ├── chunking.py         # Умное разбиение на чанки
│   ├── indexing.py         # FAISS + BM25 + SQLite DocStore + квантизированные эмбеддинги
│   ├── retrieval.py        # Гибридный поиск (vector + BM25 + reranker + HyDE)
│   ├── generation.py       # LLM-клиенты (GigaChat, OpenAI, Ollama) + пайплайн генерации
│   ├── validation.py       # Валидация ответов (неуверенность, источники, семантика)
│   ├── quarantine.py       # Карантин сомнительных чанков (SQLite)
│   ├── safety.py           # Фильтрация опасного контента (prompt-injection, конфиденц.)
│   └── evaluation.py       # Оценка качества RAG (LLM-as-Judge, 4 метрики)
│
├── api/                    # FastAPI-слой
│   ├── __init__.py
│   ├── app.py              # FastAPI приложение
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── health.py       # Health check эндпоинт
│   │   └── vk_callback.py  # VK Callback API обработчик
│   └── middleware/
│       ├── __init__.py
│       └── logging.py      # Логирование HTTP-запросов
│
├── data/
│   └── docs/               # Директория для документов (PDF, DOCX)
│
└── tests/                  # Unit-тесты
    ├── __init__.py
    ├── test_chunking.py
    ├── test_generation.py
    ├── test_retrieval.py
    ├── test_utils.py
    └── test_validation.py
```

## Архитектура

### Пайплайн обработки запроса

```
Пользовательский запрос
       │
       ▼
┌──────────────┐
│   Safety     │ ← Проверка на prompt-injection
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Retriever  │ ← Определение интента (таблица/формула/текст)
│  (Hybrid)    │ ← HyDE (опционально, кэш в SQLite)
│              │ ← Векторный поиск (FAISS) + BM25
│              │ ← Reciprocal Rank Fusion
│              │ ← Переранжирование (MiniLM-L-6-v2)
│              │ ← Фильтрация карантина
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Generation  │ ← Формирование промпта с контекстом
│   (LLM)      │ ← GigaChat / OpenAI / Ollama
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Validation  │ ← Проверка неуверенности (критично)
│              │ ← Проверка источников (предупреждение)
│              │ ← Семантическая согласованность (предупреждение)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Safety     │ ← Фильтрация конфиденциальных данных
└──────┬───────┘
       │
       ▼
   Ответ пользователю
```

### Слой данных

| Компонент | Технология | Назначение |
|-----------|-----------|------------|
| Векторный индекс | FAISS (cpu) | Семантический поиск по текстам и сводкам |
| BM25-индекс | rank_bm25 | Лексический поиск с фильтрацией стоп-слов |
| DocStore | SQLite | Полный контент таблиц/формул (ленивая загрузка) |
| Кэш HyDE | SQLite | Кэширование гипотетических ответов |
| Карантин | SQLite | Изоляция сомнительных чанков |

## Установка

### 1. Системные зависимости

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-rus poppler-utils

# Проверка Tesseract
tesseract --version
```

### 2. Python-зависимости

```bash
cd koib-v4.5
pip install -r requirements.txt
```

### 3. Конфигурация

```bash
cp .env.example .env
# Отредактируйте .env, указав ваши ключи API
```

Обязательные переменные для GigaChat:
```env
GIGACHAT_CREDENTIALS=Base64(client_id:client_secret)
```

## Использование

### Индексация документов

```bash
# Полная индексация
python main.py --ingest

# Инкрементальная (только новые файлы)
python main.py --ingest --incremental

# Указать директорию с документами
python main.py --ingest --docs-dir /path/to/docs
```

### Одиночный запрос

```bash
python main.py --query "Какие параметры у модели АИИС-001?"
python main.py --query "Покажи таблицу характеристик" --top-k 5
python main.py --query "Формула расчёта" --model АИИС-001
```

### Запуск API-сервера

```bash
# Стандартный запуск
python main.py --serve

# С параметрами
python main.py --serve --host 0.0.0.0 --port 8000 --vk-confirm-code ВАШ_КОД
```

Сервер будет доступен по адресу:
- API: `http://localhost:8000`
- Документация: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`
- VK Callback: `http://localhost:8000/vk_callback`

### Оценка качества

```bash
python main.py --evaluate questions.json
```

Формат `questions.json`:
```json
[
  {
    "question_id": "1",
    "question": "Какой вес у модели?",
    "reference": "Вес модели составляет 100 кг.",
    "category": "параметры"
  }
]
```

## Запуск тестов

```bash
pytest tests/ -v
```

## Оптимизации v4.5 (по сравнению с v4.3)

| Параметр | v4.3 | v4.5 | Обоснование |
|----------|------|------|-------------|
| Эмбеддинг-модель | e5-large (1.3 ГБ) | e5-small (33 МБ) | 40x меньше RAM |
| Квантизация | bitsandbytes (CUDA) | PyTorch dynamic (int8) | Работает на CPU |
| DocStore | JSON/pickle | SQLite | Ленивая загрузка, индексы |
| Кэш HyDE | JSON-файл | SQLite | Атомарность, быстрый поиск |
| OCR | EasyOCR + Tesseract | Только Tesseract | 0 RAM overhead |
| Сводки таблиц | LLM-генерация | Эвристики | 5-10с экономии на таблицу |
| VECTOR_SEARCH_K | 20 | 8 | Меньше кандидатов = быстрее |
| FINAL_TOP_K | 5 | 3 | Меньше контекста = быстрее LLM |
| MAX_TOKENS | 2048 | 700 | Короткие ответы = меньше задержка |
| Реранкер | bge-reranker-v2-m3 | MiniLM-L-6-v2 | Легче, быстрее на CPU |
| Устройство | auto (CUDA/CPU) | Принудительно CPU | Стабильность на VPS |
| API-сервер | — | FastAPI + VK Callback | Готов к деплою бота |

## Переменные окружения

Полный список переменных смотрите в файле `.env.example`.

## Лицензия

Проект разработан в рамках диссертационного исследования.
