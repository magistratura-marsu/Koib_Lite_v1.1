# -*- coding: utf-8 -*-
"""
Koib-V-4.5 — Централизованная конфигурация системы
=====================================================
Оптимизирована для сервера: 1 vCPU / 2 ГБ ОЗУ.
Все параметры управляются через переменные окружения (.env)
или принимают значения по умолчанию, пригодные для слабого VPS.

Ключевые отличия от v4.3:
  - Лёгкая модель эмбеддингов (e5-small вместо e5-large)
  - Принудительный CPU-режим (без CUDA)
  - Уменьшенные лимиты поиска (K=8, TOP_K=3)
  - Лёгкий реранкер (MiniLM-L-6-v2)
  - Жёсткие таймауты и лимиты токенов
  - HyDE отключён по умолчанию
"""
import os
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# Пути к директориям
# ═══════════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = Path(os.getenv("KOIB_DOCS_DIR", str(DATA_DIR / "docs")))
OUTPUT_DIR = Path(os.getenv("KOIB_OUTPUT_DIR", str(BASE_DIR / "output")))

# Поддиректории вывода
INDEX_DIR = OUTPUT_DIR / "index"
DOCSTORE_DIR = OUTPUT_DIR / "docstore"
FIGURES_DIR = OUTPUT_DIR / "figures"
LOGS_DIR = OUTPUT_DIR / "logs"
METADATA_DIR = OUTPUT_DIR / "metadata"

# ═══════════════════════════════════════════════════════════════
# Провайдеры LLM и эмбеддингов
# ═══════════════════════════════════════════════════════════════
# "gigachat" — LLM через GigaChat API (Сбер), рекомендуется
# "openai"   — LLM и/или эмбеддинги через OpenAI API
# "local"    — все модели загружаются локально (HuggingFace / Ollama)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gigachat")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")

# ═══════════════════════════════════════════════════════════════
# Эмбеддинг-модели
# ═══════════════════════════════════════════════════════════════
# e5-small — 33 МБ, 384 измерения, идеально для 2 ГБ ОЗУ
LOCAL_EMBEDDING_MODEL = os.getenv(
    "LOCAL_EMBEDDING_MODEL",
    "intfloat/multilingual-e5-small",
)
OPENAI_EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-small",
)

# Префиксы для instruction-tuned эмбеддингов (e5-серия)
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ═══════════════════════════════════════════════════════════════
# Параметры чанкинга
# ═══════════════════════════════════════════════════════════════
TEXT_CHUNK_SIZE = int(os.getenv("TEXT_CHUNK_SIZE", "800"))
TEXT_CHUNK_OVERLAP = int(os.getenv("TEXT_CHUNK_OVERLAP", "80"))
MIN_CHUNK_LENGTH = int(os.getenv("MIN_CHUNK_LENGTH", "50"))

# Таблицы и формулы НЕ дробятся — хранятся целиком в SQLite DocStore,
# в векторный индекс идёт только эвристическая сводка (summary).

# ═══════════════════════════════════════════════════════════════
# Параметры поиска (оптимизированы для 1 vCPU)
# ═══════════════════════════════════════════════════════════════
VECTOR_SEARCH_K = int(os.getenv("VECTOR_SEARCH_K", "8"))
BM25_SEARCH_K = int(os.getenv("BM25_SEARCH_K", "8"))
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "3"))
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.6"))

# ═══════════════════════════════════════════════════════════════
# Переранжирование (лёгкая модель)
# ═══════════════════════════════════════════════════════════════
USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)

# ═══════════════════════════════════════════════════════════════
# HyDE (отключён по умолчанию для экономии времени)
# ═══════════════════════════════════════════════════════════════
USE_HYDE = os.getenv("USE_HYDE", "false").lower() == "true"
BM25_USE_STOPWORDS = os.getenv("BM25_USE_STOPWORDS", "true").lower() == "true"

# ═══════════════════════════════════════════════════════════════
# LLM-параметры (жёсткие лимиты для стабильности)
# ═══════════════════════════════════════════════════════════════
# --- GigaChat ---
GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_CREDENTIALS", "")
GIGACHAT_MODEL = os.getenv("GIGACHAT_MODEL", "GigaChat")
GIGACHAT_TEMPERATURE = float(os.getenv("GIGACHAT_TEMPERATURE", "0.2"))
GIGACHAT_MAX_TOKENS = int(os.getenv("GIGACHAT_MAX_TOKENS", "700"))
GIGACHAT_TIMEOUT = int(os.getenv("GIGACHAT_TIMEOUT", "30"))
GIGACHAT_VERIFY_SSL = os.getenv("GIGACHAT_VERIFY_SSL", "false").lower() == "true"

# --- OpenAI ---
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "700"))

# --- Локальная LLM (Ollama / llama-cpp) ---
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "IlyaGusev/saiga_mistral_7b")
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://localhost:11434")

# ═══════════════════════════════════════════════════════════════
# Валидация ответов
# ═══════════════════════════════════════════════════════════════
VALIDATION_IGNORE_QUOTES = os.getenv("VALIDATION_IGNORE_QUOTES", "true").lower() == "true"
UNCERTAINTY_MIN_LENGTH = int(os.getenv("UNCERTAINTY_MIN_LENGTH", "50"))

# ═══════════════════════════════════════════════════════════════
# OCR и изображения
# ═══════════════════════════════════════════════════════════════
OCR_DPI = int(os.getenv("OCR_DPI", "300"))
OCR_MIN_TEXT_CHARS = int(os.getenv("OCR_MIN_TEXT_CHARS", "50"))
MIN_IMAGE_WIDTH = int(os.getenv("MIN_IMAGE_WIDTH", "80"))
MIN_IMAGE_HEIGHT = int(os.getenv("MIN_IMAGE_HEIGHT", "80"))

# ═══════════════════════════════════════════════════════════════
# Парсинг: выбор движка
# ═══════════════════════════════════════════════════════════════
# "pymupdf" — базовый парсер (всегда доступен, рекомендуется)
PARSING_ENGINE = os.getenv("PARSING_ENGINE", "pymupdf")

# ═══════════════════════════════════════════════════════════════
# FastAPI-сервер
# ═══════════════════════════════════════════════════════════════
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
VK_CONFIRM_CODE = os.getenv("VK_CONFIRM_CODE", "12345678")
VK_GROUP_ID = os.getenv("VK_GROUP_ID", "")
VK_ACCESS_TOKEN = os.getenv("VK_ACCESS_TOKEN", "")


def get_device() -> str:
    """
    Определить устройство вычислений.
    В v4.5 всегда возвращает 'cpu', так как система
    оптимизирована для работы на слабом VPS без GPU.
    """
    return "cpu"


def ensure_dirs() -> None:
    """Создать все необходимые директории при запуске."""
    for d in [
        DOCS_DIR, OUTPUT_DIR, INDEX_DIR,
        DOCSTORE_DIR, FIGURES_DIR, LOGS_DIR, METADATA_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)
