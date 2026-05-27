# -*- coding: utf-8 -*-
"""
Koib-V-4.5 — Общие утилиты
============================
Хэширование, очистка текста, оценка токенов, извлечение моделей,
поиск подписей к рисункам, определение заголовков, генерация ID.
"""
import re
import uuid
import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger("koib.utils")


# ═══════════════════════════════════════════════════════════════
# Очистка текста
# ═══════════════════════════════════════════════════════════════
def clean_text(text: str) -> str:
    """
    Очистить текст от лишних пробелов, спецсимволов и артефактов.

    Выполняет:
      - Удаление непечатных символов (кроме переносов строк)
      - Схлопывание множественных пробелов
      - Удаление пробелов в начале/конце строк
      - Удаление пустых строк в начале и конце
    """
    if not text:
        return ""
    # Удаляем непечатные символы, кроме переноса строки и табуляции
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # Схлопываем множественные пробелы
    text = re.sub(r'[ \t]+', ' ', text)
    # Удаляем пробелы в начале/конце каждой строки
    lines = [line.strip() for line in text.split('\n')]
    # Удаляем пустые строки в начале и конце
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════
# Хэширование
# ═══════════════════════════════════════════════════════════════
def text_hash(text: str) -> str:
    """
    SHA-256 хэш текста. Используется для идентификаторов
    чанков и элементов, а также для кэширования.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ═══════════════════════════════════════════════════════════════
# Оценка токенов
# ═══════════════════════════════════════════════════════════════
def estimate_tokens(text: str) -> int:
    """
    Приблизительная оценка количества токенов для русского текста.
    Эмпирическое правило: ~0.6 токена на символ для русского языка
    (русские слова длиннее, но токенизатор дробит их на подслова).
    Минимум 1 токен для непустой строки.
    """
    if not text:
        return 0
    return max(1, int(len(text) * 0.6))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """
    Обрезать текст до приблизительного лимита токенов.
    Работает быстрее, чем точная токенизация, и достаточен
    для целей ограничения контекста.
    """
    if not text:
        return ""
    max_chars = int(max_tokens / 0.6)
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


# ═══════════════════════════════════════════════════════════════
# Генерация уникальных ID
# ═══════════════════════════════════════════════════════════════
def generate_unique_id(prefix: str = "") -> str:
    """
    Генерация уникального идентификатора на основе UUID4.
    Опциональный префикс добавляется через подчёркивание.
    """
    uid = uuid.uuid4().hex[:12]
    return f"{prefix}{uid}" if prefix else uid


# ═══════════════════════════════════════════════════════════════
# Детекция модели устройства
# ═══════════════════════════════════════════════════════════════
# Шаблоны для поиска названия модели в тексте документа
_MODEL_PATTERNS = [
    re.compile(r'\b([A-ZА-Я]{2,}[\-\s]?\d{1,4}[A-ZА-Яа-я0-9\-/]*)\b'),
    re.compile(r'\b(модель\s+[A-ZА-Яа-я0-9\-/]+)\b', re.IGNORECASE),
    re.compile(r'\b(тип\s+[A-ZА-Яа-я0-9\-/]+)\b', re.IGNORECASE),
    re.compile(r'\b(марка\s+[A-ZА-Яа-я0-9\-/]+)\b', re.IGNORECASE),
]

_FILENAME_MODEL_PATTERNS = [
    re.compile(r'([A-Z]{2,}[\-]?\d{2,4}[A-Z0-9\-]*)'),
]


def detect_model_in_text(text: str) -> str:
    """
    Попытка определить название модели/устройства в тексте.
    Возвращает первое совпадение или 'unknown'.
    """
    for pattern in _MODEL_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    return "unknown"


def detect_model_from_filename(filename: str) -> str:
    """
    Попытка определить название модели из имени файла.
    Например, 'ПАСПОРТ_АИИС-001.pdf' -> 'АИИС-001'.
    """
    for pattern in _FILENAME_MODEL_PATTERNS:
        match = pattern.search(filename)
        if match:
            return match.group(1).strip()
    return "unknown"


# ═══════════════════════════════════════════════════════════════
# Поиск подписей к рисункам
# ═══════════════════════════════════════════════════════════════
_FIGURE_CAPTION_PATTERNS = [
    re.compile(r'(?:Рис\.|Рисунок|рис\.|рисунок)\s*\d+[\.\:]?\s*(.+?)(?:\n|$)', re.IGNORECASE),
    re.compile(r'(?:Схема|схема)\s*\d+[\.\:]?\s*(.+?)(?:\n|$)', re.IGNORECASE),
    re.compile(r'(?:Чертёж|чертёж)\s*\d+[\.\:]?\s*(.+?)(?:\n|$)', re.IGNORECASE),
]


def find_figure_caption(text: str) -> str:
    """
    Найти подпись к рисунку/схеме/чертежу в тексте.
    Возвращает подпись или пустую строку.
    """
    for pattern in _FIGURE_CAPTION_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    return ""


# ═══════════════════════════════════════════════════════════════
# Извлечение заголовков
# ═══════════════════════════════════════════════════════════════
_HEADING_PATTERNS = [
    re.compile(r'^(\d+(?:\.\d+)*)\s+(.+)$'),         # "1.2.3 Заголовок"
    re.compile(r'^([А-ЯЁ][А-ЯЁ\s]{2,})$'),            # "ВВЕДЕНИЕ"
    re.compile(r'^([А-ЯЁ][а-яё].{3,})$'),             # "Общие сведения"
]


def extract_headings(text: str) -> List[str]:
    """
    Извлечь строки, похожие на заголовки, из текста.
    Возвращает список найденных заголовков.
    """
    headings = []
    for line in text.split('\n'):
        line_stripped = line.strip()
        if not line_stripped or len(line_stripped) < 4:
            continue
        for pattern in _HEADING_PATTERNS:
            if pattern.match(line_stripped):
                headings.append(line_stripped)
                break
    return headings
