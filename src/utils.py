# -*- coding: utf-8 -*-
"""
Koib-V-4.5 — Общие утилиты
"""
import re
import uuid
import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger("koib.utils")


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    lines = [line.strip() for line in text.split('\n')]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return '\n'.join(lines)


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text) * 0.6))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    if not text:
        return ""
    max_chars = int(max_tokens / 0.6)
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def generate_unique_id(prefix: str = "") -> str:
    uid = uuid.uuid4().hex[:12]
    return f"{prefix}{uid}" if prefix else uid


# ═══════════════════════════════════════════════════════════════
# ★ ВОЗВРАЩЕНО: специализированные паттерны КОИБ + уверенность
# ═══════════════════════════════════════════════════════════════
KNOWN_MODELS = {"koib2010", "koib2017a", "koib2017b"}

KOIB_MODEL_PATTERNS: Dict[str, List[str]] = {
    "koib2010": [
        r"КОИБ[-\s]?2010", r"КОИБ\s*2010", r"0912054",
        r"PRINT_KOIB2010", r"2010.*руководство",
        r"модель\s*17404049\.438900\.001",
    ],
    "koib2017a": [
        r"КОИБ[-\s]?2017\s*[АA]", r"КОИБ[-\s]?2017А",
        r"модель\s*17404049\.5013009\.008-01",
        r"17404049\.5013009", r"PRINT_KOIB2017[АA]",
    ],
    "koib2017b": [
        r"КОИБ[-\s]?2017\s*[БB]", r"КОИБ[-\s]?2017Б",
        r"БАВУ\.201119", r"0912053", r"PRINT_KOIB2017[БB]",
    ],
}

# Общие паттерны для устройств (fallback)
_MODEL_PATTERNS = [
    re.compile(r'\b([A-ZА-Я]{2,}[\-\s]?\d{1,4}[A-ZА-Яа-я0-9\-/]*)\b'),
    re.compile(r'\b(модель\s+[A-ZА-Яа-я0-9\-/]+)\b', re.IGNORECASE),
    re.compile(r'\b(тип\s+[A-ZА-Яа-я0-9\-/]+)\b', re.IGNORECASE),
    re.compile(r'\b(марка\s+[A-ZА-Яа-я0-9\-/]+)\b', re.IGNORECASE),
]

_FILENAME_MODEL_PATTERNS = [
    re.compile(r'([A-Z]{2,}[\-]?\d{2,4}[A-Z0-9\-]*)'),
]


def detect_model_in_text(text: str) -> Tuple[str, float]:
    """
    ★ ВОЗВРАЩЕНО: возвращает (модель, уверенность).
    Сначала проверяет специфичные паттерны КОИБ (высокая уверенность),
    затем общие паттерны устройств (низкая уверенность).
    """
    if not text or len(text.strip()) < 5:
        return ("unknown", 0.0)

    # 1. Специфичные КОИБ-паттерны
    scores: Dict[str, float] = {}
    for model_key, patterns in KOIB_MODEL_PATTERNS.items():
        match_count = 0
        for pat in patterns:
            if re.findall(pat, text, re.IGNORECASE):
                match_count += 1
        if match_count > 0:
            scores[model_key] = match_count

    if scores:
        best = max(scores, key=scores.get)
        confidence = min(scores[best] / 3.0, 1.0)
        return (best, round(confidence, 3))

    # 2. Общие паттерны устройств (низкая уверенность)
    for pattern in _MODEL_PATTERNS:
        match = pattern.search(text)
        if match:
            return (match.group(1).strip(), 0.3)

    return ("unknown", 0.0)


def detect_model_from_filename(filename: str) -> str:
    # Специфичные КОИБ-паттерны
    fn = filename.lower()
    for model_key, patterns in KOIB_MODEL_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, fn, re.IGNORECASE):
                return model_key
    # Общие паттерны
    for pattern in _FILENAME_MODEL_PATTERNS:
        match = pattern.search(filename)
        if match:
            return match.group(1).strip()
    return "unknown"


_FIGURE_CAPTION_PATTERNS = [
    re.compile(r'(?:Рис\.|Рисунок|рис\.|рисунок)\s*\d+[\.\:]?\s*(.+?)(?:\n|$)', re.IGNORECASE),
    re.compile(r'(?:Схема|схема)\s*\d+[\.\:]?\s*(.+?)(?:\n|$)', re.IGNORECASE),
    re.compile(r'(?:Чертёж|чертёж)\s*\d+[\.\:]?\s*(.+?)(?:\n|$)', re.IGNORECASE),
]


def find_figure_caption(text: str) -> str:
    for pattern in _FIGURE_CAPTION_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    return ""


_HEADING_PATTERNS = [
    re.compile(r'^(\d+(?:\.\d+)*)\s+(.+)$'),
    re.compile(r'^([А-ЯЁ][А-ЯЁ\s]{2,})$'),
    re.compile(r'^([А-ЯЁ][а-яё].{3,})$'),
]


def extract_headings(text: str) -> List[str]:
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