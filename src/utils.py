# -*- coding: utf-8 -*-
"""
Koib-V-4.8 — Общие утилиты + Память диалога
============================================
★ База: clean_text, text_hash, estimate_tokens, детекция моделей, парсинг
★ НОВОЕ: ConversationMemory (SQLite) + Query Rewriting для RAG-пайплайна
"""
import re
import uuid
import hashlib
import logging
import sqlite3
import asyncio
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from config import METADATA_DIR

logger = logging.getLogger("koib.utils")


# ═══════════════════════════════════════════════════════════════
# Базовые утилиты
# ═══════════════════════════════════════════════════════════════
def clean_text(text: str) -> str:
    """Очистка текста от мусорных символов."""
    if not text:
        return ""
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(
        r'[^\w\s\-\+\=\*\/\(\)\[\]\{\}\$\<\>\,\.\;\:\!\?\%\&\|\^\~`\"\'\\@\#№°'
        r'±≥≤≈×÷→←↑↓∈∑∫∂∇∞≈≠√∏∝∧∨¬⊂⊃⊆⊇∅∩∪'
        r'\u0400-\u04FF\u2116\n\r\t]',
        '', text, flags=re.UNICODE
    )
    lines = [line.strip() for line in text.split('\n')]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return '\n'.join(lines)


def text_hash(text: str) -> str:
    """SHA-256 хэш текста, укороченный до 16 символов."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def estimate_tokens(text: str) -> int:
    """
    Оценка количества токенов для русского (BPE).
    1 токен ≈ 2.5 символа (коэффициент 0.4).
    """
    if not text:
        return 0
    return max(1, int(len(text) * 0.4))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Обрезать текст до заданного количества токенов."""
    if not text:
        return ""
    max_chars = int(max_tokens * 2.5)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(' ', 1)[0] + "..."


def generate_unique_id(prefix: str = "") -> str:
    """Сгенерировать уникальный 12-символьный ID."""
    uid = uuid.uuid4().hex[:12]
    return f"{prefix}{uid}" if prefix else uid


# ═══════════════════════════════════════════════════════════════
# Детекция моделей КОИБ в тексте и именах файлов
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

_MODEL_PATTERNS = [
    re.compile(r'\b([A-ZА-Я]{2,}[\-\s]?\d{1,4}[A-ZА-Яа-я0-9\-/]*)\b'),
    re.compile(r'\b(модель\s+[A-ZА-Яа-я0-9\-/]+)\b', re.IGNORECASE),
]

_FILENAME_MODEL_PATTERNS = [
    re.compile(r'([A-Z]{2,}[\-]?\d{2,4}[A-Z0-9\-]*)'),
]


def detect_model_in_text(text: str) -> Tuple[str, float]:
    """
    Определить модель КОИБ в тексте.
    Возвращает (model_name, confidence).
    """
    if not text or len(text.strip()) < 5:
        return ("unknown", 0.0)

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

    for pattern in _MODEL_PATTERNS:
        match = pattern.search(text)
        if match:
            return (match.group(1).strip(), 0.3)

    return ("unknown", 0.0)


def detect_model_from_filename(filename: str) -> str:
    """Определить модель КОИБ по имени файла."""
    fn = filename.lower()
    for model_key, patterns in KOIB_MODEL_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, fn, re.IGNORECASE):
                return model_key

    for pattern in _FILENAME_MODEL_PATTERNS:
        match = pattern.search(filename)
        if match:
            return match.group(1).strip()

    return "unknown"


# ═══════════════════════════════════════════════════════════════
# Парсинг заголовков и подписей к рисункам
# ═══════════════════════════════════════════════════════════════
_FIGURE_CAPTION_PATTERNS = [
    re.compile(r'(?:Рис\.|Рисунок)\s*\d+[\.\:]?\s*(.+?)(?:\n|$)', re.IGNORECASE),
    re.compile(r'(?:Схема|схема)\s*\d+[\.\:]?\s*(.+?)(?:\n|$)', re.IGNORECASE),
    re.compile(r'(?:Чертёж|чертёж)\s*\d+[\.\:]?\s*(.+?)(?:\n|$)', re.IGNORECASE),
]


def find_figure_caption(text: str) -> str:
    """Найти подпись к рисунку в тексте."""
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
    """Извлечь заголовки из текста."""
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


# ═══════════════════════════════════════════════════════════════
# ★ НОВОЕ: Память диалога (SQLite для persistence)
# ═══════════════════════════════════════════════════════════════
class ConversationMemory:
    """
    Хранение истории диалога для Query Rewriting.
    SQLite для persistence между рестартами.
    """
    def __init__(self, db_path: Optional[Path] = None, max_history: int = 5):
        self.db_path = db_path or (METADATA_DIR / "conversation_memory.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_history = max_history
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._init_db()

    def _init_db(self):
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS history (
                    user_id TEXT,
                    role TEXT,
                    content TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            self.conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_user_timestamp
                ON history(user_id, timestamp)
            ''')

    async def add_message(self, user_id: str, role: str, content: str):
        """Добавить сообщение в историю (async wrapper)."""
        await asyncio.to_thread(self._sync_add, user_id, role, content)

    def _sync_add(self, user_id: str, role: str, content: str):
        with self.conn:
            self.conn.execute(
                'INSERT INTO history (user_id, role, content) VALUES (?, ?, ?)',
                (user_id, role, content),
            )

    async def get_history(self, user_id: str) -> List[Dict[str, str]]:
        """Получить последние N сообщений пользователя."""
        return await asyncio.to_thread(self._sync_get, user_id)

    def _sync_get(self, user_id: str) -> List[Dict[str, str]]:
        cur = self.conn.cursor()
        cur.execute('''
            SELECT role, content FROM history
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (user_id, self.max_history))
        rows = cur.fetchall()
        return [{'role': row[0], 'content': row[1]} for row in reversed(rows)]

    async def clear_history(self, user_id: str):
        """Очистить историю пользователя."""
        await asyncio.to_thread(self._sync_clear, user_id)

    def _sync_clear(self, user_id: str):
        with self.conn:
            self.conn.execute(
                'DELETE FROM history WHERE user_id = ?',
                (user_id,),
            )


# ═══════════════════════════════════════════════════════════════
# ★ НОВОЕ: Query Rewriting (разрешение местоимений)
# ═══════════════════════════════════════════════════════════════
QUERY_REWRITE_PROMPT = """История диалога:
{history}

Текущий вопрос пользователя:
{query}

Переформулируй текущий вопрос в самостоятельный запрос для поиска по технической документации.
Раскрой все местоимения ("она", "его", "этот параметр") на основе контекста диалога.
Верни ТОЛЬКО переформулированный вопрос, без пояснений."""


async def rewrite_query(
    query: str,
    history: List[Dict[str, str]],
    llm_client,
) -> str:
    """
    Переформулировать запрос с учётом истории диалога.
    Разрешает местоимения: "а какая у неё мощность?" -> "какая мощность у КОИБ-2017?"

    Args:
        query: Текущий запрос пользователя
        history: История диалога от ConversationMemory
        llm_client: Экземпляр LLMClient для генерации

    Returns:
        Переформулированный запрос (или оригинальный, если rewriting не удался)
    """
    if not history or len(history) < 2:
        return query

    history_text = '\n'.join(
        f"{msg['role'].capitalize()}: {msg['content'][:200]}"
        for msg in history[-4:]
    )
    prompt = QUERY_REWRITE_PROMPT.format(history=history_text, query=query)

    try:
        rewritten = await llm_client.generate_async(
            prompt, max_tokens=150, temperature=0.01
        )
        rewritten = rewritten.strip()
        if 10 < len(rewritten) < 500:
            logger.info(f"Query rewritten: '{query}' -> '{rewritten}'")
            return rewritten
    except Exception as exc:
        logger.warning(f"Query rewrite failed: {exc}")

    return query
