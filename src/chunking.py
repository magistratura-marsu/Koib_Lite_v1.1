# -*- coding: utf-8 -*-
"""
Koib-V-4.5 — Модуль умного чанкинга
======================================
Разбиение извлечённых элементов на чанки с учётом типа контента:
- Текст: семантическое разбиение ~800 токенов с перекрытием 10%
- Таблицы: не дробятся, хранятся целиком в SQLite DocStore
- Формулы: не дробятся, хранятся целиком в SQLite DocStore
- Рисунки: описание/подпись хранится как отдельный чанк

В v4.5 LLM-сводки полностью заменены эвристическими:
генерация через LLM отнимала 5-10 секунд на таблицу при
пакетной загрузке, что недопустимо для VPS с 1 vCPU.
"""
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from langchain_core.documents import Document

from .parsing import DocumentElement
from .utils import estimate_tokens, generate_unique_id
from config import TEXT_CHUNK_SIZE, TEXT_CHUNK_OVERLAP, MIN_CHUNK_LENGTH

logger = logging.getLogger("koib.chunking")


# ═══════════════════════════════════════════════════════════════
# Структура чанка
# ═══════════════════════════════════════════════════════════════
@dataclass
class Chunk:
    """
    Чанк документа, готовый к индексации.

    Attributes:
        chunk_id:     Уникальный идентификатор чанка
        content:      Текст чанка / эвристическая сводка таблицы
        full_content: Полный контент (для таблиц/формул из DocStore)
        chunk_type:   Тип чанка: text | table | formula | figure
        metadata:     Метаданные чанка (источник, страница, заголовок)
    """
    chunk_id: str
    content: str
    full_content: Optional[str] = None
    chunk_type: str = "text"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_langchain_doc(self) -> Document:
        """Конвертировать в LangChain Document для векторного индекса."""
        return Document(
            page_content=self.content,
            metadata={
                "chunk_id": self.chunk_id,
                "chunk_type": self.chunk_type,
                **self.metadata,
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        """Сериализовать чанк в словарь."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "full_content": self.full_content,
            "chunk_type": self.chunk_type,
            "metadata": self.metadata,
        }


# ═══════════════════════════════════════════════════════════════
# Семантическое разбиение текста
# ═══════════════════════════════════════════════════════════════
def _split_text_semantic(
    text: str,
    max_tokens: int = TEXT_CHUNK_SIZE,
    overlap_tokens: int = TEXT_CHUNK_OVERLAP,
) -> List[str]:
    """
    Семантическое разбиение текста на чанки.

    Стратегия:
      1. Делим текст по двойным переносам строк (абзацы)
      2. Группируем абзацы, пока не превысим max_tokens
      3. Добавляем overlap из последних абзацев предыдущего чанка

    Args:
        text:          Исходный текст
        max_tokens:    Максимальное количество токенов в чанке
        overlap_tokens: Перекрытие между чанками (в токенах)

    Returns:
        Список текстовых чанков
    """
    if not text or len(text.strip()) < MIN_CHUNK_LENGTH:
        return []

    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    chunks: List[str] = []
    current_parts: List[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = estimate_tokens(para)

        if current_tokens + para_tokens > max_tokens and current_parts:
            # Сохраняем текущий чанк
            chunks.append('\n\n'.join(current_parts))

            # Overlap: берём последние абзацы, которые влезают
            overlap_parts: List[str] = []
            overlap_tok = 0
            for p in reversed(current_parts):
                pt = estimate_tokens(p)
                if overlap_tok + pt > overlap_tokens:
                    break
                overlap_parts.insert(0, p)
                overlap_tok += pt

            current_parts = overlap_parts
            current_tokens = overlap_tok

        current_parts.append(para)
        current_tokens += para_tokens

    # Последний чанк
    if current_parts:
        chunk_text = '\n\n'.join(current_parts)
        if estimate_tokens(chunk_text) >= MIN_CHUNK_LENGTH // 4:
            chunks.append(chunk_text)

    return chunks


# ═══════════════════════════════════════════════════════════════
# Эвристические сводки (без LLM — работают за миллисекунды)
# ═══════════════════════════════════════════════════════════════
def _generate_table_summary(table_markdown: str, metadata: Dict) -> str:
    """
    Эвристическая сводка таблицы для индексации.

    Извлекает:
      - Количество строк и столбцов
      - Заголовки столбцов
      - Первые 2 строки данных как пример

    Сводка нужна для векторного поиска: по ней пользователь находит
    таблицу, а полный контент берётся из SQLite DocStore.
    """
    lines = table_markdown.strip().split('\n')
    header_line = lines[0] if lines else ""
    num_rows = metadata.get("num_rows", 0)
    num_cols = metadata.get("num_cols", 0)

    headers = [h.strip() for h in header_line.split('|') if h.strip()]

    summary_parts = [f"Таблица ({num_rows} строк, {num_cols} столбцов)."]
    if headers:
        summary_parts.append(f"Столбцы: {', '.join(headers[:10])}.")

    # Первые 2 строки данных
    data_lines = [l for l in lines[2:] if l.strip() and '---' not in l][:2]
    if data_lines:
        summary_parts.append("Пример данных:")
        for dl in data_lines:
            cells = [c.strip() for c in dl.split('|') if c.strip()]
            summary_parts.append("  " + " | ".join(cells[:5]))

    return " ".join(summary_parts)


def _generate_formula_summary(formula_content: str, metadata: Dict) -> str:
    """
    Эвристическая сводка формулы для индексации.

    Определяет тип формулы и добавляет первые 200 символов содержания.
    """
    formula_type = metadata.get("formula_type", "unknown")
    type_desc = {
        "latex_inline": "Формула (LaTeX, строковая)",
        "latex_block": "Формула (LaTeX, блочная)",
        "suspected_formula": "Подозреваемая формула",
        "unknown": "Формула",
    }.get(formula_type, "Формула")

    content_preview = formula_content[:200]
    return f"{type_desc}: {content_preview}"


# ═══════════════════════════════════════════════════════════════
# Основной класс чанкера
# ═══════════════════════════════════════════════════════════════
class SmartChunker:
    """
    Умный чанкер с разделением по типам контента.

    - Текст разбивается на чанки ~800 токенов с перекрытием
    - Таблицы и формулы хранятся целиком с эвристической сводкой
    - Рисунки получают описание/подпись как сводку
    """

    def __init__(
        self,
        chunk_size: int = TEXT_CHUNK_SIZE,
        chunk_overlap: int = TEXT_CHUNK_OVERLAP,
        min_chunk_length: int = MIN_CHUNK_LENGTH,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length

    def chunk_elements(self, elements: List[DocumentElement]) -> List[Chunk]:
        """
        Разбить список DocumentElement на чанки.

        Обрабатывает элементы последовательно, накапливая текстовые
        элементы в буфер и сбрасывая его при встрече структурированного
        элемента (таблица/формула/рисунок).

        Args:
            elements: Список DocumentElement из модуля парсинга

        Returns:
            Список Chunk, готовых к индексации
        """
        chunks: List[Chunk] = []
        text_buffer: List[DocumentElement] = []
        current_heading = ""

        for element in elements:
            # Обновляем текущий заголовок
            if element.element_type == "heading":
                current_heading = element.content

            if element.element_type in ("table", "formula", "figure"):
                # Сбрасываем буфер текста
                if text_buffer:
                    chunks.extend(self._chunk_text_buffer(text_buffer, current_heading))
                    text_buffer = []
                # Структурированный элемент — отдельный чанк
                chunks.append(self._chunk_structured_element(element, current_heading))
            else:
                # Накапливаем текстовые элементы
                text_buffer.append(element)

        # Сброс оставшегося буфера
        if text_buffer:
            chunks.extend(self._chunk_text_buffer(text_buffer, current_heading))

        logger.info(f"Создано {len(chunks)} чанков из {len(elements)} элементов")
        return chunks

    def _chunk_text_buffer(
        self,
        elements: List[DocumentElement],
        heading: str,
    ) -> List[Chunk]:
        """Разбить буфер текстовых элементов на чанки."""
        combined = '\n\n'.join(e.content for e in elements if e.content.strip())

        if not combined or len(combined.strip()) < self.min_chunk_length:
            return []

        text_chunks = _split_text_semantic(
            combined,
            max_tokens=self.chunk_size,
            overlap_tokens=self.chunk_overlap,
        )

        chunks: List[Chunk] = []
        source = elements[0].source if elements else ""
        page = elements[0].page if elements else 0
        model = elements[0].model if elements else "unknown"

        for i, text in enumerate(text_chunks):
            text = text.strip()
            if len(text) < self.min_chunk_length:
                continue

            chunk_id = generate_unique_id(prefix=f"txt_{source}_{page}_")
            chunks.append(Chunk(
                chunk_id=chunk_id,
                content=text,
                full_content=None,
                chunk_type="text",
                metadata={
                    "source": source,
                    "page": page,
                    "heading": heading,
                    "model": model,
                    "chunk_index": i,
                },
            ))

        return chunks

    def _chunk_structured_element(
        self,
        element: DocumentElement,
        heading: str,
    ) -> Chunk:
        """Создать чанк для структурированного элемента."""
        if element.element_type == "table":
            summary = _generate_table_summary(element.content, element.metadata)
        elif element.element_type == "formula":
            summary = _generate_formula_summary(element.content, element.metadata)
        else:
            summary = element.content

        chunk_id = generate_unique_id(
            prefix=f"{element.element_type}_{element.source}_{element.page}_"
        )

        return Chunk(
            chunk_id=chunk_id,
            content=summary,
            full_content=element.content,
            chunk_type=element.element_type,
            metadata={
                "source": element.source,
                "page": element.page,
                "heading": heading or element.heading,
                "model": element.model,
                "element_id": element.element_id,
                **element.metadata,
            },
        )
