# -*- coding: utf-8 -*-
"""
Koib-V-4.5 — Модуль гибридного поиска
========================================
Объединяет векторный поиск (FAISS), лексический (BM25) и
переранжирование (CrossEncoder) для получения наиболее
релевантных фрагментов документации.

Ключевые отличия от v4.3:
  - SQLite-кэш ответов HyDE (вместо JSON-файла)
  - Уменьшенные K=8 и TOP_K=3 для скорости
  - Лёгкий реранкер MiniLM-L-6-v2
  - Определение интента запроса (таблица/формула/рисунок)
  - Фильтрация карантинных чанков
"""
import json
import logging
import sqlite3
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from src.indexing import IndexBuilder
from config import (
    QUERY_PREFIX, VECTOR_SEARCH_K, BM25_SEARCH_K, FINAL_TOP_K,
    HYBRID_ALPHA, USE_RERANKER, RERANKER_MODEL,
    USE_HYDE, EMBEDDING_PROVIDER, METADATA_DIR,
)

logger = logging.getLogger("koib.retrieval")


# ═══════════════════════════════════════════════════════════════
# SQLite Кэш ответов HyDE (мгновенная выдача повторных запросов)
# ═══════════════════════════════════════════════════════════════
class ResponseCache:
    """
    SQLite-кэш для гипотетических ответов HyDE.

    При повторном запросе с тем же текстом система мгновенно
    извлекает ранее сгенерированный гипотетический ответ,
    экономя 5-10 секунд на вызове LLM.
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = path or METADATA_DIR / "response_cache.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS cache (
                    query_hash TEXT PRIMARY KEY,
                    hypothetical TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

    def _hash(self, text: str) -> str:
        """MD5-хэш нормализованного текста запроса."""
        return hashlib.md5(text.lower().strip().encode()).hexdigest()

    def get(self, query: str) -> Optional[str]:
        """Получить кэшированный гипотетический ответ."""
        cur = self.conn.cursor()
        cur.execute(
            'SELECT hypothetical FROM cache WHERE query_hash = ?',
            (self._hash(query),),
        )
        row = cur.fetchone()
        return row[0] if row else None

    def set(self, query: str, hypothetical: str) -> None:
        """Сохранить гипотетический ответ в кэш."""
        with self.conn:
            self.conn.execute(
                'INSERT OR REPLACE INTO cache (query_hash, hypothetical) VALUES (?, ?)',
                (self._hash(query), hypothetical),
            )

    def clear(self) -> None:
        """Очистить весь кэш."""
        with self.conn:
            self.conn.execute('DELETE FROM cache')


# ═══════════════════════════════════════════════════════════════
# Структура результата поиска
# ═══════════════════════════════════════════════════════════════
@dataclass
class RetrievalResult:
    """
    Результат поиска одного фрагмента.

    Attributes:
        chunk_id:     Идентификатор чанка
        content:      Сводный текст (или основной текст для text-чанков)
        full_content: Полный контент (из DocStore, для таблиц/формул)
        score:        Оценка релевантности
        source:       Имя файла-источника
        page:         Номер страницы
        heading:      Заголовок раздела
        model:        Модель устройства
        chunk_type:   Тип чанка
        metadata:     Полные метаданные
    """
    chunk_id: str
    content: str
    full_content: Optional[str] = None
    score: float = 0.0
    source: str = ""
    page: int = 0
    heading: str = ""
    model: str = "unknown"
    chunk_type: str = "text"
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_context_string(self) -> str:
        """
        Форматировать результат для подачи в LLM-промпт.

        Включает метаданные источника, заголовок раздела
        и полный контент (или основной, если полного нет).
        """
        parts = [f"[Документ: {self.source}, стр. {self.page}]"]
        if self.heading:
            parts.append(f"Раздел: {self.heading}")

        display_content = self.full_content or self.content

        if self.chunk_type == "table":
            parts.append(f"ТАБЛИЦА:\n{display_content}")
        elif self.chunk_type == "formula":
            parts.append(f"ФОРМУЛА: {display_content}")
        elif self.chunk_type == "figure":
            parts.append(f"РИСУНОК: {display_content}")
        else:
            parts.append(display_content)

        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════
# Определение интента запроса
# ═══════════════════════════════════════════════════════════════
TABLE_KEYWORDS = {"таблиц", "значени", "параметр", "сводк", "данные", "показател"}
FORMULA_KEYWORDS = {"формул", "вычислен", "расчёт", "уравнен", "коэффициент"}
FIGURE_KEYWORDS = {"схем", "рисунок", "диаграмм", "чертёж", "график"}


def _detect_query_intent(query: str) -> Dict[str, float]:
    """
    Определить интент запроса по ключевым словам.

    Возвращает словарь весов для каждого типа контента:
      - table, formula, figure, text

    Если запрос содержит ключевые слова таблиц, вес table
    повышается, а вес text снижается, но остаётся не ниже 0.3.
    """
    query_lower = query.lower()
    intent = {"table": 0.0, "formula": 0.0, "figure": 0.0, "text": 1.0}

    table_hits = sum(1 for kw in TABLE_KEYWORDS if kw in query_lower)
    formula_hits = sum(1 for kw in FORMULA_KEYWORDS if kw in query_lower)
    figure_hits = sum(1 for kw in FIGURE_KEYWORDS if kw in query_lower)

    total_hits = table_hits + formula_hits + figure_hits
    if total_hits > 0:
        intent["table"] = min(table_hits / 2.0, 1.0)
        intent["formula"] = min(formula_hits / 2.0, 1.0)
        intent["figure"] = min(figure_hits / 2.0, 1.0)
        intent["text"] = max(0.3, 1.0 - total_hits * 0.2)

    return intent


# ═══════════════════════════════════════════════════════════════
# Гибридный поисковик
# ═══════════════════════════════════════════════════════════════
class HybridRetriever:
    """
    Гибридный поисковик: FAISS (векторный) + BM25 (лексический)
    с переранжированием и HyDE.

    Pipeline:
      1. Определение интента запроса
      2. (Опционально) HyDE — генерация гипотетического ответа
      3. Векторный поиск по двум индексам (текст + сводки)
      4. BM25-поиск
      5. Reciprocal Rank Fusion (объединение результатов)
      6. (Опционально) Переранжирование CrossEncoder
      7. Подгрузка полного контента из DocStore
    """

    def __init__(self, index_builder: Optional[IndexBuilder] = None):
        self.index_builder = index_builder or IndexBuilder()
        self.index_builder.load()
        self._reranker = None
        self._cache = ResponseCache()

    def _get_reranker(self):
        """
        Ленивая загрузка переранжировщика.

        CrossEncoder загружается только при первом использовании,
        чтобы не расходовать RAM, если переранжирование отключено.
        """
        if self._reranker is not None:
            return self._reranker
        if not USE_RERANKER:
            return None
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Загрузка переранжировщика: {RERANKER_MODEL}")
            self._reranker = CrossEncoder(RERANKER_MODEL)
            return self._reranker
        except Exception as exc:
            logger.warning(f"Не удалось загрузить переранжировщик: {exc}")
            return None

    def search(
        self,
        query: str,
        k: int = FINAL_TOP_K,
        model_filter: str = "",
        use_hyde: Optional[bool] = None,
    ) -> List[RetrievalResult]:
        """
        Выполнить гибридный поиск по запросу.

        Args:
            query:        Поисковый запрос пользователя
            k:            Количество финальных результатов
            model_filter: Фильтр по модели устройства
            use_hyde:     Использовать ли HyDE (None = по конфигу)

        Returns:
            Список RetrievalResult, отсортированный по релевантности
        """
        # Шаг 1: Интент запроса
        intent = _detect_query_intent(query)

        # Шаг 2: HyDE (опционально)
        search_query = query
        if use_hyde if use_hyde is not None else USE_HYDE:
            hyde_result = self._apply_hyde(query)
            if hyde_result:
                search_query = hyde_result

        # Шаг 3: Векторный поиск
        vector_results = self._vector_search(search_query, intent, model_filter)

        # Шаг 4: BM25-поиск
        bm25_results = self._bm25_search(query, model_filter)

        # Шаг 5: Reciprocal Rank Fusion
        fused = self._reciprocal_rank_fusion(vector_results, bm25_results)

        # Фильтрация карантинных чанков
        try:
            from .quarantine import filter_quarantined_chunks
            fused = filter_quarantined_chunks(fused)
        except Exception:
            pass

        # Шаг 6: Переранжирование (опционально)
        if USE_RERANKER and len(fused) > k:
            reranker = self._get_reranker()
            if reranker:
                fused = self._rerank(query, fused, reranker)

        # Шаг 7: Ограничение количества результатов
        results = fused[:k]

        # Подгрузка полного контента из DocStore
        for r in results:
            if r.chunk_type in ("table", "formula", "figure") and r.full_content is None:
                full = self.index_builder.docstore.get_content(r.chunk_id)
                if full:
                    r.full_content = full

        return results

    def _vector_search(
        self,
        query: str,
        intent: Dict[str, float],
        model_filter: str = "",
    ) -> List[RetrievalResult]:
        """
        Векторный поиск по двум FAISS-индексам.

        Выполняет поиск по текстовому индексу и индексу сводок,
        затем объединяет результаты с учётом интента запроса.
        """
        results: List[RetrievalResult] = []
        search_text = f"{QUERY_PREFIX}{query}" if QUERY_PREFIX else query

        # Поиск по текстовому индексу
        if self.index_builder.text_vectorstore is not None:
            try:
                docs = self.index_builder.text_vectorstore.similarity_search_with_score(
                    search_text,
                    k=VECTOR_SEARCH_K,
                )
                for doc, score in docs:
                    chunk_type = doc.metadata.get("chunk_type", "text")
                    r = RetrievalResult(
                        chunk_id=doc.metadata.get("chunk_id", ""),
                        content=doc.page_content,
                        score=float(score) * intent.get("text", 1.0),
                        source=doc.metadata.get("source", ""),
                        page=doc.metadata.get("page", 0),
                        heading=doc.metadata.get("heading", ""),
                        model=doc.metadata.get("model", "unknown"),
                        chunk_type=chunk_type,
                        metadata=doc.metadata,
                    )
                    if not model_filter or r.model == model_filter:
                        results.append(r)
            except Exception as exc:
                logger.warning(f"Ошибка векторного поиска по текстам: {exc}")

        # Поиск по индексу сводок
        if self.index_builder.summary_vectorstore is not None:
            try:
                docs = self.index_builder.summary_vectorstore.similarity_search_with_score(
                    search_text,
                    k=VECTOR_SEARCH_K,
                )
                for doc, score in docs:
                    chunk_type = doc.metadata.get("chunk_type", "text")
                    type_weight = intent.get(chunk_type, 1.0)
                    r = RetrievalResult(
                        chunk_id=doc.metadata.get("chunk_id", ""),
                        content=doc.page_content,
                        score=float(score) * type_weight,
                        source=doc.metadata.get("source", ""),
                        page=doc.metadata.get("page", 0),
                        heading=doc.metadata.get("heading", ""),
                        model=doc.metadata.get("model", "unknown"),
                        chunk_type=chunk_type,
                        metadata=doc.metadata,
                    )
                    if not model_filter or r.model == model_filter:
                        results.append(r)
            except Exception as exc:
                logger.warning(f"Ошибка векторного поиска по сводкам: {exc}")

        return results

    def _bm25_search(
        self,
        query: str,
        model_filter: str = "",
    ) -> List[RetrievalResult]:
        """
        BM25-лексический поиск.

        Использует токенизацию с фильтрацией стоп-слов
        для повышения точности поиска на русском языке.
        """
        results: List[RetrievalResult] = []

        bm25_hits = self.index_builder.bm25.search(query, k=BM25_SEARCH_K)
        for metadata, score in bm25_hits:
            r = RetrievalResult(
                chunk_id=metadata.get("chunk_id", ""),
                content=metadata.get("content", ""),
                score=score,
                source=metadata.get("source", ""),
                page=metadata.get("page", 0),
                heading=metadata.get("heading", ""),
                model=metadata.get("model", "unknown"),
                chunk_type=metadata.get("chunk_type", "text"),
                metadata=metadata,
            )
            if not model_filter or r.model == model_filter:
                results.append(r)

        return results

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[RetrievalResult],
        bm25_results: List[RetrievalResult],
        k_rrf: int = 60,
    ) -> List[RetrievalResult]:
        """
        Reciprocal Rank Fusion (RRF) для объединения результатов.

        Формула: RRF_score = sum(1 / (k_rrf + rank_i))

        Args:
            vector_results: Результаты векторного поиска
            bm25_results:   Результаты BM25-поиска
            k_rrf:          Параметр сглаживания RRF (по умолчанию 60)

        Returns:
            Объединённый список RetrievalResult
        """
        chunk_scores: Dict[str, float] = {}
        chunk_map: Dict[str, RetrievalResult] = {}

        # Векторные результаты
        for rank, r in enumerate(vector_results, 1):
            if r.chunk_id not in chunk_scores:
                chunk_scores[r.chunk_id] = 0.0
                chunk_map[r.chunk_id] = r
            chunk_scores[r.chunk_id] += HYBRID_ALPHA / (k_rrf + rank)

        # BM25-результаты
        for rank, r in enumerate(bm25_results, 1):
            if r.chunk_id not in chunk_scores:
                chunk_scores[r.chunk_id] = 0.0
                chunk_map[r.chunk_id] = r
            chunk_scores[r.chunk_id] += (1 - HYBRID_ALPHA) / (k_rrf + rank)

        # Сортировка по итоговому RRF-скору
        sorted_ids = sorted(chunk_scores.keys(), key=lambda x: chunk_scores[x], reverse=True)

        results = []
        for cid in sorted_ids:
            r = chunk_map[cid]
            r.score = chunk_scores[cid]
            results.append(r)

        return results

    def _rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        reranker,
    ) -> List[RetrievalResult]:
        """
        Переранжирование результатов через CrossEncoder.

        Модель оценивает пару (запрос, документ) и возвращает
        оценку релевантности, которая используется для пересортировки.
        """
        try:
            pairs = [(query, r.content) for r in results]
            scores = reranker.predict(pairs)

            for r, score in zip(results, scores):
                r.score = float(score)

            results.sort(key=lambda x: x.score, reverse=True)
            return results
        except Exception as exc:
            logger.warning(f"Ошибка переранжирования: {exc}")
            return results

    def _apply_hyde(self, query: str) -> Optional[str]:
        """
        HyDE (Hypothetical Document Embeddings).

        Генерирует гипотетический ответ на запрос через LLM,
        затем использует его для векторного поиска вместо
        оригинального запроса. Это улучшает качество поиска,
        так как ответы и документы лежат в одном семантическом
        пространстве.

        Результат кэшируется в SQLite для мгновенного
        извлечения при повторных запросах.
        """
        # Проверяем кэш
        cached = self._cache.get(query)
        if cached:
            return cached

        try:
            from src.generation import LLMClient
            client = LLMClient()
            hypothetical = client.generate(
                f"Ответь кратко на вопрос, как если бы ты был экспертом "
                f"по технической документации:\n{query}",
                max_tokens=300,
            )
            if hypothetical and len(hypothetical) > 20:
                self._cache.set(query, hypothetical)
                return hypothetical
        except Exception as exc:
            logger.debug(f"HyDE ошибка: {exc}")

        return None
