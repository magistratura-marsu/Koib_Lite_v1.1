# -*- coding: utf-8 -*-
"""
Koib-V-4.5 — Модуль гибридного поиска
★ ДОБАВЛЕНО: Query Expansion для таблиц (вместо HyDE)
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


class ResponseCache:
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
        return hashlib.md5(text.lower().strip().encode()).hexdigest()

    def get(self, query: str) -> Optional[str]:
        cur = self.conn.cursor()
        cur.execute('SELECT hypothetical FROM cache WHERE query_hash = ?',
                    (self._hash(query),))
        row = cur.fetchone()
        return row[0] if row else None

    def set(self, query: str, hypothetical: str) -> None:
        with self.conn:
            self.conn.execute(
                'INSERT OR REPLACE INTO cache (query_hash, hypothetical) VALUES (?, ?)',
                (self._hash(query), hypothetical),
            )

    def clear(self) -> None:
        with self.conn:
            self.conn.execute('DELETE FROM cache')


@dataclass
class RetrievalResult:
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


TABLE_KEYWORDS = {"таблиц", "значени", "параметр", "сводк", "данные", "показател"}
FORMULA_KEYWORDS = {"формул", "вычислен", "расчёт", "уравнен", "коэффициент"}
FIGURE_KEYWORDS = {"схем", "рисунок", "диаграмм", "чертёж", "график"}

# ★ НОВОЕ: расширения запросов для таблиц (без LLM, чистая эвристика)
TABLE_EXPANSION_SUFFIXES = [
    " таблица параметры значения",
    " характеристики спецификация",
]


def _detect_query_intent(query: str) -> Dict[str, float]:
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


def _expand_query_for_tables(query: str, intent: Dict[str, float]) -> List[str]:
    """
    ★ НОВОЕ: Query Expansion для табличных запросов.
    Вместо HyDE (который шумит на слабой LLM) генерируем несколько
    эвристических вариантов запроса с табличными ключевыми словами.
    Это повышает recall без затрат CPU/RAM.
    """
    queries = [query]
    if intent.get("table", 0) >= 0.5:
        for suffix in TABLE_EXPANSION_SUFFIXES:
            expanded = query + suffix
            queries.append(expanded)
    return queries


class HybridRetriever:
    def __init__(self, index_builder: Optional[IndexBuilder] = None):
        self.index_builder = index_builder or IndexBuilder()
        self.index_builder.load()
        self._reranker = None
        self._cache = ResponseCache()

    def _get_reranker(self):
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
        intent = _detect_query_intent(query)

        search_query = query
        use_hyde_flag = use_hyde if use_hyde is not None else USE_HYDE
        if use_hyde_flag:
            hyde_result = self._apply_hyde(query)
            if hyde_result:
                search_query = hyde_result

        # ★ ИЗМЕНЕНО: векторный поиск с query expansion для таблиц
        vector_results = self._vector_search(search_query, intent, model_filter)
        bm25_results = self._bm25_search(query, model_filter)
        fused = self._reciprocal_rank_fusion(vector_results, bm25_results)

        try:
            from .quarantine import filter_quarantined_chunks
            fused = filter_quarantined_chunks(fused)
        except Exception:
            pass

        if USE_RERANKER and len(fused) > k:
            reranker = self._get_reranker()
            if reranker:
                fused = self._rerank(query, fused, reranker)

        results = fused[:k]
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
        results: List[RetrievalResult] = []
        seen_ids: set = set()

        # ★ Query Expansion: несколько вариантов запроса для таблиц
        queries = _expand_query_for_tables(query, intent)

        for q_idx, search_query in enumerate(queries):
            search_text = f"{QUERY_PREFIX}{search_query}" if QUERY_PREFIX else search_query
            # Вес снижается для расширенных вариантов
            expansion_weight = 1.0 if q_idx == 0 else 0.85

            if self.index_builder.text_vectorstore is not None:
                try:
                    docs = self.index_builder.text_vectorstore.similarity_search_with_score(
                        search_text, k=VECTOR_SEARCH_K,
                    )
                    for doc, score in docs:
                        chunk_id = doc.metadata.get("chunk_id", "")
                        if chunk_id in seen_ids:
                            continue
                        seen_ids.add(chunk_id)
                        chunk_type = doc.metadata.get("chunk_type", "text")
                        r = RetrievalResult(
                            chunk_id=chunk_id,
                            content=doc.page_content,
                            score=float(score) * intent.get("text", 1.0) * expansion_weight,
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

            if self.index_builder.summary_vectorstore is not None:
                try:
                    docs = self.index_builder.summary_vectorstore.similarity_search_with_score(
                        search_text, k=VECTOR_SEARCH_K,
                    )
                    for doc, score in docs:
                        chunk_id = doc.metadata.get("chunk_id", "")
                        if chunk_id in seen_ids:
                            continue
                        seen_ids.add(chunk_id)
                        chunk_type = doc.metadata.get("chunk_type", "text")
                        type_weight = intent.get(chunk_type, 1.0)
                        r = RetrievalResult(
                            chunk_id=chunk_id,
                            content=doc.page_content,
                            score=float(score) * type_weight * expansion_weight,
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
        chunk_scores: Dict[str, float] = {}
        chunk_map: Dict[str, RetrievalResult] = {}
        for rank, r in enumerate(vector_results, 1):
            if r.chunk_id not in chunk_scores:
                chunk_scores[r.chunk_id] = 0.0
                chunk_map[r.chunk_id] = r
            chunk_scores[r.chunk_id] += HYBRID_ALPHA / (k_rrf + rank)
        for rank, r in enumerate(bm25_results, 1):
            if r.chunk_id not in chunk_scores:
                chunk_scores[r.chunk_id] = 0.0
                chunk_map[r.chunk_id] = r
            chunk_scores[r.chunk_id] += (1 - HYBRID_ALPHA) / (k_rrf + rank)
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