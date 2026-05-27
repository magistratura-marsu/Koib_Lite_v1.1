# -*- coding: utf-8 -*-
"""
Koib-V-4.7 — Модуль индексации
★ ОБНОВЛЕНО: pymorphy2-лемматизация для FTS5 (fix для Recall)
"""
import json
import re
import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from config import (
    INDEX_DIR, DOCSTORE_DIR, METADATA_DIR,
    EMBEDDING_PROVIDER, LOCAL_EMBEDDING_MODEL, OPENAI_EMBEDDING_MODEL,
    OPENAI_API_KEY, BM25_USE_STOPWORDS, PASSAGE_PREFIX,
    BM25_USE_LEMMATIZATION,
)

logger = logging.getLogger("koib.indexing")

# ═══════════════════════════════════════════════════════════════
# Синглтон эмбеддингов
# ═══════════════════════════════════════════════════════════════
_GLOBAL_EMBEDDINGS = None


def get_global_embeddings():
    global _GLOBAL_EMBEDDINGS
    if _GLOBAL_EMBEDDINGS is not None:
        return _GLOBAL_EMBEDDINGS
    if EMBEDDING_PROVIDER == "local":
        from langchain_huggingface import HuggingFaceEmbeddings
        _GLOBAL_EMBEDDINGS = HuggingFaceEmbeddings(
            model_name=LOCAL_EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    elif EMBEDDING_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings
        _GLOBAL_EMBEDDINGS = OpenAIEmbeddings(
            model=OPENAI_EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY,
        )
    else:
        raise ValueError(f"Unknown EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER}")
    logger.info(f"Эмбеддинги загружены: {EMBEDDING_PROVIDER}")
    return _GLOBAL_EMBEDDINGS


# ═══════════════════════════════════════════════════════════════
# Русская токенизация + лемматизация для FTS5
# ═══════════════════════════════════════════════════════════════
RU_STOPWORDS = {
    "и", "в", "на", "с", "по", "для", "из", "к", "от", "о", "об", "а", "но",
    "да", "не", "что", "как", "это", "то", "же", "бы", "вы", "мы", "он", "она",
    "они", "оно", "я", "ты", "его", "её", "их", "мой", "твой", "наш", "ваш",
    "свой", "этот", "тот", "такой", "который", "весь", "все", "вся", "всё",
    "быть", "был", "была", "было", "были", "будет", "есть", "нет", "ещё", "уже",
    "только", "если", "или", "при", "про", "за", "до", "после", "между",
    "через", "над", "под", "перед", "так", "тоже", "лишь", "ведь", "вот",
    "даже", "ну", "ли", "ни", "тебя", "мне", "мной", "ним", "ней", "нами",
    "вам", "вас", "нас", "них", "чего", "чему", "чем", "кем", "ком", "где",
    "когда", "зачем", "почему", "куда", "откуда", "какой", "какая", "какие",
}

_TOKEN_RE = re.compile(r'[а-яёa-z0-9]+', re.IGNORECASE)

# Ленивая инициализация морфологического анализатора
_MORPH_ANALYZER = None


def _get_morph():
    global _MORPH_ANALYZER
    if _MORPH_ANALYZER is None and BM25_USE_LEMMATIZATION:
        try:
            import pymorphy2
            _MORPH_ANALYZER = pymorphy2.MorphAnalyzer()
            logger.info("pymorphy2 инициализирован для FTS5")
        except Exception as exc:
            logger.warning(f"pymorphy2 недоступен, fallback на raw tokens: {exc}")
    return _MORPH_ANALYZER


def _lemmatize_token(token: str) -> str:
    """Лемматизировать одно слово через pymorphy2."""
    morph = _get_morph()
    if morph is None:
        return token
    try:
        return morph.parse(token)[0].normal_form
    except Exception:
        return token


def tokenize_ru(text: str) -> str:
    """
    Токенизация + лемматизация русского текста для FTS5.
    ★ КРИТИЧНО: «бюллетень», «бюллетеня», «бюллетеней» → «бюллетень»
    """
    if not text:
        return ""
    raw_tokens = [t.lower() for t in _TOKEN_RE.findall(text) if len(t) > 1]
    if BM25_USE_STOPWORDS:
        raw_tokens = [t for t in raw_tokens if t not in RU_STOPWORDS]
    if BM25_USE_LEMMATIZATION:
        tokens = [_lemmatize_token(t) for t in raw_tokens]
    else:
        tokens = raw_tokens
    return " ".join(tokens)


def prepare_fts_query(query: str) -> str:
    """
    Подготовка запроса для FTS5 MATCH.
    ★ ВАЖНО: лемматизируем запрос так же, как и корпус, иначе не будет матчинга.
    """
    raw_tokens = [t.lower() for t in _TOKEN_RE.findall(query) if len(t) > 1]
    if BM25_USE_STOPWORDS:
        raw_tokens = [t for t in raw_tokens if t not in RU_STOPWORDS]
    if BM25_USE_LEMMATIZATION:
        tokens = [_lemmatize_token(t) for t in raw_tokens]
    else:
        tokens = raw_tokens
    if not tokens:
        return ""
    # Убираем дубликаты (после лемматизации могут появиться)
    seen = set()
    unique = []
    for t in tokens[:20]:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return " OR ".join(f'"{t}"' for t in unique)


# ═══════════════════════════════════════════════════════════════
# DocStore: SQLite хранилище full_content
# ═══════════════════════════════════════════════════════════════
class DocStore:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or (DOCSTORE_DIR / "docstore.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._init_db()

    def _init_db(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS docstore (
                    chunk_id TEXT PRIMARY KEY,
                    content TEXT,
                    chunk_type TEXT,
                    metadata TEXT
                )
            """)

    def add(self, chunk) -> None:
        if not chunk.full_content:
            return
        try:
            with self.conn:
                self.conn.execute(
                    "INSERT OR REPLACE INTO docstore "
                    "(chunk_id, content, chunk_type, metadata) VALUES (?, ?, ?, ?)",
                    (chunk.chunk_id, chunk.full_content, chunk.chunk_type,
                     json.dumps(chunk.metadata, ensure_ascii=False)),
                )
        except Exception as exc:
            logger.debug(f"DocStore add error: {exc}")

    def add_many(self, chunks) -> None:
        rows = [
            (c.chunk_id, c.full_content, c.chunk_type,
             json.dumps(c.metadata, ensure_ascii=False))
            for c in chunks if c.full_content
        ]
        if not rows:
            return
        try:
            with self.conn:
                self.conn.executemany(
                    "INSERT OR REPLACE INTO docstore "
                    "(chunk_id, content, chunk_type, metadata) VALUES (?, ?, ?, ?)",
                    rows,
                )
        except Exception as exc:
            logger.warning(f"DocStore add_many error: {exc}")

    def get_content(self, chunk_id: str) -> Optional[str]:
        cur = self.conn.cursor()
        cur.execute("SELECT content FROM docstore WHERE chunk_id = ?", (chunk_id,))
        row = cur.fetchone()
        return row[0] if row else None


# ═══════════════════════════════════════════════════════════════
# BM25 через SQLite FTS5
# ═══════════════════════════════════════════════════════════════
class BM25FTSIndex:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or (INDEX_DIR / "bm25_fts.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._init_db()

    def _init_db(self):
        with self.conn:
            self.conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    content,
                    chunk_type UNINDEXED,
                    source UNINDEXED,
                    page UNINDEXED,
                    heading UNINDEXED,
                    model UNINDEXED,
                    metadata UNINDEXED,
                    tokenize='unicode61 remove_diacritics 1'
                )
            """)

    def clear(self) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM chunks_fts")

    def add_chunks(self, chunks) -> None:
        rows = []
        for c in chunks:
            text_for_index = c.full_content if c.full_content else c.content
            tokenized = tokenize_ru(text_for_index)
            if not tokenized:
                continue
            rows.append((
                c.chunk_id, tokenized, c.chunk_type,
                c.metadata.get("source", ""),
                str(c.metadata.get("page", 0)),
                c.metadata.get("heading", ""),
                c.metadata.get("model", "unknown"),
                json.dumps(c.metadata, ensure_ascii=False),
            ))
        if not rows:
            return
        try:
            with self.conn:
                self.conn.executemany(
                    "INSERT INTO chunks_fts "
                    "(chunk_id, content, chunk_type, source, page, heading, model, metadata) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )
            logger.info(f"FTS5: добавлено {len(rows)} чанков (лемматизация={'ON' if BM25_USE_LEMMATIZATION else 'OFF'})")
        except Exception as exc:
            logger.warning(f"FTS5 add_chunks error: {exc}")

    def search(self, query: str, k: int = 10) -> List[Tuple[Dict[str, Any], float]]:
        fts_query = prepare_fts_query(query)
        if not fts_query:
            return []
        try:
            cur = self.conn.cursor()
            cur.execute(
                """
                SELECT chunk_id, content, chunk_type, source, page,
                       heading, model, metadata, bm25(chunks_fts) AS rank
                FROM chunks_fts
                WHERE chunks_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, k),
            )
            results = []
            for row in cur.fetchall():
                try:
                    metadata = json.loads(row[7]) if row[7] else {}
                except Exception:
                    metadata = {}
                metadata.setdefault("chunk_id", row[0])
                metadata.setdefault("chunk_type", row[2])
                metadata.setdefault("source", row[3])
                metadata.setdefault("page", int(row[4]) if row[4] else 0)
                metadata.setdefault("heading", row[5])
                metadata.setdefault("model", row[6])
                metadata.setdefault("content", row[1])
                score = -float(row[8]) if row[8] is not None else 0.0
                results.append((metadata, score))
            return results
        except Exception as exc:
            logger.warning(f"FTS5 search error: {exc}")
            return []

    def count(self) -> int:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM chunks_fts")
        row = cur.fetchone()
        return row[0] if row else 0


# ═══════════════════════════════════════════════════════════════
# IndexBuilder
# ═══════════════════════════════════════════════════════════════
class IndexBuilder:
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir) if output_dir else INDEX_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.text_vectorstore = None
        self.summary_vectorstore = None
        self.bm25 = BM25FTSIndex(self.output_dir / "bm25_fts.db")
        self.docstore = DocStore(DOCSTORE_DIR / "docstore.db")
        self._text_docs: List = []
        self._summary_docs: List = []

    def add_chunks(self, chunks) -> None:
        from langchain_core.documents import Document
        self.docstore.add_many(chunks)
        self.bm25.add_chunks(chunks)
        for c in chunks:
            lc_doc = c.to_langchain_doc()
            if c.chunk_type == "text":
                self._text_docs.append(lc_doc)
            else:
                self._summary_docs.append(lc_doc)
        if len(self._text_docs) + len(self._summary_docs) > 2000:
            self._flush_vectorstores()

    def _flush_vectorstores(self) -> None:
        if not self._text_docs and not self._summary_docs:
            return
        embeddings = get_global_embeddings()
        try:
            from langchain_community.vectorstores import FAISS
            if self._text_docs:
                if self.text_vectorstore is None:
                    self.text_vectorstore = FAISS.from_documents(
                        self._text_docs, embeddings
                    )
                else:
                    self.text_vectorstore.add_documents(self._text_docs)
                self.text_vectorstore.save_local(
                    str(self.output_dir), index_name="text_index"
                )
                logger.info(f"FAISS text: {len(self._text_docs)} docs added")
                self._text_docs = []
            if self._summary_docs:
                if self.summary_vectorstore is None:
                    self.summary_vectorstore = FAISS.from_documents(
                        self._summary_docs, embeddings
                    )
                else:
                    self.summary_vectorstore.add_documents(self._summary_docs)
                self.summary_vectorstore.save_local(
                    str(self.output_dir), index_name="summary_index"
                )
                logger.info(f"FAISS summary: {len(self._summary_docs)} docs added")
                self._summary_docs = []
        except Exception as exc:
            logger.error(f"Ошибка сборки FAISS: {exc}")

    def save(self) -> None:
        self._flush_vectorstores()
        logger.info(f"Индексы сохранены. FTS5 чанков: {self.bm25.count()}")

    def load(self) -> None:
        embeddings = get_global_embeddings()
        try:
            from langchain_community.vectorstores import FAISS
            text_path = self.output_dir / "text_index.faiss"
            if text_path.exists():
                self.text_vectorstore = FAISS.load_local(
                    str(self.output_dir), embeddings,
                    index_name="text_index", allow_dangerous_deserialization=True,
                )
                logger.info("FAISS text_index загружен")
            summary_path = self.output_dir / "summary_index.faiss"
            if summary_path.exists():
                self.summary_vectorstore = FAISS.load_local(
                    str(self.output_dir), embeddings,
                    index_name="summary_index", allow_dangerous_deserialization=True,
                )
                logger.info("FAISS summary_index загружен")
        except Exception as exc:
            logger.warning(f"Ошибка загрузки FAISS: {exc}")
        logger.info(f"FTS5 чанков в индексе: {self.bm25.count()}")
