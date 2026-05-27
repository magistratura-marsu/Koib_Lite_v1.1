# -*- coding: utf-8 -*-
"""
Koib-V-4.5 — Модуль индексации
=================================
Векторная индексация (FAISS), BM25, SQLite DocStore.
Синглтон эмбеддингов с динамической квантизацией PyTorch (int8).

Ключевые отличия от v4.3:
  - SQLite вместо JSON/pickle для DocStore (меньше RAM)
  - Синглтон эмбеддингов с CPU-квантизацией
  - Принудительный CPU-режим (без CUDA)
  - e5-small вместо e5-large (33 МБ вместо 1.3 ГБ)
"""
import json
import pickle
import logging
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from .chunking import Chunk
from config import (
    INDEX_DIR, DOCSTORE_DIR, METADATA_DIR,
    LOCAL_EMBEDDING_MODEL, EMBEDDING_PROVIDER, OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL, PASSAGE_PREFIX,
    BM25_USE_STOPWORDS, get_device, ensure_dirs,
)

logger = logging.getLogger("koib.indexing")


# ═══════════════════════════════════════════════════════════════
# Стоп-слова для BM25
# ═══════════════════════════════════════════════════════════════
RUSSIAN_STOPWORDS = {
    "и", "в", "на", "с", "по", "для", "из", "к", "от", "о", "об",
    "а", "но", "что", "как", "это", "не", "да", "нет", "бы", "ли",
    "же", "при", "до", "за", "во", "со", "ко", "без", "над", "под",
    "через", "между", "около", "у", "про", "после", "перед", "вместо",
    "кроме", "среди", "вокруг", "вдоль", "поперёк",
    "он", "она", "оно", "они", "мы", "вы", "я", "ты",
    "его", "её", "их", "наш", "ваш", "свой",
    "этот", "тот", "такой", "который", "какой", "чей",
    "быть", "было", "будет", "есть", "были",
    "все", "всё", "вся", "весь", "каждый", "любой",
    "очень", "более", "менее", "также", "тоже",
}


# ═══════════════════════════════════════════════════════════════
# Синглтон эмбеддингов с квантизацией
# ═══════════════════════════════════════════════════════════════
_GLOBAL_EMBEDDINGS = None


def get_global_embeddings():
    """
    Получить глобальный экземпляр эмбеддинг-модели (синглтон).

    При первом вызове загружает модель и применяет динамическую
    квантизацию PyTorch (int8) для уменьшения потребления RAM.
    Последующие вызовы возвращают уже созданный экземпляр.

    Важно: bitsandbytes оптимизирована для CUDA, на CPU она либо
    не скомпилируется, либо замедлит инференс. Нативная
    квантизация PyTorch идеально ужимает модель без потери скорости.
    """
    global _GLOBAL_EMBEDDINGS
    if _GLOBAL_EMBEDDINGS is not None:
        return _GLOBAL_EMBEDDINGS

    device = get_device()

    if EMBEDDING_PROVIDER == "openai" and OPENAI_API_KEY:
        from langchain_openai import OpenAIEmbeddings
        _GLOBAL_EMBEDDINGS = OpenAIEmbeddings(
            model=OPENAI_EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY,
        )
        logger.info("Загружены OpenAI эмбеддинги")
    else:
        from langchain_huggingface import HuggingFaceEmbeddings
        import torch

        logger.info(f"Загрузка и CPU-квантизация {LOCAL_EMBEDDING_MODEL}...")
        hf_embeddings = HuggingFaceEmbeddings(
            model_name=LOCAL_EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
            model_kwargs={"device": device},
        )

        # Динамическая квантизация PyTorch (int8) для CPU
        # Уменьшает размер модели в ~2 раза и ускоряет инференс
        hf_embeddings.client = torch.quantization.quantize_dynamic(
            hf_embeddings.client,
            {torch.nn.Linear},
            dtype=torch.qint8,
        )
        _GLOBAL_EMBEDDINGS = hf_embeddings
        logger.info(f"Модель {LOCAL_EMBEDDING_MODEL} загружена с квантизацией int8")

    return _GLOBAL_EMBEDDINGS


# ═══════════════════════════════════════════════════════════════
# SQLite DocStore
# ═══════════════════════════════════════════════════════════════
class SQLiteDocStore:
    """
    SQLite-хранилище полного контента чанков.

    В отличие от JSON/pickle файлов, SQLite:
      - Не загружает все данные в RAM
      - Поддерживает эффективный поиск по ключу
      - Имеет индексы для быстрого доступа
      - Атомарно записывает данные
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = path or DOCSTORE_DIR / "docstore.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._init_db()

    def _init_db(self) -> None:
        """Создать таблицы и индексы при первом запуске."""
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS docs (
                    chunk_id TEXT PRIMARY KEY,
                    full_content TEXT,
                    metadata TEXT,
                    source TEXT
                )
            ''')
            self.conn.execute(
                'CREATE INDEX IF NOT EXISTS idx_source ON docs(source)'
            )

    def add(
        self,
        chunk_id: str,
        full_content: str,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Добавить или обновить запись в хранилище.

        Args:
            chunk_id:     Идентификатор чанка
            full_content: Полный текст контента
            metadata:     Метаданные чанка (сериализуются в JSON)
        """
        with self.conn:
            self.conn.execute(
                'INSERT OR REPLACE INTO docs (chunk_id, full_content, metadata, source) '
                'VALUES (?, ?, ?, ?)',
                (
                    chunk_id,
                    full_content,
                    json.dumps(metadata, ensure_ascii=False),
                    metadata.get("source", ""),
                ),
            )

    def get_content(self, chunk_id: str) -> Optional[str]:
        """
        Получить полный контент чанка по ID.

        Args:
            chunk_id: Идентификатор чанка

        Returns:
            Полный текст или None, если чанк не найден
        """
        cur = self.conn.cursor()
        cur.execute('SELECT full_content FROM docs WHERE chunk_id = ?', (chunk_id,))
        row = cur.fetchone()
        return row[0] if row else None

    def get_metadata(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Получить метаданные чанка по ID."""
        cur = self.conn.cursor()
        cur.execute('SELECT metadata FROM docs WHERE chunk_id = ?', (chunk_id,))
        row = cur.fetchone()
        if row:
            try:
                return json.loads(row[0])
            except json.JSONDecodeError:
                return None
        return None

    @property
    def size(self) -> int:
        """Количество записей в хранилище."""
        cur = self.conn.cursor()
        cur.execute('SELECT COUNT(*) FROM docs')
        return cur.fetchone()[0]

    def remove_by_source(self, source: str) -> int:
        """
        Удалить все записи для указанного источника.

        Используется при повторной индексации документа.

        Args:
            source: Имя файла-источника

        Returns:
            Количество удалённых записей
        """
        with self.conn:
            cur = self.conn.cursor()
            cur.execute('DELETE FROM docs WHERE source = ?', (source,))
            return cur.rowcount

    def close(self) -> None:
        """Закрыть соединение с базой данных."""
        if self.conn:
            self.conn.close()


# ═══════════════════════════════════════════════════════════════
# BM25-индекс
# ═══════════════════════════════════════════════════════════════
class BM25Index:
    """
    BM25-индекс для лексического поиска.

    Токенизация с фильтрацией русских стоп-слов.
    Сериализация через pickle.
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = path or INDEX_DIR / "bm25_index.pkl"
        self._texts: List[str] = []
        self._metadatas: List[Dict[str, Any]] = []
        self._bm25 = None

    def _tokenize(self, text: str, use_stopwords: bool = True) -> List[str]:
        """
        Токенизация текста для BM25.

        Извлекает слова из кириллицы и латиницы, фильтрует стоп-слова.
        """
        import re
        tokens = re.findall(
            r'[а-яёa-z0-9]+(?:[-_][а-яёa-z0-9]+)*',
            text.lower(),
        )
        if BM25_USE_STOPWORDS and use_stopwords:
            tokens = [t for t in tokens if t not in RUSSIAN_STOPWORDS and len(t) > 1]
        return tokens

    def build(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """
        Построить BM25-индекс по текстам.

        Args:
            texts:      Список текстов для индексации
            metadatas:  Метаданные для каждого текста
        """
        try:
            from rank_bm25 import BM25Okapi
            tokenized = [self._tokenize(t) for t in texts]
            # Заменяем пустые токены на плейсхолдер
            tokenized = [t if t else ["_empty_"] for t in tokenized]
            self._texts = texts
            self._metadatas = metadatas
            self._bm25 = BM25Okapi(tokenized)
        except ImportError:
            logger.warning("rank_bm25 не установлен. BM25-поиск недоступен.")

    def search(
        self,
        query: str,
        k: int = 8,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Поиск по BM25-индексу.

        Args:
            query: Поисковый запрос
            k:     Количество возвращаемых результатов

        Returns:
            Список кортежей (метаданные, оценка)
        """
        if self._bm25 is None:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        return [
            (self._metadatas[idx], float(score))
            for idx, score in indexed_scores[:k]
            if score > 0
        ]

    def save(self) -> None:
        """Сохранить BM25-индекс на диск."""
        if self._bm25 is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "texts": self._texts,
            "metadatas": self._metadatas,
            "bm25_corpus": getattr(self._bm25, 'corpus', None),
        }
        with open(self.path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"BM25-индекс сохранён: {self.path}")

    def load(self) -> bool:
        """
        Загрузить BM25-индекс с диска.

        Returns:
            True, если индекс успешно загружен
        """
        if not self.path.exists():
            return False
        try:
            with open(self.path, 'rb') as f:
                data = pickle.load(f)
            self.build(data["texts"], data["metadatas"])
            logger.info(f"BM25-индекс загружен: {self.path}")
            return True
        except Exception as exc:
            logger.warning(f"Не удалось загрузить BM25-индекс: {exc}")
            return False


# ═══════════════════════════════════════════════════════════════
# Построитель индекса
# ═══════════════════════════════════════════════════════════════
class IndexBuilder:
    """
    Построитель составного индекса: FAISS (векторный) + BM25 + DocStore.

    Создаёт два векторных индекса:
      1. text_index — для текстовых чанков
      2. summary_index — для сводок таблиц/формул

    SQLite DocStore хранит полный контент структурированных элементов.
    BM25 обеспечивает лексический поиск по текстам чанков.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        from langchain_community.vectorstores import FAISS

        self.output_dir = output_dir or INDEX_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.text_index_path = self.output_dir / "text_index"
        self.summary_index_path = self.output_dir / "summary_index"

        self.docstore = SQLiteDocStore()
        self.bm25 = BM25Index(self.output_dir / "bm25_index.pkl")

        self.text_vectorstore: Optional[FAISS] = None
        self.summary_vectorstore: Optional[FAISS] = None

    def _get_embeddings(self):
        """Получить экземпляр эмбеддингов через синглтон."""
        return get_global_embeddings()

    def build(self, chunks: List[Chunk]) -> None:
        """
        Построить все индексы по списку чанков.

        Разделяет чанки на текстовые и структурированные,
        строит отдельные векторные индексы для каждого типа,
        заполняет DocStore и BM25-индекс.

        Args:
            chunks: Список чанков из SmartChunker
        """
        from langchain_community.vectorstores import FAISS

        embeddings = self._get_embeddings()

        text_docs = []
        summary_docs = []
        bm25_texts = []
        bm25_metadatas = []

        for chunk in chunks:
            # Все чанки идут в BM25
            bm25_texts.append(chunk.content)
            bm25_metadatas.append(chunk.metadata.copy())

            # Структурированные чанки — в DocStore
            if chunk.full_content is not None:
                self.docstore.add(
                    chunk_id=chunk.chunk_id,
                    full_content=chunk.full_content,
                    metadata=chunk.metadata,
                )

            # Разделение на текстовые и структурированные
            if chunk.chunk_type == "text":
                text_docs.append(chunk.to_langchain_doc())
            else:
                # Сводка для векторного поиска
                summary_docs.append(chunk.to_langchain_doc())

        # Векторный индекс текстов
        if text_docs:
            logger.info(f"Построение текстового векторного индекса ({len(text_docs)} чанков)...")
            self.text_vectorstore = FAISS.from_documents(
                text_docs, embeddings,
            )
            self.text_vectorstore.save_local(str(self.text_index_path))
            logger.info(f"Текстовый индекс сохранён: {self.text_index_path}")

        # Векторный индекс сводок
        if summary_docs:
            logger.info(f"Построение индекса сводок ({len(summary_docs)} чанков)...")
            self.summary_vectorstore = FAISS.from_documents(
                summary_docs, embeddings,
            )
            self.summary_vectorstore.save_local(str(self.summary_index_path))
            logger.info(f"Индекс сводок сохранён: {self.summary_index_path}")

        # BM25-индекс
        if bm25_texts:
            self.bm25.build(bm25_texts, bm25_metadatas)
            self.bm25.save()

        logger.info(
            f"Индексы построены: текст={len(text_docs)}, "
            f"сводки={len(summary_docs)}, BM25={len(bm25_texts)}"
        )

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """
        Добавить чанки к существующим индексам (инкрементальное обновление).

        Args:
            chunks: Список новых чанков
        """
        from langchain_community.vectorstores import FAISS

        embeddings = self._get_embeddings()

        text_docs = []
        summary_docs = []
        bm25_texts = []
        bm25_metadatas = []

        for chunk in chunks:
            bm25_texts.append(chunk.content)
            bm25_metadatas.append(chunk.metadata.copy())

            if chunk.full_content is not None:
                self.docstore.add(
                    chunk_id=chunk.chunk_id,
                    full_content=chunk.full_content,
                    metadata=chunk.metadata,
                )

            if chunk.chunk_type == "text":
                text_docs.append(chunk.to_langchain_doc())
            else:
                summary_docs.append(chunk.to_langchain_doc())

        # Добавление в текстовый индекс
        if text_docs:
            if self.text_vectorstore is not None:
                self.text_vectorstore.add_documents(text_docs)
            else:
                self.text_vectorstore = FAISS.from_documents(text_docs, embeddings)
            self.text_vectorstore.save_local(str(self.text_index_path))

        # Добавление в индекс сводок
        if summary_docs:
            if self.summary_vectorstore is not None:
                self.summary_vectorstore.add_documents(summary_docs)
            else:
                self.summary_vectorstore = FAISS.from_documents(summary_docs, embeddings)
            self.summary_vectorstore.save_local(str(self.summary_index_path))

        # Перестройка BM25 (инкрементальное обновление не поддерживается)
        if bm25_texts:
            all_texts = self.bm25._texts + bm25_texts
            all_metadatas = self.bm25._metadatas + bm25_metadatas
            self.bm25.build(all_texts, all_metadatas)
            self.bm25.save()

        logger.info(f"Добавлено {len(chunks)} чанков к индексам")

    def load(self) -> bool:
        """
        Загрузить все индексы с диска.

        Returns:
            True, если хотя бы один индекс загружен успешно
        """
        from langchain_community.vectorstores import FAISS

        embeddings = self._get_embeddings()
        loaded = False

        # Текстовый векторный индекс
        if self.text_index_path.exists():
            try:
                self.text_vectorstore = FAISS.load_local(
                    str(self.text_index_path),
                    embeddings,
                    allow_dangerous_deserialization=True,
                )
                loaded = True
                logger.info("Текстовый векторный индекс загружен")
            except Exception as exc:
                logger.warning(f"Не удалось загрузить текстовый индекс: {exc}")

        # Индекс сводок
        if self.summary_index_path.exists():
            try:
                self.summary_vectorstore = FAISS.load_local(
                    str(self.summary_index_path),
                    embeddings,
                    allow_dangerous_deserialization=True,
                )
                loaded = True
                logger.info("Индекс сводок загружен")
            except Exception as exc:
                logger.warning(f"Не удалось загрузить индекс сводок: {exc}")

        # BM25-индекс
        if self.bm25.load():
            loaded = True

        return loaded
