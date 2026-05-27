# -*- coding: utf-8 -*-
"""
Koib-V-4.6 — Модуль индексации
★ ИСПРАВЛЕНО: квантизация применяется к auto_model (стабильно)
★ SQLite DocStore вместо JSON (экономия RAM)
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


_GLOBAL_EMBEDDINGS = None


def get_global_embeddings():
    """
    ★ ИСПРАВЛЕНО: квантизация через auto_model (стабильно во всех версиях
    sentence-transformers). Применение ко всему SentenceTransformer
    могло приводить к падению encode().
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
        return _GLOBAL_EMBEDDINGS

    from langchain_huggingface import HuggingFaceEmbeddings
    import torch

    logger.info(f"Загрузка и CPU-квантизация {LOCAL_EMBEDDING_MODEL}...")
    hf_embeddings = HuggingFaceEmbeddings(
        model_name=LOCAL_EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
        model_kwargs={"device": device},
    )

    # ★ ИСПРАВЛЕНО: квантизуем только внутреннюю языковую модель
    try:
        auto_model = hf_embeddings.client[0].auto_model
        hf_embeddings.client[0].auto_model = torch.quantization.quantize_dynamic(
            auto_model,
            {torch.nn.Linear},
            dtype=torch.qint8,
        )
        logger.info(f"Модель {LOCAL_EMBEDDING_MODEL} квантизована (int8, auto_model)")
    except Exception as exc:
        logger.warning(f"Квантизация не удалась: {exc}. Используем FP32.")

    _GLOBAL_EMBEDDINGS = hf_embeddings
    return _GLOBAL_EMBEDDINGS


class SQLiteDocStore:
    """SQLite-хранилище полного контента чанков."""

    def __init__(self, path: Optional[Path] = None):
        self.path = path or DOCSTORE_DIR / "docstore.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._init_db()

    def _init_db(self) -> None:
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS docs (
                    chunk_id TEXT PRIMARY KEY,
                    full_content TEXT,
                    metadata TEXT,
                    source TEXT
                )
            ''')
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_source ON docs(source)')

    def add(self, chunk_id: str, full_content: str, metadata: Dict[str, Any]) -> None:
        with self.conn:
            self.conn.execute(
                'INSERT OR REPLACE INTO docs (chunk_id, full_content, metadata, source) '
                'VALUES (?, ?, ?, ?)',
                (chunk_id, full_content,
                 json.dumps(metadata, ensure_ascii=False),
                 metadata.get("source", "")),
            )

    def get_content(self, chunk_id: str) -> Optional[str]:
        cur = self.conn.cursor()
        cur.execute('SELECT full_content FROM docs WHERE chunk_id = ?', (chunk_id,))
        row = cur.fetchone()
        return row[0] if row else None

    def get_metadata(self, chunk_id: str) -> Optional[Dict[str, Any]]:
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
        cur = self.conn.cursor()
        cur.execute('SELECT COUNT(*) FROM docs')
        return cur.fetchone()[0]

    def remove_by_source(self, source: str) -> int:
        with self.conn:
            cur = self.conn.cursor()
            cur.execute('DELETE FROM docs WHERE source = ?', (source,))
            return cur.rowcount

    def close(self) -> None:
        if self.conn:
            self.conn.close()


class BM25Index:
    def __init__(self, path: Optional[Path] = None):
        self.path = path or INDEX_DIR / "bm25_index.pkl"
        self._texts: List[str] = []
        self._metadatas: List[Dict[str, Any]] = []
        self._bm25 = None

    def _tokenize(self, text: str, use_stopwords: bool = True) -> List[str]:
        import re
        tokens = re.findall(r'[а-яёa-z0-9]+(?:[-_][а-яёa-z0-9]+)*', text.lower())
        if BM25_USE_STOPWORDS and use_stopwords:
            tokens = [t for t in tokens if t not in RUSSIAN_STOPWORDS and len(t) > 1]
        return tokens

    def build(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> None:
        try:
            from rank_bm25 import BM25Okapi
            tokenized = [self._tokenize(t) for t in texts]
            tokenized = [t if t else ["_empty_"] for t in tokenized]
            self._texts = texts
            self._metadatas = metadatas
            self._bm25 = BM25Okapi(tokenized)
        except ImportError:
            logger.warning("rank_bm25 не установлен.")

    def search(self, query: str, k: int = 10) -> List[Tuple[Dict[str, Any], float]]:
        if self._bm25 is None:
            return []
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        scores = self._bm25.get_scores(query_tokens)
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        return [(self._metadatas[idx], float(score))
                for idx, score in indexed_scores[:k] if score > 0]

    def save(self) -> None:
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

    def load(self) -> bool:
        if not self.path.exists():
            return False
        try:
            with open(self.path, 'rb') as f:
                data = pickle.load(f)
            self.build(data["texts"], data["metadatas"])
            return True
        except Exception as exc:
            logger.warning(f"Не удалось загрузить BM25: {exc}")
            return False


class IndexBuilder:
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or INDEX_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.text_index_path = self.output_dir / "text_index"
        self.summary_index_path = self.output_dir / "summary_index"
        self.docstore = SQLiteDocStore()
        self.bm25 = BM25Index(self.output_dir / "bm25_index.pkl")
        self.text_vectorstore = None
        self.summary_vectorstore = None

    def _get_embeddings(self):
        return get_global_embeddings()

    def build(self, chunks: List[Chunk]) -> None:
        from langchain_community.vectorstores import FAISS
        ensure_dirs()
        embeddings = self._get_embeddings()

        text_docs, summary_docs = [], []
        bm25_texts, bm25_metadatas = [], []

        for chunk in chunks:
            bm25_text = chunk.content
            if chunk.full_content:
                bm25_text += "\n" + chunk.full_content
            bm25_texts.append(bm25_text)
            meta = chunk.metadata.copy()
            meta["chunk_id"] = chunk.chunk_id
            meta["chunk_type"] = chunk.chunk_type
            meta["content"] = chunk.content
            bm25_metadatas.append(meta)

            if chunk.full_content is not None:
                self.docstore.add(chunk.chunk_id, chunk.full_content, chunk.metadata)

            doc = chunk.to_langchain_doc()
            if EMBEDDING_PROVIDER == "local":
                doc.page_content = PASSAGE_PREFIX + doc.page_content
            if chunk.chunk_type == "text":
                text_docs.append(doc)
            else:
                summary_docs.append(doc)

        if text_docs:
            self.text_vectorstore = FAISS.from_documents(text_docs, embeddings)
            self.text_vectorstore.save_local(str(self.text_index_path))
        if summary_docs:
            self.summary_vectorstore = FAISS.from_documents(summary_docs, embeddings)
            self.summary_vectorstore.save_local(str(self.summary_index_path))
        if bm25_texts:
            self.bm25.build(bm25_texts, bm25_metadatas)
            self.bm25.save()

        logger.info(
            f"Индексы построены: text={len(text_docs)}, "
            f"summary={len(summary_docs)}, bm25={len(bm25_texts)}"
        )

    def add_chunks(self, chunks: List[Chunk]) -> None:
        from langchain_community.vectorstores import FAISS
        embeddings = self._get_embeddings()
        for chunk in chunks:
            if chunk.full_content is not None:
                self.docstore.add(chunk.chunk_id, chunk.full_content, chunk.metadata)
            doc = chunk.to_langchain_doc()
            if EMBEDDING_PROVIDER == "local":
                doc.page_content = PASSAGE_PREFIX + doc.page_content
            if chunk.chunk_type == "text":
                if self.text_vectorstore is not None:
                    self.text_vectorstore.add_documents([doc])
                else:
                    self.text_vectorstore = FAISS.from_documents([doc], embeddings)
                self.text_vectorstore.save_local(str(self.text_index_path))
            else:
                if self.summary_vectorstore is not None:
                    self.summary_vectorstore.add_documents([doc])
                else:
                    self.summary_vectorstore = FAISS.from_documents([doc], embeddings)
                self.summary_vectorstore.save_local(str(self.summary_index_path))
        # Перестраиваем BM25 (инкрементальное добавление не поддерживается)
        all_texts = self.bm25._texts + [
            c.content + ("\n" + c.full_content if c.full_content else "")
            for c in chunks
        ]
        all_metas = self.bm25._metadatas + [
            {**c.metadata, "chunk_id": c.chunk_id, "chunk_type": c.chunk_type,
             "content": c.content}
            for c in chunks
        ]
        self.bm25.build(all_texts, all_metas)
        self.bm25.save()

    def load(self) -> bool:
        from langchain_community.vectorstores import FAISS
        embeddings = self._get_embeddings()
        loaded = False
        if self.text_index_path.exists():
            try:
                self.text_vectorstore = FAISS.load_local(
                    str(self.text_index_path), embeddings,
                    allow_dangerous_deserialization=True,
                )
                loaded = True
            except Exception as exc:
                logger.warning(f"Ошибка загрузки text index: {exc}")
        if self.summary_index_path.exists():
            try:
                self.summary_vectorstore = FAISS.load_local(
                    str(self.summary_index_path), embeddings,
                    allow_dangerous_deserialization=True,
                )
                loaded = True
            except Exception as exc:
                logger.warning(f"Ошибка загрузки summary index: {exc}")
        self.bm25.load()
        return loaded