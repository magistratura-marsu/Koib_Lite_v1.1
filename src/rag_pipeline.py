# -*- coding: utf-8 -*-
import asyncio, logging, time
from typing import Dict, List, Optional, Any
from config import USE_USHAPED_CONTEXT, MAX_CONCURRENT_GENERATIONS, EMBEDDING_PROVIDER, QUERY_PREFIX
from .indexing import get_global_embeddings
from .retrieval import HybridRetriever, SemanticCache, reorder_u_shape
from .generation import LLMClient, build_prompt
from .validation import AnswerValidator, get_blocked_response
from .utils import ConversationMemory, rewrite_query

logger = logging.getLogger("koib.rag_pipeline")
CONTEXT_PRONOUNS = {"он", "она", "оно", "они", "его", "её", "их", "нему", "ней", "ними", "этом", "этот", "тот", "такой", "там", "это", "неё", "него", "у неё", "у него"}

class RAGPipeline:
    def __init__(self):
        self.retriever = HybridRetriever()
        self.llm = LLMClient()
        self.semantic_cache = SemanticCache()
        self.memory = ConversationMemory()
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_GENERATIONS)

    async def answer(self, query: str, user_id: str = "anonymous", k: int = 4, model_filter: str = "", use_memory: bool = True, validate: bool = True) -> Dict[str, Any]:
        t0 = time.time()
        async with self._semaphore:
            history = await self.memory.get_history(user_id) if use_memory and user_id != "anonymous" else []
            search_query = query
            if history and any(w.strip(",.!?").lower() in CONTEXT_PRONOUNS for w in query.split()):
                search_query = await rewrite_query(query, history, self.llm)

            query_embedding = None
            try:
                emb = get_global_embeddings()
                txt = (QUERY_PREFIX + search_query) if EMBEDDING_PROVIDER == "local" else search_query
                query_embedding = await asyncio.to_thread(emb.embed_query, txt)
            except Exception as e: logger.error(f"Emb error: {e}")

            if query_embedding:
                cached = self.semantic_cache.get(search_query, query_embedding)
                if cached:
                    if use_memory and user_id != "anonymous":
                        await self.memory.add_message(user_id, "user", query)
                        await self.memory.add_message(user_id, "assistant", cached["answer"][:500])
                    return {"answer": cached["answer"], "sources": cached.get("sources", []), "status": "approved", "latency": time.time() - t0}

            results = await asyncio.to_thread(self.retriever.search, search_query, k=k, model_filter=model_filter)
            if not results:
                return {"answer": "По вашему запросу не найдено релевантных фрагментов в официальной документации.", "sources": [], "status": "review", "latency": time.time() - t0}

            if USE_USHAPED_CONTEXT and len(results) > 2: results = reorder_u_shape(results)

            prompt = build_prompt(search_query, results)
            answer = await self.llm.generate_async(prompt)

            status = "approved"
            if validate:
                try:
                    vr = AnswerValidator().validate(answer, results, query)
                    if vr.status == "rejected": status = "rejected"; answer = get_blocked_response()
                    elif vr.status == "review": status = "review"
                except Exception as e: logger.warning(f"Validation error: {e}")

            if use_memory and user_id != "anonymous":
                await self.memory.add_message(user_id, "user", query)
                await self.memory.add_message(user_id, "assistant", answer[:500])

            if status == "approved" and query_embedding:
                self.semantic_cache.set(search_query, query_embedding, answer, [{"document": r.source, "page": r.page, "heading": r.heading} for r in results])

            return {"answer": answer, "sources": [{"document": r.source, "page": r.page, "heading": r.heading, "chunk_type": r.chunk_type, "score": r.score} for r in results], "status": status, "latency": time.time() - t0}
