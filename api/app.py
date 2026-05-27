# -*- coding: utf-8 -*-
import logging, aiohttp
from fastapi import FastAPI
from api.middleware.logging import LoggingMiddleware
from api.routes.health import router as health_router
from api.routes.vk_callback import router as vk_router
from config import ensure_dirs

logger = logging.getLogger("koib.api")
app = FastAPI(title="KOIB RAG API", description="Экстремально оптимизированная RAG-система для операторов КОИБ", version="4.8")
app.add_middleware(LoggingMiddleware)
app.include_router(health_router, tags=["health"])
app.include_router(vk_router, tags=["vk"])

@app.on_event("startup")
async def startup_event():
    ensure_dirs()
    # ★ ГЛОБАЛЬНЫЙ СИНГЛТОН СЕССИИ
    app.state.vk_session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=100, ttl_dns_cache=300))
    
    # Очистка устаревшего кэша
    try:
        from src.retrieval import SemanticCache
        SemanticCache().purge_stale(days=30)
    except Exception: pass
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
    logger.info("KOIB RAG API v4.8 запущен с пулом постоянных сетевых соединений")

@app.on_event("shutdown")
async def shutdown_event():
    if hasattr(app.state, "vk_session") and not app.state.vk_session.closed:
        await app.state.vk_session.close()
        logger.info("Глобальная HTTP-сессия aiohttp успешно закрыта")
