# -*- coding: utf-8 -*-
"""
Koib-V-4.5 — FastAPI приложение
==================================
Асинхронный HTTP-сервер для приёма запросов от VK Callback API
и других клиентов. Готов к деплою на VPS с 1 vCPU / 2 ГБ ОЗУ.

Запуск:
  python main.py --serve
  или
  uvicorn api.app:app --host 0.0.0.0 --port 8000
"""
import logging

from fastapi import FastAPI

from api.middleware.logging import LoggingMiddleware
from api.routes.health import router as health_router
from api.routes.vk_callback import router as vk_router
from config import ensure_dirs

logger = logging.getLogger("koib.api")

# Создание приложения FastAPI
app = FastAPI(
    title="KOIB RAG API",
    description="Оптимизированная RAG-система для технической документации",
    version="4.5",
)

# Подключение промежуточного ПО
app.add_middleware(LoggingMiddleware)

# Подключение маршрутов
app.include_router(health_router, tags=["health"])
app.include_router(vk_router, tags=["vk"])


@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске сервера."""
    ensure_dirs()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.info("KOIB RAG API v4.5 запущен")


@app.on_event("shutdown")
async def shutdown_event():
    """Очистка при остановке сервера."""
    logger.info("KOIB RAG API v4.5 остановлен")
