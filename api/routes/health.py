# -*- coding: utf-8 -*-
"""
Koib-V-4.5 — Маршрут проверки здоровья
=========================================
Health check эндпоинт для мониторинга состояния сервера.
"""
import time
import logging
from typing import Dict, Any

from fastapi import APIRouter

logger = logging.getLogger("koib.api.health")

router = APIRouter()

# Время запуска сервера
_start_time = time.time()


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Проверка состояния сервера.

    Возвращает:
      - status: "ok" если сервер работает
      - uptime: Время работы в секундах
      - version: Версия системы
    """
    uptime = time.time() - _start_time
    return {
        "status": "ok",
        "uptime_seconds": round(uptime, 1),
        "version": "4.5",
    }


@router.get("/")
async def root() -> Dict[str, str]:
    """Корневой маршрут — описание API."""
    return {
        "name": "KOIB RAG API",
        "version": "4.5",
        "description": "Оптимизированная RAG-система для технической документации",
        "docs": "/docs",
    }
