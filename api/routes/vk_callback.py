# -*- coding: utf-8 -*-
"""
Koib-V-4.8 — VK Callback API
★ ИСПРАВЛЕНО: BackgroundTasks для мгновенного ответа VK (защита от ретраев)
★ ИСПРАВЛЕНО: детерминированный random_id через hashlib
★ ИСПРАВЛЕНО: Pydantic-валидация payload
★ ИНТЕГРАЦИЯ: RAGPipeline с памятью диалога
"""
import asyncio
import hashlib
import logging
import time
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, Request, BackgroundTasks, HTTPException
import aiohttp

from src.rag_pipeline import RAGPipeline
from src.safety import check_query_safety, check_answer_safety, sanitize_answer
from src.validation import get_blocked_response
from config import VK_CONFIRM_CODE, VK_ACCESS_TOKEN

logger = logging.getLogger("koib.api.vk")
router = APIRouter()

_pipeline: Optional[RAGPipeline] = None


# ═══════════════════════════════════════════════════════════════
# Pydantic-модели для валидации VK payload
# ═══════════════════════════════════════════════════════════════
class VKMessage(BaseModel):
    from_id: int
    text: str = ""


class VKObject(BaseModel):
    message: VKMessage


class VKCallbackPayload(BaseModel):
    type: str
    object: Optional[VKObject] = None
    group_id: Optional[int] = None


# ═══════════════════════════════════════════════════════════════
# Rate Limiter на TTLCache (защита от утечки RAM)
# ═══════════════════════════════════════════════════════════════
try:
    from cachetools import TTLCache
    _user_requests = TTLCache(maxsize=10000, ttl=60)
except ImportError:
    # Fallback если cachetools не установлен
    from collections import defaultdict
    _user_requests = defaultdict(list)


def _check_rate_limit(user_id: int, limit: int = 5) -> bool:
    if hasattr(_user_requests, 'ttl'):  # TTLCache
        current = _user_requests.get(user_id, 0)
        if current >= limit:
            return False
        _user_requests[user_id] = current + 1
        return True
    else:  # defaultdict fallback
        now = time.time()
        _user_requests[user_id] = [t for t in _user_requests[user_id] if now - t < 60]
        if len(_user_requests[user_id]) >= limit:
            return False
        _user_requests[user_id].append(now)
        return True


def _get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


async def send_vk_message(user_id: int, text: str) -> bool:
    if not VK_ACCESS_TOKEN:
        logger.warning("VK_ACCESS_TOKEN не задан")
        return False
    url = "https://api.vk.com/method/messages.send"
    # ★ Детерминированный random_id (не ломается при рестарте)
    random_id = int(hashlib.md5(
        f"{user_id}:{text}:{int(time.time() // 60)}".encode()
    ).hexdigest()[:8], 16)
    params = {
        "user_id": user_id, "message": text[:4090],
        "access_token": VK_ACCESS_TOKEN,
        "v": "5.131", "random_id": random_id,
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                data = await resp.json()
                if "error" in data:
                    logger.error(f"VK API ошибка: {data['error']}")
                    return False
                return True
    except Exception as exc:
        logger.error(f"Ошибка отправки VK: {exc}")
        return False


async def _process_vk_request(user_id: int, text: str):
    """Фоновая обработка запроса (вне webhook-таймаута VK)."""
    try:
        if not _check_rate_limit(user_id):
            await send_vk_message(user_id, "Слишком много запросов. Подождите минуту.")
            return

        is_safe, reason = check_query_safety(text)
        if not is_safe:
            logger.warning(f"Небезопасный запрос от {user_id}: {reason}")
            await send_vk_message(
                user_id,
                "Этот вопрос требует обращения в службу поддержки. "
                "Пожалуйста, свяжитесь с нами по телефону горячей линии."
            )
            return

        pipeline = _get_pipeline()
        # ★ user_id передаётся для памяти диалога (Query Rewriting)
        result = await pipeline.answer(
            query=text,
            user_id=str(user_id),
            k=4,
            use_memory=True,
            validate=False,  # Валидация выключена для скорости VK-бота
        )

        answer = result.get("answer", "Не удалось сгенерировать ответ.")
        if result.get("status") == "rejected":
            answer = get_blocked_response()

        is_safe, _ = check_answer_safety(answer)
        if not is_safe:
            answer = sanitize_answer(answer)

        await send_vk_message(user_id, answer)
    except Exception as exc:
        logger.error(f"Ошибка фоновой обработки VK: {exc}")


@router.post("/vk_callback")
async def vk_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
) -> str:
    """
    ★ КРИТИЧНО: мгновенно возвращаем "ok" VK, обработка идёт в фоне.
    Это предотвращает ретраи VK API (таймаут ~5 сек).
    """
    try:
        raw_data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # Confirmation handshake
    if raw_data.get("type") == "confirmation":
        return VK_CONFIRM_CODE

    # Валидация payload через Pydantic
    try:
        payload = VKCallbackPayload(**raw_data)
    except Exception:
        return "ok"  # Молча игнорируем мусорные запросы

    if payload.type == "message_new" and payload.object:
        user_id = payload.object.message.from_id
        text = payload.object.message.text.strip()
        if text:
            logger.info(f"Запрос от {user_id}: {text[:100]}")
            # ★ Обработка в фоне — webhook отвечает мгновенно
            background_tasks.add_task(_process_vk_request, user_id, text)

    return "ok"
