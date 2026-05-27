# -*- coding: utf-8 -*-
"""
Koib-V-4.6 — VK Callback API (полностью асинхронный)
★ ИСПРАВЛЕНО: async ответ без блокировки event loop
★ ДОБАВЛЕНО: семантическое кэширование для повторяющихся вопросов
"""
import asyncio
import logging
from typing import Dict, Any
from fastapi import APIRouter, Request
import aiohttp
from src.generation import AnswerGenerator
from src.safety import check_query_safety, check_answer_safety, sanitize_answer
from src.validation import get_blocked_response
from config import VK_CONFIRM_CODE, VK_ACCESS_TOKEN, VK_GROUP_ID

logger = logging.getLogger("koib.api.vk")
router = APIRouter()

_generator: AnswerGenerator = None


def get_generator() -> AnswerGenerator:
    global _generator
    if _generator is None:
        _generator = AnswerGenerator()
    return _generator


async def send_vk_message(user_id: int, text: str) -> bool:
    if not VK_ACCESS_TOKEN:
        logger.warning("VK_ACCESS_TOKEN не задан")
        return False
    url = "https://api.vk.com/method/messages.send"
    params = {
        "user_id": user_id,
        "message": text,
        "access_token": VK_ACCESS_TOKEN,
        "v": "5.131",
        "random_id": abs(hash(f"{user_id}:{text}")) % (2**31),
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=params) as resp:
                data = await resp.json()
                if "error" in data:
                    logger.error(f"VK API ошибка: {data['error']}")
                    return False
                return True
    except Exception as exc:
        logger.error(f"Ошибка отправки VK: {exc}")
        return False


@router.post("/vk_callback")
async def vk_webhook(request: Request) -> Dict[str, Any]:
    data = await request.json()

    if data.get("type") == "confirmation":
        return VK_CONFIRM_CODE

    if data.get("type") == "message_new":
        try:
            msg = data["object"]["message"]
            user_id = msg["from_id"]
            text = msg["text"].strip()
            if not text:
                return "ok"

            logger.info(f"Запрос от {user_id}: {text[:100]}")

            # Проверка безопасности запроса
            is_safe, reason = check_query_safety(text)
            if not is_safe:
                logger.warning(f"Небезопасный запрос от {user_id}: {reason}")
                await send_vk_message(
                    user_id,
                    "Этот вопрос требует обращения в службу поддержки. "
                    "Пожалуйста, свяжитесь с нами по телефону горячей линии."
                )
                return "ok"

            # ★ НОВОЕ: полностью асинхронная генерация
            generator = get_generator()
            result = await generator.answer_async(text)
            answer = result.get("answer", "Не удалось сгенерировать ответ.")

            validation = result.get("validation")
            if validation and validation.get("status") == "rejected":
                answer = get_blocked_response()

            is_answer_safe, _ = check_answer_safety(answer)
            if not is_answer_safe:
                answer = sanitize_answer(answer)

            if len(answer) > 4096:
                answer = answer[:4090] + "..."

            await send_vk_message(user_id, answer)

        except Exception as exc:
            logger.error(f"Ошибка обработки VK: {exc}")

    return "ok"