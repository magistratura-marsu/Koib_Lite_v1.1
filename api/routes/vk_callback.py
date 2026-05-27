# -*- coding: utf-8 -*-
"""
Koib-V-4.5 — Маршрут VK Callback API
========================================
Обработка входящих событий от VK Callback API:
  - confirmation — подтверждение сервера
  - message_new — новое сообщение от пользователя

VK отправляет POST-запросы с JSON-телом, содержащим тип события
и данные сообщения. Ответ отправляется через VK API (messages.send).
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

# Глобальный генератор ответов (инициализируется при запуске)
_generator: AnswerGenerator = None


def get_generator() -> AnswerGenerator:
    """Получить или создать экземпляр AnswerGenerator."""
    global _generator
    if _generator is None:
        _generator = AnswerGenerator()
    return _generator


async def send_vk_message(user_id: int, text: str) -> bool:
    """
    Отправить сообщение пользователю через VK API.

    Args:
        user_id: ID пользователя ВКонтакте
        text:    Текст сообщения

    Returns:
        True, если сообщение успешно отправлено
    """
    if not VK_ACCESS_TOKEN:
        logger.warning("VK_ACCESS_TOKEN не задан, отправка невозможна")
        return False

    url = "https://api.vk.com/method/messages.send"
    params = {
        "user_id": user_id,
        "message": text,
        "access_token": VK_ACCESS_TOKEN,
        "v": "5.131",
        "random_id": hash(f"{user_id}:{text}") % (2**31),
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=params) as resp:
                data = await resp.json()
                if "error" in data:
                    logger.error(f"VK API ошибка: {data['error']}")
                    return False
                logger.info(f"Сообщение отправлено пользователю {user_id}")
                return True
    except Exception as exc:
        logger.error(f"Ошибка отправки VK сообщения: {exc}")
        return False


@router.post("/vk_callback")
async def vk_webhook(request: Request) -> Dict[str, Any]:
    """
    Обработчик VK Callback API.

    Обрабатывает два типа событий:
      1. confirmation — подтверждение сервера (при настройке Callback)
      2. message_new — входящее сообщение от пользователя

    Для message_new:
      - Проверка безопасности запроса
      - Генерация ответа через RAG-пайплайн
      - Проверка безопасности ответа
      - Отправка ответа пользователю
    """
    data = await request.json()

    # Подтверждение сервера VK
    if data.get("type") == "confirmation":
        logger.info("VK confirmation запрос получен")
        return VK_CONFIRM_CODE

    # Обработка нового сообщения
    if data.get("type") == "message_new":
        try:
            msg = data["object"]["message"]
            user_id = msg["from_id"]
            text = msg["text"].strip()

            if not text:
                return "ok"

            logger.info(f"Запрос от пользователя {user_id}: {text[:100]}")

            # Проверка безопасности запроса
            is_safe, reason = check_query_safety(text)
            if not is_safe:
                logger.warning(f"Небезопасный запрос от {user_id}: {reason}")
                await send_vk_message(user_id, "Запрос отклонён по соображениям безопасности.")
                return "ok"

            # Генерация ответа (в пуле потоков, чтобы не блокировать event loop)
            generator = get_generator()
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                generator.answer,
                text,
            )

            answer = result.get("answer", "Не удалось сгенерировать ответ.")

            # Проверка валидации
            validation = result.get("validation")
            if validation and validation.get("status") == "rejected":
                answer = get_blocked_response()

            # Проверка безопасности ответа
            is_answer_safe, _ = check_answer_safety(answer)
            if not is_answer_safe:
                answer = sanitize_answer(answer)

            # Ограничение длины ответа для VK (максимум 4096 символов)
            if len(answer) > 4096:
                answer = answer[:4090] + "..."

            # Отправка ответа
            await send_vk_message(user_id, answer)

        except Exception as exc:
            logger.error(f"Ошибка обработки VK сообщения: {exc}")

    return "ok"
