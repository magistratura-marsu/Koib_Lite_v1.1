# -*- coding: utf-8 -*-
"""
Koib-V-4.6 — Модуль безопасности
★ ИСПРАВЛЕНО: интерфейс check_query_safety / check_answer_safety / sanitize_answer
для использования в VK-боте
"""
import re
import logging
from typing import Tuple, List
from dataclasses import dataclass

logger = logging.getLogger("koib.safety")


@dataclass
class SensitiveTopic:
    category: str
    patterns: List[str]
    description: str


SENSITIVE_TOPICS = [
    SensitiveTopic("invalid_ballot", [
        r"\bнедействительн(?:ый|ые|ых)\s+бюллетен",
        r"\bаннулировани(?:е|я)\s+бюллетен",
    ], "Вопросы о недействительных бюллетенях"),
    SensitiveTopic("complaint", [
        r"\bжалоб(?:а|ы|у)\b", r"\bпожаловаться\b", r"\bпротест\b",
        r"\bнарушени(?:е|я|й)\b", r"\bапелляци(?:я|и|ю)\b",
    ], "Жалобы и нарушения"),
    SensitiveTopic("technical_failure", [
        r"\bтехнический\s+сбой\b", r"\bнеисправност",
        r"\bотказ\s+(?:оборудования|терминала|КОИБ)\b",
        r"\bзавис(?:ание|ания|ает)\b", r"\bперезагрузк",
        r"\bне\s+работает\b", r"\bне\s+включается\b",
    ], "Технические сбои"),
    SensitiveTopic("security", [
        r"\bвзлом\b", r"\bнесанкционированный\s+доступ\b",
        r"\bутечка\s+данных\b", r"\bфальсификаци",
    ], "Вопросы безопасности"),
]

FORBIDDEN_ANSWER_PATTERNS = [
    r"\bя\s+не\s+могу\s+помочь\b",
    r"\bотказываюсь\s+отвечать\b",
    r"<\s*script",
    r"javascript:",
]


def check_query_safety(query: str) -> Tuple[bool, str]:
    """Проверить запрос на чувствительные темы. Возвращает (is_safe, reason)."""
    query_lower = query.lower()
    for topic in SENSITIVE_TOPICS:
        for pattern in topic.patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                logger.info(f"Обнаружена чувствительная тема: {topic.category}")
                return False, topic.description
    return True, ""


def check_answer_safety(answer: str) -> Tuple[bool, str]:
    """Проверить ответ на запрещённые паттерны."""
    for pattern in FORBIDDEN_ANSWER_PATTERNS:
        if re.search(pattern, answer, re.IGNORECASE):
            return False, f"Forbidden pattern: {pattern}"
    if len(answer) > 10000:
        return False, "Answer too long"
    return True, ""


def sanitize_answer(answer: str) -> str:
    """Очистить ответ от потенциально опасного контента."""
    answer = re.sub(r"<\s*script[^>]*>.*?</script>", "", answer,
                    flags=re.IGNORECASE | re.DOTALL)
    answer = re.sub(r"javascript:", "", answer, flags=re.IGNORECASE)
    return answer.strip()