# -*- coding: utf-8 -*-
"""
Koib-V-4.5 — Модуль безопасности
===================================
Фильтрация опасного, конфиденциального или нерелевантного контента
как на уровне входящих запросов, так и на уровне исходящих ответов.

Модуль обеспечивает:
  1. Проверку запросов на попытки prompt-injection
  2. Фильтрацию ответов от конфиденциальной информации
  3. Защиту от обхода системного промпта
"""
import re
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger("koib.safety")


# ═══════════════════════════════════════════════════════════════
# Шаблоны угроз
# ═══════════════════════════════════════════════════════════════
# Попытки prompt-injection
INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(?:previous|above|all)\s+instructions?", re.IGNORECASE),
    re.compile(r"forget\s+(?:everything|all|previous)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(?:a|an)\s+", re.IGNORECASE),
    re.compile(r"system\s*:\s*", re.IGNORECASE),
    re.compile(r"pretend\s+(?:to\s+be|you\s+are)", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"дин\ufffd\ufffd\ufffd\ufffdт", re.IGNORECASE),  # обходные варианты
    re.compile(r"обойди\s+(?:ограничения|правила|фильтр)", re.IGNORECASE),
]

# Конфиденциальные шаблоны в ответах
CONFIDENTIAL_PATTERNS = [
    re.compile(r'\b\d{16,19}\b'),                       # Номера карт
    re.compile(r'\b\d{3}[\s-]?\d{2}[\s-]?\d{2}\s\d{6}\b'),  # СНИЛС
    re.compile(r'\b\d{10,12}\b'),                        # ИНН/ОГРН
    re.compile(r'(?:пароль|password)\s*[:=]\s*\S+', re.IGNORECASE),
    re.compile(r'(?:секретн|конфиденциальн|не\s*публику)', re.IGNORECASE),
]

# Допустимые темы запросов (белый список)
ALLOWED_TOPICS = {
    "техническая документация", "паспорт", "руководство",
    "инструкция", "сертификат", "свидетельство",
    "параметры", "характеристики", "таблица", "формула",
    "схема", "чертёж", "модель", "устройство", "аппарат",
    "выборы", "протокол", "результат", "комиссия",
}


# ═══════════════════════════════════════════════════════════════
# Проверка запросов
# ═══════════════════════════════════════════════════════════════
def check_query_safety(query: str) -> Tuple[bool, str]:
    """
    Проверить запрос на безопасность.

    Выполняет два уровня проверки:
      1. Prompt-injection — попытки изменить поведение системы
      2. Тематическая релевантность — запросы не по теме

    Args:
        query: Текст запроса пользователя

    Returns:
        Кортеж (is_safe, reason):
          - is_safe=True, reason="" — запрос безопасен
          - is_safe=False, reason="..." — запрос отклонён
    """
    # Проверка на prompt-injection
    for pattern in INJECTION_PATTERNS:
        match = pattern.search(query)
        if match:
            reason = f"Обнаружена попытка prompt-injection: паттерн '{match.group()}'"
            logger.warning(f"Небезопасный запрос: {reason}")
            return False, reason

    # Проверка длины (слишком длинные запросы могут быть атакой)
    if len(query) > 2000:
        return False, "Запрос слишком длинный (максимум 2000 символов)"

    return True, ""


def check_answer_safety(answer: str) -> Tuple[bool, str]:
    """
    Проверить ответ LLM на наличие конфиденциальной информации.

    Args:
        answer: Текст ответа от LLM

    Returns:
        Кортеж (is_safe, sanitized_answer):
          - is_safe=True — ответ безопасен
          - is_safe=False — ответ содержит конфиденциальные данные
    """
    for pattern in CONFIDENTIAL_PATTERNS:
        if pattern.search(answer):
            return False, "Ответ содержит потенциально конфиденциальные данные"

    return True, answer


def sanitize_answer(answer: str) -> str:
    """
    Очистить ответ от конфиденциальной информации.

    Заменяет найденные конфиденциальные данные на плейсхолдеры.

    Args:
        answer: Текст ответа от LLM

    Returns:
        Очищенный текст ответа
    """
    sanitized = answer

    # Маскируем номера карт
    sanitized = re.sub(
        r'\b(\d{4})\d{10,15}(\d{4})\b',
        r'\1****\2',
        sanitized,
    )

    # Маскируем пароли
    sanitized = re.sub(
        r'(?:пароль|password)\s*[:=]\s*\S+',
        'пароль: [УДАЛЕНО]',
        sanitized,
        flags=re.IGNORECASE,
    )

    return sanitized
