# -*- coding: utf-8 -*-
"""
Koib-V-4.5 — Модуль генерации ответов
========================================
LLM-клиенты для GigaChat, OpenAI и локальных моделей.
Формирование промптов с контекстом, генерация ответов
с жёсткими таймаутами для стабильной работы на слабом VPS.

Ключевые отличия от v4.3:
  - Таймауты на все сетевые запросы (10с авторизация, 30с генерация)
  - Сокращённые лимиты токенов (700 вместо 2048)
  - Единый интерфейс LLMClient для всех провайдеров
  - AnswerValidator интегрирован в пайплайн генерации
"""
import logging
import requests
import urllib3
from typing import List, Dict, Any, Optional

from .retrieval import RetrievalResult
from config import (
    LLM_PROVIDER, GIGACHAT_CREDENTIALS, GIGACHAT_MODEL,
    GIGACHAT_TEMPERATURE, GIGACHAT_MAX_TOKENS, GIGACHAT_TIMEOUT,
    GIGACHAT_VERIFY_SSL, OPENAI_API_KEY, OPENAI_LLM_MODEL,
    OPENAI_TEMPERATURE, OPENAI_MAX_TOKENS, LOCAL_LLM_MODEL,
    LOCAL_LLM_URL,
)

logger = logging.getLogger("koib.generation")


# ═══════════════════════════════════════════════════════════════
# Системный промпт
# ═══════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """Ты — эксперт-ассистент по технической документации.
Твоя задача — отвечать на вопросы пользователя строго на основе предоставленного контекста.

ПРАВИЛА ОТВЕТА:
1. Опирайся ТОЛЬКО на предоставленный контекст.
2. Цитируй источники в формате: [Документ: {имя_файла}, стр. {номер}].
3. Таблицы выводи в Markdown-формате.
4. Отвечай кратко и структурировано.
5. Если в контексте нет ответа — скажи об этом прямо."""


# ═══════════════════════════════════════════════════════════════
# Формирование промпта
# ═══════════════════════════════════════════════════════════════
def build_prompt(query: str, results: List[RetrievalResult]) -> str:
    """
    Сформировать промпт для LLM на основе результатов поиска.

    Включает каждый фрагмент контекста с метаданными источника
    и форматирует его для удобного восприятия моделью.

    Args:
        query:   Вопрос пользователя
        results: Список результатов поиска

    Returns:
        Готовый промпт для LLM
    """
    context_parts = []
    for i, r in enumerate(results, 1):
        context_parts.append(f"--- Фрагмент {i} ---\n{r.to_context_string()}\n")

    return (
        f"КОНТЕКСТ:\n{''.join(context_parts)}\n"
        f"ВОПРОС:\n{query}\n"
        f"Отвечай строго по контексту."
    )


# ═══════════════════════════════════════════════════════════════
# LLM-клиент
# ═══════════════════════════════════════════════════════════════
class LLMClient:
    """
    Унифицированный клиент для работы с LLM.

    Поддерживаемые провайдеры:
      - "gigachat" — GigaChat API (Сбер)
      - "openai"   — OpenAI API
      - "local"    — Ollama / llama-cpp

    Все сетевые запросы имеют жёсткие таймауты для предотвращения
    зависаний при работе на слабом VPS.
    """

    def __init__(self, provider: Optional[str] = None):
        self.provider = provider or LLM_PROVIDER

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = GIGACHAT_MAX_TOKENS,
        temperature: float = GIGACHAT_TEMPERATURE,
    ) -> str:
        """
        Сгенерировать ответ через LLM.

        Args:
            prompt:         Текст запроса
            system_prompt:  Системный промпт (по умолчанию SYSTEM_PROMPT)
            max_tokens:     Максимум токенов в ответе
            temperature:    Температура генерации

        Returns:
            Текст ответа от LLM
        """
        sys_prompt = system_prompt or SYSTEM_PROMPT

        if self.provider == "gigachat":
            return self._generate_gigachat(prompt, sys_prompt, max_tokens, temperature)
        elif self.provider == "openai":
            return self._generate_openai(prompt, sys_prompt, max_tokens, temperature)
        elif self.provider == "local":
            return self._generate_local(prompt, sys_prompt, max_tokens, temperature)
        else:
            return f"Провайдер '{self.provider}' не поддерживается."

    def _generate_gigachat(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """
        Генерация через GigaChat API (Сбер).

        Выполняет OAuth2-авторизацию и отправляет запрос к чат-модели.
        При 401 автоматически обновляет токен и повторяет запрос.
        """
        if not GIGACHAT_CREDENTIALS:
            return "Ошибка: GIGACHAT_CREDENTIALS не заданы."

        if not GIGACHAT_VERIFY_SSL:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        try:
            # Шаг 1: Авторизация OAuth2
            auth_resp = requests.post(
                "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
                headers={
                    "Authorization": f"Basic {GIGACHAT_CREDENTIALS}",
                    "RqUID": "koib-rag-001",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                data={"scope": "GIGACHAT_API_PERS"},
                verify=GIGACHAT_VERIFY_SSL,
                timeout=10,
            )

            if auth_resp.status_code != 200:
                return f"Ошибка авторизации GigaChat: {auth_resp.status_code}"

            token = auth_resp.json()["access_token"]

            # Шаг 2: Запрос к чат-модели
            chat_resp = requests.post(
                "https://gigachat.devices.sberbank.ru/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": GIGACHAT_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                verify=GIGACHAT_VERIFY_SSL,
                timeout=GIGACHAT_TIMEOUT,
            )

            if chat_resp.status_code == 401:
                # Токен истёк — повторная авторизация
                auth_resp = requests.post(
                    "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
                    headers={
                        "Authorization": f"Basic {GIGACHAT_CREDENTIALS}",
                        "RqUID": "koib-rag-001",
                        "Content-Type": "application/x-www-form-urlencoded",
                    },
                    data={"scope": "GIGACHAT_API_PERS"},
                    verify=GIGACHAT_VERIFY_SSL,
                    timeout=10,
                )
                if auth_resp.status_code != 200:
                    return f"Ошибка повторной авторизации: {auth_resp.status_code}"
                token = auth_resp.json()["access_token"]

                chat_resp = requests.post(
                    "https://gigachat.devices.sberbank.ru/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": GIGACHAT_MODEL,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    },
                    verify=GIGACHAT_VERIFY_SSL,
                    timeout=GIGACHAT_TIMEOUT,
                )

            if chat_resp.status_code != 200:
                return f"Ошибка API GigaChat: {chat_resp.status_code}"

            return chat_resp.json()["choices"][0]["message"]["content"].strip()

        except requests.exceptions.Timeout:
            return "Таймаут запроса к GigaChat."
        except requests.exceptions.ConnectionError:
            return "Ошибка соединения с GigaChat."
        except Exception as e:
            return f"Ошибка генерации GigaChat: {e}"

    def _generate_openai(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """
        Генерация через OpenAI API.

        Использует библиотеку openai для совместимости
        с последними версиями API.
        """
        if not OPENAI_API_KEY:
            return "Ошибка: OPENAI_API_KEY не задан."

        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)

            response = client.chat.completions.create(
                model=OPENAI_LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=GIGACHAT_TIMEOUT,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"Ошибка генерации OpenAI: {e}"

    def _generate_local(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """
        Генерация через локальную LLM (Ollama / llama-cpp).

        Отправляет запрос к Ollama API, запущенному на localhost.
        """
        try:
            response = requests.post(
                f"{LOCAL_LLM_URL}/api/generate",
                json={
                    "model": LOCAL_LLM_MODEL,
                    "prompt": f"{system_prompt}\n\n{prompt}",
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature,
                    },
                },
                timeout=GIGACHAT_TIMEOUT,
            )

            if response.status_code != 200:
                return f"Ошибка локальной LLM: {response.status_code}"

            return response.json().get("response", "").strip()

        except requests.exceptions.Timeout:
            return "Таймаут запроса к локальной LLM."
        except requests.exceptions.ConnectionError:
            return "Локальная LLM недоступна. Запустите Ollama."
        except Exception as e:
            return f"Ошибка локальной LLM: {e}"


# ═══════════════════════════════════════════════════════════════
# Генератор ответов (полный пайплайн)
# ═══════════════════════════════════════════════════════════════
class AnswerGenerator:
    """
    Полный пайплайн генерации ответа: поиск → промпт → LLM → валидация.

    Объединяет HybridRetriever, LLMClient и AnswerValidator
    в единый процесс обработки запроса пользователя.
    """

    def __init__(self):
        from .retrieval import HybridRetriever
        self.retriever = HybridRetriever()
        self.llm = LLMClient()

    def answer(
        self,
        query: str,
        k: int = 3,
        model_filter: str = "",
        validate: bool = True,
    ) -> Dict[str, Any]:
        """
        Сгенерировать ответ на запрос пользователя.

        Args:
            query:        Вопрос пользователя
            k:            Количество контекстных фрагментов
            model_filter: Фильтр по модели устройства
            validate:     Проводить ли валидацию ответа

        Returns:
            Словарь с ключами: answer, sources, validation, latency
        """
        import time
        t0 = time.time()

        # Шаг 1: Поиск релевантных фрагментов
        results = self.retriever.search(query, k=k, model_filter=model_filter)

        if not results:
            return {
                "answer": "По вашему запросу не найдено релевантных фрагментов в документации.",
                "sources": [],
                "validation": None,
                "latency": time.time() - t0,
            }

        # Шаг 2: Формирование промпта
        prompt = build_prompt(query, results)

        # Шаг 3: Генерация ответа
        answer = self.llm.generate(prompt)

        # Шаг 4: Валидация (опционально)
        validation_result = None
        if validate:
            try:
                from .validation import AnswerValidator
                validator = AnswerValidator()
                validation_result = validator.validate(answer, results, query)
                validation_result = validation_result.to_dict()
            except Exception as exc:
                logger.warning(f"Ошибка валидации: {exc}")

        # Шаг 5: Формирование источников
        sources = [
            {
                "source": r.source,
                "page": r.page,
                "heading": r.heading,
                "chunk_type": r.chunk_type,
            }
            for r in results
        ]

        latency = time.time() - t0
        logger.info(f"Ответ сгенерирован за {latency:.2f}с")

        return {
            "answer": answer,
            "sources": sources,
            "validation": validation_result,
            "latency": latency,
        }
