# -*- coding: utf-8 -*-
"""
Koib-V-4.5 — Модуль оценки качества RAG
==========================================
Автоматическая оценка через LLM-as-Judge (GigaChat/OpenAI).

Метрики:
  1. Faithfulness       — верность ответа контексту (0-1)
  2. Answer Relevancy   — релевантность ответа вопросу (0-1)
  3. Context Precision  — точность: доля полезных чанков (0-1)
  4. Context Recall     — полнота: покрытие нужных фактов (0-1)
  5. Token F1           — токен-совпадение с эталоном (если задан)

Итоговый RAG-score — среднее четырёх LLM-метрик.
"""
import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

from config import LLM_PROVIDER, METADATA_DIR

logger = logging.getLogger("koib.evaluation")


# ═══════════════════════════════════════════════════════════════
# Структура результата оценки
# ═══════════════════════════════════════════════════════════════
@dataclass
class EvalResult:
    """
    Результат оценки одного вопроса.

    Attributes:
        question_id:       Идентификатор вопроса
        question:          Текст вопроса
        category:          Категория вопроса
        koib_model:        Модель устройства
        answer:            Ответ системы
        reference_answer:  Эталонный ответ (если есть)
        context_chunks:    Количество контекстных фрагментов
        faithfulness:      Верность ответа контексту (0-1)
        answer_relevancy:  Релевантность ответа вопросу (0-1)
        context_precision: Точность контекста (0-1)
        context_recall:    Полнота контекста (0-1)
        token_f1:          F1 по токенам с эталоном (0-1)
        has_reference:     Есть ли эталонный ответ
        error:             Ошибка (если была)
        latency_sec:       Время генерации ответа (секунды)
    """
    question_id: str
    question: str
    category: str = ""
    koib_model: str = ""
    answer: str = ""
    reference_answer: str = ""
    context_chunks: int = 0
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    token_f1: float = 0.0
    has_reference: bool = False
    error: Optional[str] = None
    latency_sec: float = 0.0

    @property
    def rag_score(self) -> float:
        """Итоговый RAG-score: среднее 4 LLM-метрик."""
        vals = [
            self.faithfulness, self.answer_relevancy,
            self.context_precision, self.context_recall,
        ]
        return round(sum(vals) / len(vals), 3) if vals else 0.0


# ═══════════════════════════════════════════════════════════════
# Промпты для LLM-судьи
# ═══════════════════════════════════════════════════════════════
PROMPT_FAITHFULNESS = """Ты — строгий судья качества AI-ответов. Оцени ВЕРНОСТЬ ответа относительно контекста.

ВОПРОС: {question}

КОНТЕКСТ (извлечённые фрагменты документации):
{context}

ОТВЕТ СИСТЕМЫ:
{answer}

Критерий — Faithfulness (Верность):
Содержит ли ответ ТОЛЬКО информацию из контекста? Нет ли в нём домыслов?

Оцени по шкале от 0 до 10:
10 — ответ полностью основан на контексте
5 — частично из контекста, частично домыслы
0 — ответ полностью придуман

Ответь ТОЛЬКО одним числом от 0 до 10."""

PROMPT_ANSWER_RELEVANCY = """Ты — строгий судья качества AI-ответов. Оцени РЕЛЕВАНТНОСТЬ ответа вопросу.

ВОПРОС: {question}

ОТВЕТ СИСТЕМЫ:
{answer}

Критерий — Answer Relevancy (Релевантность):
Отвечает ли ответ напрямую на поставленный вопрос?

Оцени по шкале от 0 до 10:
10 — ответ точно и полно отвечает на вопрос
5 — частично отвечает, много лишнего
0 — ответ не по теме

Ответь ТОЛЬКО одним числом от 0 до 10."""

PROMPT_CONTEXT_PRECISION = """Ты — строгий судья качества AI-ответов. Оцени ТОЧНОСТЬ найденного контекста.

ВОПРОС: {question}

НАЙДЕННЫЕ ФРАГМЕНТЫ ДОКУМЕНТАЦИИ:
{context}

Критерий — Context Precision (Точность контекста):
Какая доля фрагментов действительно нужна для ответа на вопрос?

Оцени по шкале от 0 до 10:
10 — все фрагменты релевантны
5 — примерно половина по теме
0 — все нерелевантны

Ответь ТОЛЬКО одним числом от 0 до 10."""

PROMPT_CONTEXT_RECALL = """Ты — строгий судья качества AI-ответов. Оцени ПОЛНОТУ найденного контекста.

ВОПРОС: {question}

ЭТАЛОННЫЙ ОТВЕТ: {reference}

НАЙДЕННЫЕ ФРАГМЕНТЫ ДОКУМЕНТАЦИИ:
{context}

Критерий — Context Recall (Полнота контекста):
Содержит ли найденный контекст достаточно информации для полного ответа?

Оцени по шкале от 0 до 10:
10 — контекст содержит всё необходимое
5 — контекст содержит часть нужной информации
0 — контекст совсем не помогает

Ответь ТОЛЬКО одним числом от 0 до 10."""


# ═══════════════════════════════════════════════════════════════
# Утилиты
# ═══════════════════════════════════════════════════════════════
def _normalize_text(text: str) -> str:
    """Нормализация текста для токен-сравнения."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text


def token_f1(prediction: str, reference: str) -> float:
    """
    F1-мера по токенам (без LLM).

    Разбивает тексты на токены и вычисляет precision, recall, F1.
    """
    pred_tokens = set(_normalize_text(prediction).split())
    ref_tokens = set(_normalize_text(reference).split())

    if not ref_tokens:
        return 0.0

    common = pred_tokens & ref_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens) if pred_tokens else 0
    recall = len(common) / len(ref_tokens)
    return round(2 * precision * recall / (precision + recall), 3)


def _extract_score(text: str) -> float:
    """
    Извлечь числовую оценку (0-10) из ответа LLM
    и нормировать до диапазона 0-1.
    """
    nums = re.findall(r"\b(\d+(?:\.\d+)?)\b", text)
    for n in nums:
        val = float(n)
        if 0 <= val <= 10:
            return round(val / 10.0, 3)
    return 0.0


# ═══════════════════════════════════════════════════════════════
# LLM-судья
# ═══════════════════════════════════════════════════════════════
class LLMJudge:
    """
    LLM-судья для оценки качества RAG.

    Использует тот же LLMClient, что и основная генерация,
    но с сокращённым лимитом токенов (50) для получения
    только числовой оценки.
    """

    def __init__(self, provider: Optional[str] = None):
        from .generation import LLMClient
        self.llm = LLMClient(provider=provider or LLM_PROVIDER)

    def score(self, prompt: str) -> float:
        """
        Получить оценку от 0 до 1 через LLM.

        Отправляет промпт судье и извлекает числовую оценку.
        """
        try:
            response = self.llm.generate(prompt, max_tokens=50)
            return _extract_score(response)
        except Exception as exc:
            logger.warning(f"Ошибка LLM-судьи: {exc}")
            return 0.0


# ═══════════════════════════════════════════════════════════════
# Основной оценщик
# ═══════════════════════════════════════════════════════════════
class RAGEvaluator:
    """
    Оценщик качества RAG-системы.

    Проводит оценку по 4 LLM-метрикам + Token F1:
      1. Faithfulness       — верность ответа контексту
      2. Answer Relevancy   — релевантность ответа вопросу
      3. Context Precision  — точность найденного контекста
      4. Context Recall     — полнота найденного контекста
      5. Token F1           — совпадение с эталоном (если задан)
    """

    def __init__(self, judge_provider: Optional[str] = None):
        self.judge = LLMJudge(provider=judge_provider)

    def evaluate_one(
        self,
        question: str,
        answer: str,
        context: str,
        reference: str = "",
        question_id: str = "",
        category: str = "",
        koib_model: str = "",
    ) -> EvalResult:
        """
        Оценить один вопрос.

        Args:
            question:    Текст вопроса
            answer:      Ответ системы
            context:     Контекст, переданный в LLM
            reference:   Эталонный ответ (для Context Recall и Token F1)
            question_id: Идентификатор вопроса
            category:    Категория вопроса
            koib_model:  Модель устройства

        Returns:
            EvalResult с оценками по всем метрикам
        """
        result = EvalResult(
            question_id=question_id,
            question=question,
            category=category,
            koib_model=koib_model,
            answer=answer,
            reference_answer=reference,
            has_reference=bool(reference),
        )

        logger.info(f"Оценка [{question_id}]: {question[:80]}...")

        # Faithfulness
        result.faithfulness = self.judge.score(
            PROMPT_FAITHFULNESS.format(
                question=question, context=context, answer=answer,
            )
        )

        # Answer Relevancy
        result.answer_relevancy = self.judge.score(
            PROMPT_ANSWER_RELEVANCY.format(question=question, answer=answer)
        )

        # Context Precision
        result.context_precision = self.judge.score(
            PROMPT_CONTEXT_PRECISION.format(question=question, context=context)
        )

        # Context Recall (только если есть эталон)
        if reference:
            result.context_recall = self.judge.score(
                PROMPT_CONTEXT_RECALL.format(
                    question=question, reference=reference, context=context,
                )
            )

        # Token F1
        if reference:
            result.token_f1 = token_f1(answer, reference)

        logger.info(
            f"  RAG-score: {result.rag_score}, "
            f"Faith: {result.faithfulness}, "
            f"Rel: {result.answer_relevancy}, "
            f"Prec: {result.context_precision}, "
            f"Rec: {result.context_recall}"
        )

        return result

    def evaluate_batch(
        self,
        questions: List[Dict[str, Any]],
        save_path: Optional[Path] = None,
    ) -> List[EvalResult]:
        """
        Оценить пакет вопросов.

        Args:
            questions:  Список словарей с ключами: question, answer, context, reference, ...
            save_path:  Путь для сохранения результатов (JSON)

        Returns:
            Список EvalResult
        """
        results: List[EvalResult] = []

        for i, q in enumerate(questions):
            try:
                result = self.evaluate_one(
                    question=q.get("question", ""),
                    answer=q.get("answer", ""),
                    context=q.get("context", ""),
                    reference=q.get("reference", ""),
                    question_id=q.get("question_id", str(i)),
                    category=q.get("category", ""),
                    koib_model=q.get("koib_model", ""),
                )
                results.append(result)
            except Exception as exc:
                logger.error(f"Ошибка оценки вопроса {i}: {exc}")
                results.append(EvalResult(
                    question_id=str(i),
                    question=q.get("question", ""),
                    error=str(exc),
                ))

        # Сохранение результатов
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            data = [asdict(r) for r in results]
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"Результаты оценки сохранены: {save_path}")

        return results
