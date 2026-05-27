# -*- coding: utf-8 -*-
"""
Koib-V-4.5 — Модуль валидации ответов
========================================
Проверка ответов LLM на достоверность, наличие источников
и семантическую согласованность с контекстом.

Три уровня проверки:
  1. Неуверенность — маркеры сомнения в ответе (критично)
  2. Источники — наличие цитат из документов (предупреждение)
  3. Семантика — косинусное сходство ответа с контекстом (предупреждение)

Если обнаружены маркеры неуверенности, ответ блокируется.
Если нет источников или низкое семантическое сходство —
статус 'review' (требует проверки).
"""
import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from config import VALIDATION_IGNORE_QUOTES, UNCERTAINTY_MIN_LENGTH

logger = logging.getLogger("koib.validation")


# ═══════════════════════════════════════════════════════════════
# Шаблоны неуверенности
# ═══════════════════════════════════════════════════════════════
UNCERTAINTY_PATTERNS = [
    r"\bвозможно\b",
    r"\bвероятно\b",
    r"\bпредполагает(?:ся|ся)\b",
    r"\bскорее всего\b",
    r"\bможет быть\b",
    r"\bпо-видимому\b",
    r"\bочевидно\b",
    r"\bкажется\b",
    r"\bнаверное\b",
    r"\bя думаю\b",
    r"\bя полагаю\b",
    r"\bне исключено\b",
    r"\bне исключать\b",
    r"\bвполне возможно\b",
]


# ═══════════════════════════════════════════════════════════════
# Шаблоны источников
# ═══════════════════════════════════════════════════════════════
SOURCE_PATTERNS = [
    r"\[Документ:\s*[^,\]]+,\s*стр\.\s*\d+\]",
    r"\[Документ:\s*[^,\]]+,\s*страница\s*\d+\]",
    r"\(источник:\s*[^)]+\)",
    r"\bсм\.\s+раздел\s+",
    r"\bсогласно\s+(?:технической\s+)?документации\b",
]


# ═══════════════════════════════════════════════════════════════
# Структуры данных
# ═══════════════════════════════════════════════════════════════
@dataclass
class ValidationCheck:
    """
    Результат одной проверки.

    Attributes:
        name:     Название проверки
        passed:   Пройдена ли проверка
        details:  Детали результата
        severity: Критичность: 'info', 'warning', 'critical'
    """
    name: str
    passed: bool
    details: str = ""
    severity: str = "info"

    # Дополнительное поле для семантической проверки
    semantic_similarity: float = 0.0


@dataclass
class ValidationResult:
    """
    Итоговый результат валидации ответа.

    Статусы:
      - 'approved'  — ответ прошёл все проверки
      - 'review'    — есть предупреждения, требует проверки
      - 'rejected'  — ответ содержит неуверенность, блокируется
    """
    status: str = "approved"
    checks: List[ValidationCheck] = field(default_factory=list)
    requires_review_reasons: List[str] = field(default_factory=list)
    uncertainty_found: Optional[str] = None
    sources_found: List[str] = field(default_factory=list)
    semantic_similarity: float = 1.0

    def add_check(self, check: ValidationCheck) -> None:
        """
        Добавить результат проверки и обновить итоговый статус.

        Логика обновления:
          - critical + failed → rejected
          - warning + failed → review (если не rejected)
        """
        self.checks.append(check)
        if not check.passed:
            if check.severity == "critical":
                self.status = "rejected"
            elif check.severity == "warning" and self.status != "rejected":
                if self.status == "approved":
                    self.status = "review"
                self.requires_review_reasons.append(check.details)

    def to_dict(self) -> Dict[str, Any]:
        """Сериализовать результат валидации в словарь."""
        return {
            "status": self.status,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "details": c.details,
                    "severity": c.severity,
                }
                for c in self.checks
            ],
            "requires_review_reasons": self.requires_review_reasons,
            "uncertainty_found": self.uncertainty_found,
            "sources_found": self.sources_found,
            "semantic_similarity": self.semantic_similarity,
        }


# ═══════════════════════════════════════════════════════════════
# Валидатор ответов
# ═══════════════════════════════════════════════════════════════
class AnswerValidator:
    """
    Валидатор ответов LLM.

    Выполняет три уровня проверки:
      1. Неуверенность — наличие маркеров сомнения (критично)
      2. Источники — наличие ссылок на документы (предупреждение)
      3. Семантика — косинусное сходство с контекстом (предупреждение)

    Эмбеддинги берутся из глобального синглтона get_global_embeddings(),
    что исключает передачу тяжёлого объекта через конструктор.
    """

    def __init__(
        self,
        embeddings=None,
        similarity_threshold: float = 0.75,
    ):
        self.embeddings = embeddings
        self.similarity_threshold = similarity_threshold

    def validate(
        self,
        answer: str,
        context_chunks: List[Any],
        query: str = "",
    ) -> ValidationResult:
        """
        Провести полную валидацию ответа.

        Args:
            answer:          Текст ответа от LLM
            context_chunks:  Контекстные фрагменты (RetrievalResult или dict)
            query:           Оригинальный запрос (для будущего использования)

        Returns:
            ValidationResult с итоговым статусом и деталями проверок
        """
        result = ValidationResult()

        # Проверка 1: Неуверенность
        uncertainty_check = self._check_uncertainty(answer)
        result.add_check(uncertainty_check)
        if not uncertainty_check.passed:
            result.uncertainty_found = uncertainty_check.details
            return result

        # Проверка 2: Источники
        sources_check = self._check_sources(answer)
        result.add_check(sources_check)
        result.sources_found = (
            sources_check.details.split("; ") if sources_check.passed else []
        )

        # Проверка 3: Семантическая согласованность
        if not self.embeddings:
            try:
                from src.indexing import get_global_embeddings
                self.embeddings = get_global_embeddings()
            except Exception:
                pass

        if self.embeddings and context_chunks:
            semantic_check = self._check_semantic_consistency(answer, context_chunks)
            result.add_check(semantic_check)
            result.semantic_similarity = semantic_check.semantic_similarity

        return result

    def _check_uncertainty(self, answer: str) -> ValidationCheck:
        """
        Проверка на маркеры неуверенности.

        Исключает цитаты в квадратных скобках и кавычках,
        чтобы не ложно срабатывать на контекстных ссылках.
        """
        if len(answer.strip()) < UNCERTAINTY_MIN_LENGTH:
            return ValidationCheck(
                name="uncertainty_check",
                passed=True,
                details="Ответ слишком короткий для проверки",
                severity="info",
            )

        # Удаляем цитаты перед проверкой
        text_to_check = answer
        if VALIDATION_IGNORE_QUOTES:
            text_to_check = re.sub(r'\[Документ:[^\]]*\]', '', text_to_check)
            text_to_check = re.sub(r'"[^"]*"', '', text_to_check)
            text_to_check = re.sub(r'«[^»]*»', '', text_to_check)

        text_lower = text_to_check.lower()
        found_markers = [
            m
            for p in UNCERTAINTY_PATTERNS
            for m in re.findall(p, text_lower, re.IGNORECASE)
        ]

        if found_markers:
            unique_markers = list(set(found_markers))[:3]
            return ValidationCheck(
                name="uncertainty_check",
                passed=False,
                details=f"Маркеры: {', '.join(unique_markers)}",
                severity="critical",
            )

        return ValidationCheck(
            name="uncertainty_check",
            passed=True,
            details="ОК",
            severity="info",
        )

    def _check_sources(self, answer: str) -> ValidationCheck:
        """
        Проверка на наличие ссылок на источники.

        Ищет стандартные форматы цитирования:
        [Документ: ..., стр. N], (источник: ...) и т.д.
        """
        found_sources = [
            m
            for p in SOURCE_PATTERNS
            for m in re.findall(p, answer, re.IGNORECASE)
        ]

        if found_sources:
            unique_sources = list(set(found_sources))[:5]
            return ValidationCheck(
                name="sources_check",
                passed=True,
                details="; ".join(unique_sources),
                severity="info",
            )

        return ValidationCheck(
            name="sources_check",
            passed=False,
            details="Ссылки на источники не найдены",
            severity="warning",
        )

    def _check_semantic_consistency(
        self,
        answer: str,
        context_chunks: List[Any],
    ) -> ValidationCheck:
        """
        Семантическая проверка согласованности ответа с контекстом.

        Вычисляет косинусное сходство между эмбеддингом ответа
        и эмбеддингами первых 5 контекстных фрагментов.
        Берётся максимальное сходство.
        """
        try:
            answer_embedding = self._get_embedding(answer)
            if not answer_embedding:
                return ValidationCheck(
                    name="semantic_check",
                    passed=False,
                    details="Не удалось вычислить эмбеддинг ответа",
                    severity="info",
                )

            context_texts = [
                c.content if hasattr(c, 'content') else c.get('content', '')
                for c in context_chunks[:5]
            ]
            max_similarity = 0.0

            for ctx_text in context_texts:
                ctx_embedding = self._get_embedding(ctx_text)
                if ctx_embedding:
                    similarity = self._cosine_similarity(answer_embedding, ctx_embedding)
                    max_similarity = max(max_similarity, similarity)

            passed = max_similarity >= self.similarity_threshold
            return ValidationCheck(
                name="semantic_check",
                passed=passed,
                details=f"Сходство: {max_similarity:.3f}",
                severity="warning",
                semantic_similarity=max_similarity,
            )
        except Exception as exc:
            return ValidationCheck(
                name="semantic_check",
                passed=False,
                details=str(exc),
                severity="info",
            )

    def _get_embedding(self, text: str) -> Optional[list]:
        """Получить эмбеддинг текста через глобальный синглтон."""
        try:
            if hasattr(self.embeddings, 'embed_query'):
                return self.embeddings.embed_query(text)
            elif hasattr(self.embeddings, 'embed_documents'):
                return self.embeddings.embed_documents([text])[0]
        except Exception:
            return None
        return None

    def _cosine_similarity(self, vec1: list, vec2: list) -> float:
        """Вычислить косинусное сходство между двумя векторами."""
        import numpy as np
        v1, v2 = np.array(vec1), np.array(vec2)
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))


def get_blocked_response() -> str:
    """
    Стандартный ответ для заблокированных ответов.

    Используется, когда валидатор обнаруживает маркеры
    неуверенности и отклоняет ответ LLM.
    """
    return "По вашему запросу не найдено точного ответа в официальных источниках."
