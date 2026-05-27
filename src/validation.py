# -*- coding: utf-8 -*-
"""
Koib-V-4.7 — Модуль валидации ответов
★ ПЕРЕПИСАНО: LLM-as-Judge вместо Regex-маркеров неуверенности
★ ДОБАВЛЕНО: валидация цитат против реальных источников из retrieval
★ СОХРАНЕНО: семантическая проверка через эмбеддинги
"""
import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from config import (
    VALIDATION_IGNORE_QUOTES, UNCERTAINTY_MIN_LENGTH,
    VALIDATION_USE_LLM_JUDGE, VALIDATION_CHECK_CITATIONS,
)

logger = logging.getLogger("koib.validation")


# ═══════════════════════════════════════════════════════════════
# LLM-as-Judge промпты (вместо хрупких Regex-ов)
# ═══════════════════════════════════════════════════════════════
PROMPT_FACTUALITY_CHECK = """Ты — строгий валидатор RAG-системы. Проверь, основан ли ОТВЕТ исключительно на информации из КОНТЕКСТА.

<retrieved_context>
{context}
</retrieved_context>

<generated_answer>
{answer}
</generated_answer>

Критерии проверки:
1. Содержит ли ответ факты, которых НЕТ в контексте (галлюцинации)?
2. Противоречит ли ответ информации в контексте?
3. Использует ли ответ слова неуверенности как свои ("возможно", "вероятно", "я думаю"), 
   а НЕ как цитату из контекста?

ВАЖНО: если слова неуверенности присутствуют В САМОМ КОНТЕКСТЕ (например: 
"Возможно использование резервного источника питания") и ответ их корректно цитирует — 
это НЕ является ошибкой.

Ответь СТРОГО в формате JSON (и только JSON, без пояснений):
{{"is_factual": true/false, "issues": ["список", "проблем"]}}

Если проблем нет: {{"is_factual": true, "issues": []}}"""


# ═══════════════════════════════════════════════════════════════
# Regex для извлечения цитат из ответа (для верификации)
# ═══════════════════════════════════════════════════════════════
CITATION_PATTERN = re.compile(
    r"\[Документ:\s*([^,\]]+?),\s*стр(?:аница)?\.?\s*(\d+)\]",
    re.IGNORECASE,
)


# ═══════════════════════════════════════════════════════════════
# Структуры данных
# ═══════════════════════════════════════════════════════════════
@dataclass
class ValidationCheck:
    name: str
    passed: bool
    details: str = ""
    severity: str = "info"
    semantic_similarity: float = 0.0


@dataclass
class ValidationResult:
    status: str = "approved"
    checks: List[ValidationCheck] = field(default_factory=list)
    requires_review_reasons: List[str] = field(default_factory=list)
    hallucination_found: Optional[str] = None
    sources_found: List[str] = field(default_factory=list)
    fake_citations: List[str] = field(default_factory=list)
    semantic_similarity: float = 1.0

    def add_check(self, check: ValidationCheck) -> None:
        self.checks.append(check)
        if not check.passed:
            if check.severity == "critical":
                self.status = "rejected"
            elif check.severity == "warning" and self.status != "rejected":
                if self.status == "approved":
                    self.status = "review"
                self.requires_review_reasons.append(check.details)

    def to_dict(self) -> Dict[str, Any]:
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
            "hallucination_found": self.hallucination_found,
            "sources_found": self.sources_found,
            "fake_citations": self.fake_citations,
            "semantic_similarity": self.semantic_similarity,
        }


# ═══════════════════════════════════════════════════════════════
# Валидатор ответов
# ═══════════════════════════════════════════════════════════════
class AnswerValidator:
    """
    Валидатор на базе LLM-as-Judge.
    Проверяет:
    1. Фактичность (нет ли галлюцинаций) — через LLM-судью
    2. Источники (есть ли цитаты) — regex
    3. Подлинность цитат (совпадают ли с retrieval) — верификация
    4. Семантика (cosine similarity с контекстом) — эмбеддинги
    """
    def __init__(
        self,
        embeddings=None,
        similarity_threshold: float = 0.75,
    ):
        self.embeddings = embeddings
        self.similarity_threshold = similarity_threshold
        self._llm_judge = None

    def _get_llm_judge(self):
        if self._llm_judge is None:
            try:
                from .generation import LLMClient
                self._llm_judge = LLMClient()
            except Exception as exc:
                logger.warning(f"Не удалось создать LLM-судью: {exc}")
        return self._llm_judge

    def validate(
        self,
        answer: str,
        context_chunks: List[Any],
        query: str = "",
    ) -> ValidationResult:
        result = ValidationResult()

        if len(answer.strip()) < UNCERTAINTY_MIN_LENGTH:
            result.add_check(ValidationCheck(
                name="length_check",
                passed=True,
                details="Ответ слишком короткий для полной проверки",
                severity="info",
            ))
            return result

        # ★ ПРОВЕРКА 1: Фактичность через LLM-as-Judge (вместо Regex-маркеров)
        if VALIDATION_USE_LLM_JUDGE:
            factuality_check = self._check_factuality_llm(answer, context_chunks)
            result.add_check(factuality_check)
            if not factuality_check.passed:
                result.hallucination_found = factuality_check.details
                return result  # Критический провал — дальше не проверяем

        # ПРОВЕРКА 2: Наличие цитат
        sources_check = self._check_sources(answer)
        result.add_check(sources_check)
        result.sources_found = (
            sources_check.details.split("; ") if sources_check.passed else []
        )

        # ★ ПРОВЕРКА 3: Подлинность цитат (не выдуманы ли)
        if VALIDATION_CHECK_CITATIONS and context_chunks:
            citation_check = self._check_citations_authenticity(answer, context_chunks)
            result.add_check(citation_check)
            if not citation_check.passed:
                result.fake_citations = citation_check.details.split("; ")

        # ПРОВЕРКА 4: Семантическая согласованность
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

    def _check_factuality_llm(self, answer: str, context_chunks: List[Any]) -> ValidationCheck:
        """
        ★ LLM-as-Judge: проверка на галлюцинации и неуместную неуверенность.
        Заменяет хрупкие UNCERTAINTY_PATTERNS.
        """
        judge = self._get_llm_judge()
        if judge is None:
            return ValidationCheck(
                name="factuality_check",
                passed=True,
                details="LLM-судья недоступен, пропуск",
                severity="info",
            )

        # Собираем контекст для проверки
        context_text = "\n\n".join(
            c.to_context_string() if hasattr(c, 'to_context_string')
            else (c.get('content', '') if isinstance(c, dict) else str(c))
            for c in context_chunks[:5]
        )

        prompt = PROMPT_FACTUALITY_CHECK.format(
            context=context_text[:6000],  # Ограничиваем длину
            answer=answer[:3000],
        )

        try:
            response = judge.generate(prompt, max_tokens=300, temperature=0.0)
            verdict = self._parse_llm_verdict(response)

            if verdict is None:
                return ValidationCheck(
                    name="factuality_check",
                    passed=True,
                    details=f"Не удалось распарсить ответ судьи: {response[:100]}",
                    severity="info",
                )

            is_factual = verdict.get("is_factual", True)
            issues = verdict.get("issues", [])

            if not is_factual:
                return ValidationCheck(
                    name="factuality_check",
                    passed=False,
                    details="; ".join(issues[:3]) if issues else "Обнаружены галлюцинации",
                    severity="critical",
                )
            return ValidationCheck(
                name="factuality_check",
                passed=True,
                details="ОК: ответ основан на контексте",
                severity="info",
            )
        except Exception as exc:
            logger.warning(f"Ошибка LLM-валидации: {exc}")
            return ValidationCheck(
                name="factuality_check",
                passed=True,
                details=f"Ошибка судьи: {exc}",
                severity="info",
            )

    def _parse_llm_verdict(self, response: str) -> Optional[Dict]:
        """Извлечь JSON из ответа LLM-судьи."""
        if not response:
            return None
        # Пытаемся найти JSON в ответе
        json_match = re.search(r'\{[^{}]*"is_factual"[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                import json
                return json.loads(json_match.group(0))
            except Exception:
                pass
        # Fallback: ищем ключевые слова
        response_lower = response.lower()
        if "is_factual" in response_lower:
            if '"is_factual": true' in response_lower or '"is_factual":true' in response_lower:
                return {"is_factual": True, "issues": []}
            if '"is_factual": false' in response_lower or '"is_factual":false' in response_lower:
                return {"is_factual": False, "issues": ["Обнаружены проблемы (fallback parse)"]}
        return None

    def _check_sources(self, answer: str) -> ValidationCheck:
        """Проверка на наличие цитат (regex)."""
        citations = CITATION_PATTERN.findall(answer)
        if citations:
            unique = list({f"{doc.strip()}, стр. {page}" for doc, page in citations})[:5]
            return ValidationCheck(
                name="sources_check",
                passed=True,
                details="; ".join(unique),
                severity="info",
            )
        return ValidationCheck(
            name="sources_check",
            passed=False,
            details="Ссылки на источники не найдены",
            severity="warning",
        )

    def _check_citations_authenticity(
        self,
        answer: str,
        context_chunks: List[Any],
    ) -> ValidationCheck:
        """
        ★ КРИТИЧНО: проверка, что цитаты в ответе соответствуют реальным
        источникам из retrieval. LLM не может выдумать "fake.pdf, стр. 99".
        """
        # Собираем множество реальных (source, page) из retrieval
        real_sources = set()
        for c in context_chunks:
            if hasattr(c, 'source'):
                source = c.source
                page = c.page
            elif isinstance(c, dict):
                source = c.get('source', '')
                page = c.get('page', 0)
            else:
                continue
            if source:
                # Нормализуем имя файла (убираем путь, приводим к нижнему регистру)
                source_name = source.split('/')[-1].split('\\')[-1].lower()
                real_sources.add((source_name, int(page) if page else 0))

        # Извлекаем цитаты из ответа
        cited = CITATION_PATTERN.findall(answer)
        if not cited:
            return ValidationCheck(
                name="citations_authenticity",
                passed=True,
                details="Нет цитат для проверки",
                severity="info",
            )

        fake_citations = []
        for doc, page in cited:
            doc_clean = doc.strip().lower()
            try:
                page_num = int(page)
            except ValueError:
                fake_citations.append(f"{doc}, стр. {page}")
                continue

            # Ищем совпадение (допускаем частичное совпадение по имени файла)
            matched = False
            for real_src, real_page in real_sources:
                if real_page == page_num:
                    # Проверяем: имя файла совпадает точно или как подстрока
                    if doc_clean == real_src or doc_clean in real_src or real_src in doc_clean:
                        matched = True
                        break
            if not matched:
                fake_citations.append(f"[Документ: {doc}, стр. {page}]")

        if fake_citations:
            unique_fake = list(set(fake_citations))[:3]
            return ValidationCheck(
                name="citations_authenticity",
                passed=False,
                details="; ".join(unique_fake),
                severity="warning",  # warning, т.к. LLM мог сократить имя файла
            )
        return ValidationCheck(
            name="citations_authenticity",
            passed=True,
            details=f"Все {len(cited)} цитат(ы) верифицированы",
            severity="info",
        )

    def _check_semantic_consistency(
        self,
        answer: str,
        context_chunks: List[Any],
    ) -> ValidationCheck:
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
        try:
            if hasattr(self.embeddings, 'embed_query'):
                return self.embeddings.embed_query(text)
            elif hasattr(self.embeddings, 'embed_documents'):
                return self.embeddings.embed_documents([text])[0]
        except Exception:
            return None
        return None

    def _cosine_similarity(self, vec1: list, vec2: list) -> float:
        import numpy as np
        v1, v2 = np.array(vec1), np.array(vec2)
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))


def get_blocked_response() -> str:
    return "По вашему запросу не найдено точного ответа в официальных источниках."
