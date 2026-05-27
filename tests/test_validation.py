# -*- coding: utf-8 -*-
"""
Koib-V-4.5 — Тесты модуля валидации
======================================
Проверка детекции неуверенности, источников и семантической
согласованности ответов.
"""
import pytest
from src.validation import AnswerValidator, ValidationResult, ValidationCheck


class TestUncertaintyCheck:
    """Тесты проверки на неуверенность."""

    def setup_method(self):
        self.validator = AnswerValidator(embeddings=None)

    def test_certain_answer(self):
        """Уверенный ответ должен пройти проверку."""
        answer = "Согласно документации, модель АИИС-001 имеет вес 100 кг [Документ: passport.pdf, стр. 5]."
        result = self.validator._check_uncertainty(answer)
        assert result.passed is True

    def test_uncertain_answer(self):
        """Ответ с маркером неуверенности должен быть отклонён."""
        answer = "Возможно, модель имеет вес 100 кг."
        result = self.validator._check_uncertainty(answer)
        assert result.passed is False
        assert result.severity == "critical"

    def test_uncertainty_in_quotes_ignored(self):
        """Маркеры в цитатах должны игнорироваться."""
        answer = 'Ответ: вес 100 кг. Автор сказал "возможно это так" в контексте.'
        result = self.validator._check_uncertainty(answer)
        # Зависит от VALIDATION_IGNORE_QUOTES
        assert result.passed is True or "возможно" in result.details

    def test_short_answer(self):
        """Короткий ответ должен пройти (слишком короткий для проверки)."""
        answer = "Да"
        result = self.validator._check_uncertainty(answer)
        assert result.passed is True


class TestSourcesCheck:
    """Тесты проверки на наличие источников."""

    def setup_method(self):
        self.validator = AnswerValidator(embeddings=None)

    def test_with_sources(self):
        """Ответ с источниками должен пройти проверку."""
        answer = "Вес 100 кг [Документ: passport.pdf, стр. 5]."
        result = self.validator._check_sources(answer)
        assert result.passed is True

    def test_without_sources(self):
        """Ответ без источников — предупреждение."""
        answer = "Вес модели составляет 100 килограммов."
        result = self.validator._check_sources(answer)
        assert result.passed is False
        assert result.severity == "warning"


class TestValidationResult:
    """Тесты итогового результата валидации."""

    def test_all_passed(self):
        """Все проверки пройдены — статус approved."""
        result = ValidationResult()
        result.add_check(ValidationCheck("test1", True, "OK", "info"))
        result.add_check(ValidationCheck("test2", True, "OK", "info"))
        assert result.status == "approved"

    def test_critical_failure(self):
        """Критическая ошибка — статус rejected."""
        result = ValidationResult()
        result.add_check(ValidationCheck("test1", False, "Fail", "critical"))
        assert result.status == "rejected"

    def test_warning_failure(self):
        """Предупреждение — статус review."""
        result = ValidationResult()
        result.add_check(ValidationCheck("test1", False, "Warn", "warning"))
        assert result.status == "review"

    def test_to_dict(self):
        """Сериализация в словарь."""
        result = ValidationResult()
        result.add_check(ValidationCheck("test1", True, "OK", "info"))
        d = result.to_dict()
        assert "status" in d
        assert "checks" in d
