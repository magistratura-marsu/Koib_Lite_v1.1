# -*- coding: utf-8 -*-
"""
Koib-V-4.5 — Настройка логирования
=====================================
Централизованная конфигурация логирования для всех модулей.
Поддерживает вывод в консоль и ротируемый файл.
"""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

from config import LOGS_DIR, ensure_dirs


def setup_logging(
    level: int = logging.INFO,
    log_file: str = "koib.log",
    max_bytes: int = 5_000_000,
    backup_count: int = 3,
) -> logging.Logger:
    """
    Настроить логирование для проекта.

    Создаёт два обработчика:
      1. StreamHandler — вывод в консоль (stderr)
      2. RotatingFileHandler — ротируемый файл в LOGS_DIR

    Args:
        level: Уровень логирования (по умолчанию INFO)
        log_file: Имя файла лога
        max_bytes: Максимальный размер одного файла лога (5 МБ)
        backup_count: Количество ротированных файлов (3)

    Returns:
        Корневой логгер проекта 'koib'
    """
    ensure_dirs()

    root_logger = logging.getLogger("koib")
    root_logger.setLevel(level)

    # Формат лог-сообщений
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Обработчик 1: Консоль
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # Обработчик 2: Файл с ротацией
    file_path = LOGS_DIR / log_file
    file_handler = RotatingFileHandler(
        filename=str(file_path),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # Добавляем обработчики, если они ещё не были добавлены
    if not root_logger.handlers:
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)

    return root_logger


# Автоматическая настройка при импорте модуля
logger = setup_logging()
