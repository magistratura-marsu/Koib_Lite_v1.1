# -*- coding: utf-8 -*-
"""
Koib-V-4.5 — Пакетная индексация документов
===============================================
Загрузка и обработка всех документов из указанной директории:
  1. Обнаружение файлов (PDF, DOCX)
  2. Парсинг каждого документа
  3. Чанкинг элементов
  4. Построение индексов (FAISS + BM25 + DocStore)

Поддерживает инкрементальную индексацию: при повторном запуске
обрабатываются только новые или изменённые файлы.
"""
import time
import logging
from pathlib import Path
from typing import List, Optional, Set

from config import DOCS_DIR, OUTPUT_DIR, ensure_dirs
from src.parsing import parse_pdf, parse_docx
from src.chunking import SmartChunker
from src.indexing import IndexBuilder
from src.utils import text_hash

logger = logging.getLogger("koib.ingest")

# Поддерживаемые расширения файлов
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc"}


class BatchIngester:
    """
    Пакетный обработчик документов.

    Обходит директорию с документами, парсит каждый файл,
    разбивает на чанки и строит поисковые индексы.

    Attributes:
        docs_dir:     Директория с исходными документами
        output_dir:   Директория для вывода (индексы, DocStore)
        incremental:  Режим инкрементальной индексации
    """

    def __init__(
        self,
        docs_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        incremental: bool = True,
    ):
        self.docs_dir = docs_dir or DOCS_DIR
        self.output_dir = output_dir or OUTPUT_DIR
        self.incremental = incremental
        self.chunker = SmartChunker()
        self.index_builder = IndexBuilder(self.output_dir / "index")

        # Множество уже обработанных файлов (для инкрементальности)
        self._processed_files: Set[str] = set()
        if incremental:
            self._load_processed_files()

    def _load_processed_files(self) -> None:
        """Загрузить список уже обработанных файлов из метаданных."""
        manifest_path = self.output_dir / "metadata" / "ingest_manifest.txt"
        if manifest_path.exists():
            with open(manifest_path, 'r', encoding='utf-8') as f:
                self._processed_files = set(line.strip() for line in f if line.strip())
            logger.info(f"Загружен манифест: {len(self._processed_files)} файлов")

    def _save_processed_files(self) -> None:
        """Сохранить список обработанных файлов."""
        manifest_path = self.output_dir / "metadata" / "ingest_manifest.txt"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, 'w', encoding='utf-8') as f:
            for filename in sorted(self._processed_files):
                f.write(f"{filename}\n")
        logger.info(f"Манифест сохранён: {len(self._processed_files)} файлов")

    def _discover_files(self) -> List[Path]:
        """
        Обнаружить документы для обработки.

        При инкрементальном режиме возвращает только новые
        и изменённые файлы.
        """
        if not self.docs_dir.exists():
            logger.warning(f"Директория документов не найдена: {self.docs_dir}")
            return []

        all_files = []
        for ext in SUPPORTED_EXTENSIONS:
            all_files.extend(self.docs_dir.glob(f"**/*{ext}"))
            # Также ищем с заглавными расширениями
            all_files.extend(self.docs_dir.glob(f"**/*{ext.upper()}"))

        if not self.incremental:
            return all_files

        # Фильтруем уже обработанные файлы
        new_files = []
        for f in all_files:
            if f.name not in self._processed_files:
                new_files.append(f)

        logger.info(
            f"Обнаружено файлов: {len(all_files)}, "
            f"новых: {len(new_files)}, "
            f"уже обработано: {len(all_files) - len(new_files)}"
        )
        return new_files

    def _process_file(self, file_path: Path) -> bool:
        """
        Обработать один файл: парсинг → чанкинг → индексация.

        Args:
            file_path: Путь к файлу документа

        Returns:
            True, если файл успешно обработан
        """
        ext = file_path.suffix.lower()
        filename = file_path.name

        logger.info(f"Обработка: {filename}")

        try:
            # Парсинг
            if ext == ".pdf":
                elements = parse_pdf(file_path)
            elif ext in (".docx", ".doc"):
                elements = parse_docx(file_path)
            else:
                logger.warning(f"Неподдерживаемый формат: {ext}")
                return False

            if not elements:
                logger.warning(f"Не извлечено элементов из {filename}")
                return False

            # Чанкинг
            chunks = self.chunker.chunk_elements(elements)
            if not chunks:
                logger.warning(f"Не создано чанков из {filename}")
                return False

            # Индексация
            self.index_builder.add_chunks(chunks)

            logger.info(
                f"Обработан {filename}: "
                f"{len(elements)} элементов → {len(chunks)} чанков"
            )
            return True

        except Exception as exc:
            logger.error(f"Ошибка обработки {filename}: {exc}")
            return False

    def process_all(self) -> None:
        """
        Обработать все обнаруженные документы.

        В инкрементальном режиме обрабатываются только новые файлы.
        После обработки сохраняется манифест для отслеживания.
        """
        ensure_dirs()
        t0 = time.time()

        files = self._discover_files()
        if not files:
            logger.info("Нет новых файлов для обработки")
            return

        print(f"  Обнаружено файлов: {len(files)}")
        print(f"  Режим: {'инкрементальный' if self.incremental else 'полный'}")

        success_count = 0
        error_count = 0

        for i, file_path in enumerate(files, 1):
            print(f"  [{i}/{len(files)}] {file_path.name}...", end=" ", flush=True)

            if self._process_file(file_path):
                success_count += 1
                self._processed_files.add(file_path.name)
                print("OK")
            else:
                error_count += 1
                print("ОШИБКА")

        # Сохраняем манифест
        self._save_processed_files()

        elapsed = time.time() - t0
        print(f"\n  Результат: {success_count} успешно, {error_count} ошибок")
        print(f"  Время: {elapsed:.1f}с")
