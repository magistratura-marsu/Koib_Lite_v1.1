# -*- coding: utf-8 -*-
import time, logging
from pathlib import Path
from typing import List, Optional, Set

# ★ ПОДАВЛЕНИЕ СПАМА ОТ pymorphy2
logging.getLogger("pymorphy2").setLevel(logging.WARNING)
logging.getLogger("pymorphy2.opencorpora_dict").setLevel(logging.WARNING)

from config import DOCS_DIR, OUTPUT_DIR, ensure_dirs
from src.parsing import parse_pdf, parse_docx
from src.chunking import SmartChunker
from src.indexing import IndexBuilder

logger = logging.getLogger("koib.ingest")
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc"}

class BatchIngester:
    def __init__(self, docs_dir: Optional[Path] = None, output_dir: Optional[Path] = None, incremental: bool = True):
        self.docs_dir = docs_dir or DOCS_DIR
        self.output_dir = output_dir or OUTPUT_DIR
        self.incremental = incremental
        self.chunker = SmartChunker()
        self.index_builder = IndexBuilder(self.output_dir / "index")
        self._processed_files: Set[str] = set()
        if incremental: self._load_processed_files()

    def _load_processed_files(self):
        p = self.output_dir / "metadata" / "ingest_manifest.txt"
        if p.exists():
            with open(p, 'r', encoding='utf-8') as f: self._processed_files = set(l.strip() for l in f if l.strip())

    def _save_processed_files(self):
        p = self.output_dir / "metadata" / "ingest_manifest.txt"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, 'w', encoding='utf-8') as f:
            for fn in sorted(self._processed_files): f.write(f"{fn}\n")

    def _discover_files(self) -> List[Path]:
        if not self.docs_dir.exists(): return []
        all_files = []
        for ext in SUPPORTED_EXTENSIONS:
            all_files.extend(self.docs_dir.glob(f"**/*{ext}"))
            all_files.extend(self.docs_dir.glob(f"**/*{ext.upper()}"))
        if not self.incremental: return all_files
        return [f for f in all_files if f.name not in self._processed_files]

    def _process_file(self, file_path: Path) -> bool:
        ext = file_path.suffix.lower()
        try:
            elements = parse_pdf(file_path) if ext == ".pdf" else parse_docx(file_path) if ext in (".docx", ".doc") else None
            if not elements: return False
            
            chunks = self.chunker.chunk_elements(elements)
            if not chunks: return False
            
            # ★ ВИДИМЫЙ ПРОГРЕСС ИНДЕКСАЦИИ
            print(f"[{len(chunks)} чанков] ", end="", flush=True)
            self.index_builder.add_chunks(chunks)
            return True
        except Exception as e:
            logger.error(f"Ошибка {file_path.name}: {e}")
            return False

    def process_all(self) -> None:
        ensure_dirs()
        t0 = time.time()
        files = self._discover_files()
        if not files: return
        print(f"  Обнаружено файлов для индексации: {len(files)}")
        success_count = error_count = 0
        for i, fp in enumerate(files, 1):
            print(f"  [{i}/{len(files)}] {fp.name}...", end=" ", flush=True)
            if self._process_file(fp):
                success_count += 1; self._processed_files.add(fp.name); print("OK")
            else:
                error_count += 1; print("ОШИБКА")
        self._save_processed_files()
        self.index_builder.save()
        print(f"\nРезультат: {success_count} успешно, {error_count} ошибок")
        print(f"  Индексы успешно развернуты на диске. Время сборки: {time.time() - t0:.1f}с")
