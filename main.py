# -*- coding: utf-8 -*-
"""
Koib-V-4.5 — Главная точка входа
===================================
CLI-интерфейс для управления RAG-системой:
  --ingest    — индексация документов
  --query     — одиночный запрос
  --serve     — запуск FastAPI сервера (VK-бот и API)
  --evaluate  — оценка качества RAG

Примеры:
  python main.py --ingest
  python main.py --query "Какие параметры у модели АИИС-001?"
  python main.py --serve
  python main.py --evaluate questions.json
"""
import sys
import time
import json
import argparse
import logging
import asyncio
from pathlib import Path
from typing import Dict

# Добавляем корень проекта в PYTHONPATH
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import DOCS_DIR, OUTPUT_DIR, FINAL_TOP_K, ensure_dirs
from src.utils import clean_text

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("koib.main")


def cmd_ingest(args) -> None:
    """
    Команда индексации документов.

    Загружает документы из указанной директории, парсит,
    разбивает на чанки и строит поисковые индексы.
    """
    from batch_ingest import BatchIngester

    print("=" * 50)
    print("  KOIB-V-4.5 — ИНДЕКСАЦИЯ ДОКУМЕНТОВ")
    print("=" * 50)

    t0 = time.time()
    ingester = BatchIngester(
        docs_dir=Path(args.docs_dir),
        output_dir=Path(args.output_dir),
        incremental=args.incremental,
    )
    ingester.process_all()
    print(f"\n  Общее время: {time.time() - t0:.1f}с")


def cmd_query(args) -> None:
    """
    Команда одиночного запроса.

    Выполняет поиск по индексам, генерирует ответ
    и выводит результат в консоль.
    """
    from src.generation import AnswerGenerator

    print("=" * 50)
    print("  KOIB-V-4.5 — ЗАПРОС")
    print("=" * 50)

    generator = AnswerGenerator()
    t0 = time.time()
    result = generator.answer(
        query=args.query,
        k=args.top_k,
        model_filter=args.model,
    )

    print(f"\n  Запрос: {args.query}")
    print(f"\n  ОТВЕТ:\n  {result['answer']}")

    # Источники
    if result.get("sources"):
        print(f"\n  Источники:")
        for src in result["sources"]:
            print(f"    - {src['source']}, стр. {src['page']}", end="")
            if src.get("heading"):
                print(f" — {src['heading']}", end="")
            print()

    # Валидация
    validation = result.get("validation")
    if validation:
        print(f"\n  Валидация: {validation['status']}")
        for check in validation.get("checks", []):
            status = "+" if check["passed"] else "-"
            print(f"    [{status}] {check['name']}: {check['details']}")

    print(f"\n  Время: {time.time() - t0:.2f}с")


def cmd_serve(args) -> None:
    """
    Команда запуска FastAPI сервера.

    Поднимает асинхронный HTTP-сервер на порту 8000,
    готовый принимать Callback-события от VK.
    """
    import uvicorn

    print("=" * 50)
    print("  KOIB-V-4.5 — API СЕРВЕР")
    print("=" * 50)
    print(f"  Хост: {args.host}")
    print(f"  Порт: {args.port}")
    print(f"  VK Confirm Code: {args.vk_confirm_code}")
    print("=" * 50)

    logger.info(f"Запуск FastAPI сервера на {args.host}:{args.port}...")
    uvicorn.run(
        "api.app:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )


def cmd_evaluate(args) -> None:
    """
    Команда оценки качества RAG.

    Загружает набор вопросов из JSON-файла,
    генерирует ответы и оценивает их через LLM-as-Judge.
    """
    from src.evaluation import RAGEvaluator
    from src.generation import AnswerGenerator

    print("=" * 50)
    print("  KOIB-V-4.5 — ОЦЕНКА КАЧЕСТВА")
    print("=" * 50)

    questions_path = Path(args.questions_file)
    if not questions_path.exists():
        print(f"  Файл не найден: {questions_path}")
        return

    # Загрузка вопросов
    with open(questions_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    print(f"  Загружено вопросов: {len(questions)}")

    # Генерация ответов
    generator = AnswerGenerator()
    evaluated_questions = []

    for i, q in enumerate(questions, 1):
        query = q.get("question", "")
        print(f"  [{i}/{len(questions)}] {query[:60]}...")

        result = generator.answer(query, k=args.top_k)
        q["answer"] = result["answer"]
        q["context"] = "\n".join(
            src.get("source", "") for src in result.get("sources", [])
        )
        evaluated_questions.append(q)

    # Оценка
    evaluator = RAGEvaluator()
    output_path = Path(args.output_dir) / "evaluation_results.json"

    results = evaluator.evaluate_batch(
        evaluated_questions,
        save_path=output_path,
    )

    # Сводка
    if results:
        avg_rag = sum(r.rag_score for r in results) / len(results)
        print(f"\n  Средний RAG-score: {avg_rag:.3f}")
        print(f"  Результаты сохранены: {output_path}")


def main():
    """Главная функция CLI."""
    parser = argparse.ArgumentParser(
        description="Koib-V-4.5 — Оптимизированная RAG-система",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python main.py --ingest
  python main.py --query "Параметры модели АИИС-001"
  python main.py --serve --port 8000
  python main.py --evaluate questions.json
        """,
    )

    # Команды
    parser.add_argument("--ingest", action="store_true", help="Индексация документов")
    parser.add_argument("--query", type=str, default="", help="Одиночный запрос")
    parser.add_argument("--serve", action="store_true", help="Запуск API сервера (FastAPI)")
    parser.add_argument("--evaluate", type=str, default="", metavar="FILE",
                        help="Оценка качества (путь к JSON с вопросами)")

    # Параметры
    parser.add_argument("--docs-dir", type=str, default=str(DOCS_DIR),
                        help="Директория с документами")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="Директория для вывода")
    parser.add_argument("--model", type=str, default="",
                        help="Фильтр по модели устройства")
    parser.add_argument("--top-k", type=int, default=FINAL_TOP_K,
                        help="Количество результатов поиска")
    parser.add_argument("--incremental", action="store_true",
                        help="Инкрементальная индексация")

    # Параметры сервера
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Хост API сервера")
    parser.add_argument("--port", type=int, default=8000,
                        help="Порт API сервера")
    parser.add_argument("--vk-confirm-code", type=str, default="12345678",
                        help="Код подтверждения VK Callback")

    args = parser.parse_args()
    ensure_dirs()

    # Выполнение команды
    if args.ingest:
        cmd_ingest(args)
    elif args.query:
        cmd_query(args)
    elif args.serve:
        cmd_serve(args)
    elif args.evaluate:
        cmd_evaluate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
