# -*- coding: utf-8 -*-
"""
Koib-V-4.6 — CLI
★ ИСПРАВЛЕНО: cmd_evaluate формирует корректный context из контента чанков
"""
import sys
import time
import json
import argparse
import logging
import asyncio
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import DOCS_DIR, OUTPUT_DIR, FINAL_TOP_K, ensure_dirs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("koib.main")


def cmd_ingest(args) -> None:
    from batch_ingest import BatchIngester
    print("=" * 50)
    print("  KOIB-V-4.6 — ИНДЕКСАЦИЯ ДОКУМЕНТОВ")
    print("=" * 50)
    t0 = time.time()
    ingester = BatchIngester(
        docs_dir=Path(args.docs_dir),
        output_dir=Path(args.output_dir),
        incremental=args.incremental,
    )
    ingester.process_all()
    print(f"\nОбщее время: {time.time() - t0:.1f}с")


def cmd_query(args) -> None:
    from src.generation import AnswerGenerator
    print("=" * 50)
    print("  KOIB-V-4.6 — ЗАПРОС")
    print("=" * 50)
    generator = AnswerGenerator()
    t0 = time.time()
    result = generator.answer(
        query=args.query,
        k=args.top_k,
        model_filter=args.model,
    )
    print(f"\nЗапрос: {args.query}")
    print(f"\nОТВЕТ:\n{result['answer']}")
    if result.get("sources"):
        print("\nИсточники:")
        for src in result["sources"]:
            print(f"    - {src['document']}, стр. {src['page']}", end="")
            if src.get("heading"):
                print(f" — {src['heading']}", end="")
            print()
    print(f"\nВремя: {time.time() - t0:.2f}с")


def cmd_serve(args) -> None:
    import uvicorn
    print("=" * 50)
    print("  KOIB-V-4.6 — API СЕРВЕР")
    print("=" * 50)
    print(f"  Хост: {args.host}:{args.port}")
    print("=" * 50)
    uvicorn.run("api.app:app", host=args.host, port=args.port, log_level="info")


def cmd_evaluate(args) -> None:
    """
    ★ ИСПРАВЛЕНО: формирует context из реального контента чанков.
    Ранее передавались только имена файлов → метрики были невалидны.
    """
    from src.evaluation import RAGEvaluator, print_report
    from src.generation import AnswerGenerator
    print("=" * 50)
    print("  KOIB-V-4.6 — ОЦЕНКА КАЧЕСТВА")
    print("=" * 50)
    questions_path = Path(args.evaluate)
    if not questions_path.exists():
        print(f"  Файл не найден: {questions_path}")
        return

    with open(questions_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    print(f"  Загружено вопросов: {len(questions)}")

    generator = AnswerGenerator()
    evaluated_questions = []
    for i, q in enumerate(questions, 1):
        query = q.get("question", "")
        print(f"  [{i}/{len(questions)}] {query[:60]}...")
        result = generator.answer(query, k=args.top_k)

        # ★ ИСПРАВЛЕНО: context = реальный контент чанков
        context_parts = []
        for r in result.get("results", []):
            context_parts.append(r.to_context_string())
        q["answer"] = result["answer"]
        q["context"] = "\n\n".join(context_parts)
        q["context_chunks"] = len(result.get("results", []))
        evaluated_questions.append(q)

    evaluator = RAGEvaluator()
    output_path = Path(args.output_dir) / "evaluation_results.json"
    results = evaluator.evaluate_batch(evaluated_questions, save_path=output_path)

    if results:
        print_report(results)
        print(f"  Результаты сохранены: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Koib-V-4.6 — Production-ready RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ingest", action="store_true")
    parser.add_argument("--query", type=str, default="")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--evaluate", type=str, default="", metavar="FILE")
    parser.add_argument("--docs-dir", type=str, default=str(DOCS_DIR))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--top-k", type=int, default=FINAL_TOP_K)
    parser.add_argument("--incremental", action="store_true")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--vk-confirm-code", type=str, default="12345678")

    args = parser.parse_args()
    ensure_dirs()

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