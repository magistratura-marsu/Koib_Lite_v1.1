# -*- coding: utf-8 -*-
import sys, time, json, argparse, logging, asyncio, os
from pathlib import Path

# ★ ОТКЛЮЧАЕМ ПРОВЕРКУ ОБНОВЛЕНИЙ МОДЕЛЕЙ (Мгновенный старт без HEAD-запросов)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from config import DOCS_DIR, OUTPUT_DIR, FINAL_TOP_K, ensure_dirs

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")

def cmd_ingest(args):
    from batch_ingest import BatchIngester
    BatchIngester(Path(args.docs_dir), Path(args.output_dir), args.incremental).process_all()

def cmd_query(args):
    from src.rag_pipeline import RAGPipeline
    
    async def run():
        pipeline = RAGPipeline()
        t0 = time.time()
        result = await pipeline.answer(query=args.query, user_id="cli", k=args.top_k, use_memory=False, validate=True)
        
        print(f"\nОТВЕТ:\n{result['answer']}")
        
        if result.get("sources"):
            print("\nИсточники:")
            seen = set()
            # ★ Дедупликация: убираем повторы страниц
            for s in result["sources"]:
                key = f"{s['document']}_{s['page']}"
                if key not in seen:
                    seen.add(key)
                    print(f"  - {s['document']}, стр. {s['page']}")
                    
        print(f"\nВремя: {time.time() - t0:.2f}с")
        
        # ★ Корректное закрытие сетевых соединений (убирает красные ошибки)
        if hasattr(pipeline.llm, '_session') and pipeline.llm._session and not pipeline.llm._session.closed:
            await pipeline.llm._session.close()

    asyncio.run(run())

def cmd_serve(args):
    import uvicorn
    uvicorn.run("api.app:app", host=args.host, port=args.port, log_level="info")

def cmd_evaluate(args):
    print("Оценка качества запущена...")

def main():
    parser = argparse.ArgumentParser(description="Koib-V-4.8")
    parser.add_argument("--ingest", action="store_true")
    parser.add_argument("--query", type=str, default="")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--evaluate", type=str, default="")
    parser.add_argument("--docs-dir", type=str, default=str(DOCS_DIR))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--top-k", type=int, default=FINAL_TOP_K)
    parser.add_argument("--incremental", action="store_true")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    
    args = parser.parse_args()
    ensure_dirs()
    
    if args.ingest: cmd_ingest(args)
    elif args.query: cmd_query(args)
    elif args.serve: cmd_serve(args)
    elif args.evaluate: cmd_evaluate(args)
    else: parser.print_help()

if __name__ == "__main__":
    main()
