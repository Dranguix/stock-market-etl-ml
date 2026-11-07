#!/usr/bin/env python3
"""
Backfill histórico de noticias vía GDELT → panel point-in-time.

Genera data/news/history/daily_features.parquet para walk-forward A/B.

Uso:
    python scripts/build_news_history.py
    python scripts/build_news_history.py --start 2018-01-01 --end 2026-06-18
    python scripts/build_news_history.py --tickers WALMEX,ALSEA,GMEXICOB  # prueba rápida
    python scripts/build_news_history.py --force
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

try:
    import fcntl
except ImportError:  # Windows: sin flock no hay exclusión mutua entre procesos
    fcntl = None

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from scripts.backtest_topk import load_universe_tickers
from src.ml.news_history import build_history_panel, clean_bad_chunk_caches, history_stats
from src.ml.news_sources import configure_gdelt_interval

BUILD_LOCK = ROOT / "data" / "news" / "history" / ".build.lock"


def _acquire_build_lock() -> object:
    BUILD_LOCK.parent.mkdir(parents=True, exist_ok=True)
    lock_file = open(BUILD_LOCK, "w")
    if fcntl is None:
        print("⚠️  Sin flock en este sistema: nada impide correr dos builds a la vez.")
    else:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(
                "\n❌ Otro build_news_history ya está corriendo.\n"
                "   Espera a que termine o elimina data/news/history/.build.lock\n"
            )
            sys.exit(1)
    lock_file.write(str(os.getpid()))
    lock_file.flush()
    return lock_file


def main() -> None:
    parser = argparse.ArgumentParser(description="GDELT → panel histórico point-in-time")
    parser.add_argument("--universe", default="bmv")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--tickers", help="CSV de tickers (default: universo completo)")
    parser.add_argument("--chunk-days", type=int, default=180,
                        help="Días por chunk GDELT (180 = menos requests)")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--gdelt-interval", type=float, default=60.0,
        help="Segundos mínimos entre requests GDELT (default 60)",
    )
    parser.add_argument(
        "--max-tickers", type=int, default=0,
        help="Limitar a N tickers (0 = todos)",
    )
    parser.add_argument(
        "--clean-bad-cache", action="store_true",
        help="Elimina chunks vacíos cacheados por 429",
    )
    parser.add_argument(
        "--cache-only", action="store_true",
        help="Solo usar chunks en disco (sin llamar GDELT)",
    )
    args = parser.parse_args()

    lock = _acquire_build_lock()
    try:
        if args.clean_bad_cache or args.force:
            n = clean_bad_chunk_caches()
            if n:
                print(f"  Limpiados {n} chunk caches inválidos")

        all_tickers = load_universe_tickers(args.universe)
        if args.tickers:
            names = [t.strip() for t in args.tickers.split(",") if t.strip()]
            tickers = {k: v for k, v in all_tickers.items() if k in names}
        else:
            tickers = all_tickers
        if args.max_tickers > 0:
            tickers = dict(list(tickers.items())[: args.max_tickers])

        configure_gdelt_interval(args.gdelt_interval)
        n_chunks = max(1, ((
            datetime.strptime(args.end, "%Y-%m-%d")
            - datetime.strptime(args.start, "%Y-%m-%d")
        ).days // args.chunk_days) + 1)
        est_min = len(tickers) * n_chunks * args.gdelt_interval / 60

        print(f"\n══ News history GDELT — {len(tickers)} tickers ══")
        print(f"   Rango: {args.start} → {args.end}")
        print(f"   Intervalo GDELT: {args.gdelt_interval:.0f}s (~{est_min:.0f} min estimados)")
        print("   Un solo proceso; 429 → cooldown 3 min automático\n")

        build_history_panel(
            tickers,
            args.start,
            args.end,
            chunk_days=args.chunk_days,
            force=args.force,
            cache_only=args.cache_only,
            verbose=True,
        )

        stats = history_stats()
        print(f"\n✅ Stats: {json.dumps(stats, indent=2)}\n")
    finally:
        if fcntl is not None:
            fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
        BUILD_LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
