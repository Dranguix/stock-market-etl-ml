"""
Orquestador único de ingestas de datos — stock-market.

Ejecuta en secuencia todas las fuentes que alimentan SuperModel:
  1. OHLCV parquets     (yfinance → data/market/)
  2. Noticias          (todos los universos → data/news/)
  3. Intradía 5m        (DataBursátil → data/intraday/)
  4. Fundamentales      (DataBursátil → data/fundamentals.parquet)
  5. SQL ETL incremental (opcional → MercadoBursatil)
  6. Auditoría          (scripts/audit_data.py → data/ingest/last_run.json)

Uso:
    python scripts/run_all_ingests.py --all
    python scripts/run_all_ingests.py --all --skip-sql
    python scripts/run_all_ingests.py --ohlcv --news
    python scripts/run_all_ingests.py --all --groups bmv macro

Cron (Mac/Linux — ver scripts/install_ingest_cron.sh):
    0 13 * * 1-5  cd /path/stock-market && ./venv/bin/python scripts/run_all_ingests.py --all
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

INGEST_DIR = ROOT / "data" / "ingest"
REPORT_PATH = INGEST_DIR / "last_run.json"
FUNDAMENTALS_PATH = ROOT / "data" / "fundamentals.parquet"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_OHLCV_GROUPS = ("bmv", "usa", "etf", "crypto", "fibra", "macro")


def _step(name: str, fn, report: dict) -> bool:
    """Ejecuta un paso, registra duración y errores."""
    t0 = time.perf_counter()
    logger.info("── %s ──", name)
    try:
        result = fn()
        elapsed = round(time.perf_counter() - t0, 1)
        report["steps"][name] = {"status": "ok", "elapsed_s": elapsed, **(result or {})}
        logger.info("✓ %s (%.1fs)", name, elapsed)
        return True
    except Exception as exc:
        elapsed = round(time.perf_counter() - t0, 1)
        report["steps"][name] = {
            "status": "error",
            "elapsed_s": elapsed,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        logger.error("✗ %s: %s", name, exc)
        return False


def ingest_ohlcv(groups: tuple[str, ...]) -> dict[str, Any]:
    from src.data.store import update_group

    stats: dict[str, int] = {}
    for g in groups:
        result = update_group(g, verbose=True)
        stats[g] = len(result)
    return {"groups": list(groups), "tickers_updated": stats}


def ingest_news(
    days_back: int = 7,
    universe: str | None = None,
    gdelt: bool = False,
) -> dict[str, Any]:
    from scripts.fetch_news import run_fetch_news

    return run_fetch_news(
        days_back=days_back,
        universe=universe,
        gdelt=gdelt,
        save_cache=True,
        verbose=True,
    )


def ingest_intraday(full: bool = False) -> dict[str, Any]:
    from scripts.refresh_market_data import refresh_intraday

    refresh_intraday(full=full)
    return {"full": full}


def ingest_fundamentals(max_age_days: int = 7, force: bool = False) -> dict[str, Any]:
    if (
        not force
        and FUNDAMENTALS_PATH.exists()
        and max_age_days > 0
    ):
        age = (time.time() - FUNDAMENTALS_PATH.stat().st_mtime) / 86400
        if age < max_age_days:
            logger.info(
                "Fundamentales recientes (%.1f días) — omitido (usa --force-fundamentals)",
                age,
            )
            return {"skipped": True, "age_days": round(age, 1)}

    import subprocess

    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "fetch_fundamentals.py")],
        cwd=str(ROOT),
    )
    if r.returncode != 0:
        raise RuntimeError(f"fetch_fundamentals exit {r.returncode}")

    rows = 0
    if FUNDAMENTALS_PATH.exists():
        import pandas as pd
        rows = len(pd.read_parquet(FUNDAMENTALS_PATH))
    return {"rows": rows, "path": str(FUNDAMENTALS_PATH)}


def ingest_sql_etl() -> dict[str, Any]:
    from src.utils.config import Config

    if not Config.DATABURSATIL_API_TOKEN:
        return {"skipped": True, "reason": "sin DATABURSATIL_API_TOKEN"}
    if not Config.SQL_SERVER_PASSWORD:
        return {"skipped": True, "reason": "sin SQL_PASSWORD"}

    from src.etl.ejecutar_etl_incremental_databursatil import (
        ejecutar_etl_incremental_databursatil,
        load_config,
    )

    config = load_config()
    if not config:
        raise RuntimeError("no se pudo cargar config/tickers_config.yml")
    ok = ejecutar_etl_incremental_databursatil(config)
    if not ok:
        raise RuntimeError("ETL incremental SQL falló (ver logs)")
    return {"ok": True}


def ingest_audit() -> dict[str, Any]:
    from scripts.audit_data import run_audit

    report = run_audit()
    audit_path = INGEST_DIR / "audit.json"
    INGEST_DIR.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return {
        "audit_path": str(audit_path),
        "news_pct": report.get("news", {}).get("pct_coverage"),
        "bmv_intraday": report.get("intraday", {}).get("bmv_covered"),
    }


def ingest_tasas_hoy() -> dict[str, Any]:
    """Guarda snapshot de tasas de mercado de dinero (TIIE, CETE, Tasa_Objetivo)."""
    from src.ml.rates_features import fetch_tasas_snapshot
    snap = fetch_tasas_snapshot(sleep=0)
    if snap:
        return {"ok": True, "n_tasas": len(snap), "tiie28": snap.get("tiie28"), "tasa_obj": snap.get("tasa_obj")}
    return {"ok": False}


def ingest_hechos_hoy() -> dict[str, Any]:
    """Descarga los hechos (order flow) del día hábil anterior."""
    import subprocess
    from datetime import date, timedelta
    ayer = date.today() - timedelta(days=1)
    fecha = ayer.strftime("%Y-%m-%d")
    r = subprocess.run(
        [str(ROOT / "venv" / "bin" / "python"), "scripts/backfill_hechos.py",
         "--fecha", fecha, "--sleep", "0"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    return {"fecha": fecha, "ok": r.returncode == 0, "output": r.stdout.strip()[-200:]}


def ingest_record_signals() -> dict[str, Any]:
    """Corre SuperModel (--fast, sin Gemini) y guarda señales timestamped."""
    import subprocess
    r = subprocess.run(
        [str(ROOT / "venv" / "bin" / "python"), "scripts/record_signals.py", "--no-gemini"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    # Extraer n_bets del output
    n_bets = 0
    for line in r.stdout.splitlines():
        if "Apuestas:" in line:
            try:
                n_bets = int(line.split(":")[-1].strip())
            except ValueError:
                pass
    return {"ok": r.returncode == 0, "n_bets": n_bets, "output": r.stdout.strip()[-300:]}


def run_ingests(
    *,
    ohlcv: bool = False,
    news: bool = False,
    intraday: bool = False,
    fundamentals: bool = False,
    sql: bool = False,
    audit: bool = True,
    hechos: bool = True,
    record_signals: bool = True,
    all_steps: bool = False,
    groups: tuple[str, ...] = DEFAULT_OHLCV_GROUPS,
    news_days: int = 7,
    news_universe: str | None = None,
    news_gdelt: bool = False,
    intraday_full: bool = False,
    force_fundamentals: bool = False,
    fundamentals_max_age: int = 7,
) -> dict[str, Any]:
    if all_steps:
        ohlcv = news = intraday = fundamentals = True
        # SQL solo si no se pidió explícitamente omitir vía caller

    report: dict[str, Any] = {
        "started_at": datetime.now(tz=timezone.utc).isoformat(),
        "steps": {},
        "success": True,
    }

    steps: list[tuple[str, Any]] = []
    if ohlcv:
        steps.append(("ohlcv", lambda: ingest_ohlcv(groups)))
    if news:
        steps.append((
            "news",
            lambda: ingest_news(news_days, universe=news_universe, gdelt=news_gdelt),
        ))
    if intraday:
        steps.append(("intraday", lambda: ingest_intraday(intraday_full)))
    if fundamentals:
        steps.append((
            "fundamentals",
            lambda: ingest_fundamentals(
                max_age_days=fundamentals_max_age,
                force=force_fundamentals,
            ),
        ))
    if sql:
        steps.append(("sql_etl", ingest_sql_etl))
    steps.append(("tasas_hoy", ingest_tasas_hoy))
    if hechos:
        steps.append(("hechos_hoy", ingest_hechos_hoy))
    if record_signals:
        steps.append(("record_signals", ingest_record_signals))
    if audit:
        steps.append(("audit", ingest_audit))

    if not steps:
        raise ValueError("Ningún paso seleccionado — usa --all o flags individuales")

    logger.info("═" * 55)
    logger.info("INGEST PIPELINE — %s", datetime.now().strftime("%Y-%m-%d %H:%M"))
    logger.info("Pasos: %s", ", ".join(s for s, _ in steps))
    logger.info("═" * 55)

    critical = {"ohlcv", "news", "intraday"}
    for name, fn in steps:
        ok = _step(name, fn, report)
        if not ok and name in critical:
            report["success"] = False

    report["finished_at"] = datetime.now(tz=timezone.utc).isoformat()
    report["elapsed_s"] = round(
        sum(s.get("elapsed_s", 0) for s in report["steps"].values()), 1
    )

    INGEST_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    logger.info("Reporte: %s", REPORT_PATH)

    status = "OK" if report["success"] else "CON ERRORES"
    logger.info("═" * 55)
    logger.info("INGEST %s — %.1fs total", status, report["elapsed_s"])
    logger.info("═" * 55)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Orquestador de ingestas de datos (OHLCV, news, intradía, fundamentals, SQL)",
    )
    parser.add_argument("--all", action="store_true", help="OHLCV + news + intraday + fundamentals + audit")
    parser.add_argument("--ohlcv", action="store_true", help="Actualizar parquets yfinance")
    parser.add_argument("--news", action="store_true", help="Refrescar noticias (todos los universos)")
    parser.add_argument("--news-bmv-only", action="store_true", help="Noticias solo universo BMV")
    parser.add_argument("--news-gdelt", action="store_true", help="Incluir GDELT en noticias (lento)")
    parser.add_argument("--intraday", action="store_true", help="Refrescar barras 5m")
    parser.add_argument("--fundamentals", action="store_true", help="Refrescar fundamentales")
    parser.add_argument("--sql", action="store_true", help="ETL incremental → SQL Server")
    parser.add_argument("--skip-sql", action="store_true", help="Con --all: no ejecutar SQL ETL")
    parser.add_argument("--skip-audit", action="store_true", help="No correr auditoría al final")
    parser.add_argument("--skip-hechos", action="store_true", help="No descargar hechos del día")
    parser.add_argument("--skip-signals", action="store_true", help="No registrar señales del día")
    parser.add_argument(
        "--groups", nargs="+", default=list(DEFAULT_OHLCV_GROUPS),
        choices=list(DEFAULT_OHLCV_GROUPS),
        help="Grupos OHLCV a actualizar (default: todos)",
    )
    parser.add_argument("--news-days", type=int, default=7, help="Ventana noticias (días)")
    parser.add_argument("--intraday-full", action="store_true", help="Re-descargar todo el intradía")
    parser.add_argument("--force-fundamentals", action="store_true", help="Forzar fundamentals aunque sean recientes")
    args = parser.parse_args()

    if not any([args.all, args.ohlcv, args.news, args.intraday, args.fundamentals, args.sql]):
        parser.error("Indica --all o al menos un paso (--ohlcv, --news, ...)")

    report = run_ingests(
        ohlcv=args.all or args.ohlcv,
        news=args.all or args.news,
        intraday=args.all or args.intraday,
        fundamentals=args.all or args.fundamentals,
        sql=args.sql or (args.all and not args.skip_sql),
        audit=not args.skip_audit,
        hechos=not args.skip_hechos,
        record_signals=not args.skip_signals,
        groups=tuple(args.groups),
        news_days=args.news_days,
        news_universe="bmv" if args.news_bmv_only else None,
        news_gdelt=args.news_gdelt,
        intraday_full=args.intraday_full,
        force_fundamentals=args.force_fundamentals,
    )

    sys.exit(0 if report["success"] else 1)


if __name__ == "__main__":
    main()
