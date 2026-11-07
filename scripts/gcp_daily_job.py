#!/usr/bin/env python3
"""Job diario de Cloud Run: ingest, decide y score.

Zona horaria: America/Mexico_City.
Base: Cloud SQL Postgres, proyecto domesticospro, instancia
domesticospro:us-central1:domesticos-pg, database mercado, esquema lab.

Cadena:
  ingest (lun-vie 18:30, y sáb-dom solo noticias)
    escribe lab.price_daily y lab.news_daily.
  decide (lun-vie 19:15)
    arma el plan de entrada del siguiente día hábil.
  score (lun-vie 20:00)
    califica los planes cuyo enter_on + horizonte ya cumplió.

Escritura (src/db/lab_store.py, esquema de sql/03_lab_schema.sql):
  upsert_prices → lab.price_daily
      ticker, trade_date, open, high, low, close, volume, moneda, source
  insert_news → lab.news_daily
      ticker, trade_date, title, source, sentiment, url
  insert_decision(session_date, model, cash_usd, motivo, raw, plans)
      lab.decision + lab.entry_plan
      (ticker, usd, horizon texto "20d"/"4s", stop_usd, p_sube, enter_on)
  score lee lab.entry_plan sin lab.outcome y escribe
      lab.outcome (entry_plan_id, realized_return, hit_stop, note).
      Si lab_store no trae esas dos funciones, este job usa lab_store.connect().

Conexión: LAB_DATABASE_URL, o LAB_PGHOST + LAB_PGUSER. DATABASE_URL se copia
a LAB_DATABASE_URL si la primera no está. Sin módulo o sin URL se imprime
el plan o el scorecard y, en local, queda copia en data/lab/.
El horizonte es en días hábiles lun-vie, sin calendario de asuetos BMV.
"20d" son 20 días hábiles; "1s".."4s" son 5, 10, 15 y 20.
Si el stop se toca, realized_return sale del stop; si no, del close final.
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MX = ZoneInfo("America/Mexico_City")
MARKET = ROOT / "data" / "market"
LAB_DIR = ROOT / "data" / "lab"
PLANS_PATH = LAB_DIR / "entry_plans.jsonl"
OUTCOMES_PATH = LAB_DIR / "outcomes.jsonl"
BOOK_PATH = ROOT / "data" / "agent" / "book.json"
SIGNALS = ROOT / "data" / "signals"

HORIZON_BDAYS = {"1s": 5, "2s": 10, "3s": 15, "4s": 20}
DEFAULT_HORIZON = "4s"
P_SUBE_MIN = 0.50
TICKET_USD = 15.0
XAI_URL = "https://api.x.ai/v1/chat/completions"

logger = logging.getLogger("gcp_daily_job")

_SECRET_ENV = (
    "DATABASE_URL",
    "LAB_DATABASE_URL",
    "LAB_PGPASSWORD",
    "LAB_PASSWORD",
    "XAI_API_KEY",
    "DB_PASSWORD",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "DATABURSATIL_API_TOKEN",
    "ALPHA_VANTAGE_API_KEY",
)


def _redact(text: str) -> str:
    out = text
    for name in _SECRET_ENV:
        val = os.environ.get(name)
        if val and len(val) > 4:
            out = out.replace(val, "***")
    return out


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / ".env")


def _today_mx() -> date:
    return datetime.now(MX).date()


def _is_weekend(day: date) -> bool:
    return day.weekday() >= 5


def _next_bday(day: date) -> date:
    nxt = day + timedelta(days=1)
    while _is_weekend(nxt):
        nxt += timedelta(days=1)
    return nxt


def _add_bdays(day: date, n: int) -> date:
    step = 1 if n >= 0 else -1
    left = abs(int(n))
    cursor = day
    while left > 0:
        cursor += timedelta(days=step)
        if not _is_weekend(cursor):
            left -= 1
    return cursor


def _as_date(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _json_default(obj):
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"no serializable: {type(obj).__name__}")


def _emit(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False, default=_json_default), flush=True)


def _num(value):
    if value is None or pd.isna(value):
        return None
    return float(value)


def _market_path(symbol: str) -> Path:
    safe = symbol.replace("/", "_").replace("^", "")
    return MARKET / f"{safe}.parquet"


def _universe_index() -> dict[str, dict]:
    from src.data.universes import UNIVERSES

    index: dict[str, dict] = {}
    for _label, cfg in UNIVERSES.items():
        for ticker, symbol in cfg["tickers"].items():
            if ticker in index:
                continue
            index[ticker] = {
                "ticker": ticker,
                "symbol": symbol,
                "slug": cfg["slug"],
                "moneda": cfg["moneda"],
                "horizon_days": int(cfg["horizonte"]),
            }
    return index


def _in_mxn(symbol: str, moneda: str) -> bool:
    if symbol.endswith(".MX"):
        return True
    return moneda == "MXN"


def _tradable(ticker: str) -> bool:
    try:
        from src.data.store import gbm_tradable
    except Exception:
        return True
    return bool(gbm_tradable(ticker))


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=_json_default))
        handle.write("\n")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


# ── lab_store ────────────────────────────────────────────────────────────────


def _import_lab_store():
    """Carga src/db/lab_store.py si ya existe. None si todavía no está."""
    path = ROOT / "src" / "db" / "lab_store.py"
    if not path.exists():
        logger.info("src/db/lab_store.py todavía no existe")
        return None
    try:
        return importlib.import_module("src.db.lab_store")
    except Exception as exc:
        logger.info(
            "import de src.db.lab_store no entró (%s); cargo el archivo directo",
            type(exc).__name__,
        )
        sys.modules.pop("src.db.lab_store", None)
    import importlib.util

    spec = importlib.util.spec_from_file_location("src.db.lab_store", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules["src.db.lab_store"] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        logger.warning("lab_store no cargó: %s", _redact(str(exc)))
        return None
    return module


def _align_db_env() -> None:
    """lab_store lee LAB_DATABASE_URL. Acepta DATABASE_URL como alias."""
    if not os.environ.get("LAB_DATABASE_URL") and os.environ.get("DATABASE_URL"):
        os.environ["LAB_DATABASE_URL"] = os.environ["DATABASE_URL"]


def _db_ready(store) -> bool:
    if store is None:
        return False
    url_fn = getattr(store, "database_url", None)
    if callable(url_fn):
        try:
            if url_fn():
                return True
        except Exception as exc:
            logger.error("database_url falló: %s", _redact(str(exc)))
            return False
    if os.environ.get("LAB_DATABASE_URL") or os.environ.get("DATABASE_URL"):
        return True
    host = os.environ.get("LAB_PGHOST") or os.environ.get("LAB_HOST")
    user = os.environ.get("LAB_PGUSER") or os.environ.get("LAB_USER")
    return bool(host and user)


def _horizon_label(days: int) -> str:
    return f"{int(days)}d"


def _parse_horizon(value) -> int:
    if value is None or value == "":
        return 20
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    text = str(value).strip().lower()
    if text in HORIZON_BDAYS:
        return HORIZON_BDAYS[text]
    if text.endswith("d") and text[:-1].isdigit():
        return int(text[:-1])
    if text.isdigit():
        return int(text)
    return 20


def _write_prices(rows: list[dict]) -> int | None:
    """lab_store.upsert_prices → lab.price_daily.

    Campos: ticker, trade_date, open, high, low, close, volume, moneda, source.
    """
    store = _import_lab_store()
    if not _db_ready(store):
        logger.info("precios sin Postgres (%d filas)", len(rows))
        return None
    fn = getattr(store, "upsert_prices", None)
    if not callable(fn):
        raise RuntimeError("lab_store no define upsert_prices")
    return int(fn(rows))


def _write_news(rows: list[dict]) -> int | None:
    """lab_store.insert_news → lab.news_daily.

    Campos: ticker, trade_date, title, source, sentiment, url.
    """
    store = _import_lab_store()
    if not _db_ready(store):
        logger.info("noticias sin Postgres (%d filas)", len(rows))
        return None
    fn = getattr(store, "insert_news", None) or getattr(store, "upsert_news", None)
    if not callable(fn):
        raise RuntimeError("lab_store no define insert_news")
    return int(fn(rows))


def _plans_for_store(plans: list[dict]) -> list[dict]:
    stored = []
    for row in plans:
        days = int(row.get("horizon_days") or _parse_horizon(row.get("horizon")))
        stored.append({
            "ticker": row["ticker"],
            "usd": row["usd"],
            "horizon": row.get("horizon") if isinstance(row.get("horizon"), str) else _horizon_label(days),
            "stop_usd": row.get("stop"),
            "p_sube": row.get("p_sube"),
            "enter_on": row.get("enter_on"),
        })
    return stored


def _write_decision(decision: dict, plans: list[dict]) -> int | None:
    """lab_store.insert_decision escribe lab.decision y lab.entry_plan."""
    store = _import_lab_store()
    if not _db_ready(store):
        logger.info("decisión sin Postgres (%d nombres)", len(plans))
        return None
    insert_decision = getattr(store, "insert_decision", None)
    if not callable(insert_decision):
        raise RuntimeError("lab_store no define insert_decision")
    plan_rows = _plans_for_store(plans)
    return int(insert_decision(
        session_date=decision["decided_on"],
        model=decision.get("model") or "rules",
        cash_usd=decision["cash_usd"],
        motivo=decision["motivo"],
        raw={
            "llm_used": decision.get("llm_used"),
            "enter_on": decision.get("enter_on"),
            "book": decision.get("book"),
            "plans": plan_rows,
            **{k: decision[k] for k in ("regla", "gemelo", "vetos", "candidatos", "tendencia") if k in decision},
        },
        plans=plan_rows,
    ))


_OPEN_PLANS_SQL = """
SELECT p.id, p.ticker, p.usd, p.horizon, p.stop_usd, p.p_sube, p.enter_on
FROM lab.entry_plan p
WHERE NOT EXISTS (
    SELECT 1 FROM lab.outcome o WHERE o.entry_plan_id = p.id
)
"""

_OUTCOME_SQL = """
INSERT INTO lab.outcome (entry_plan_id, scored_at, realized_return, hit_stop, note)
VALUES (
    %(entry_plan_id)s,
    COALESCE(%(scored_at)s, now()),
    %(realized_return)s,
    %(hit_stop)s,
    %(note)s
)
"""


def _normalize_plan(row: dict) -> dict:
    stop = row.get("stop", row.get("stop_usd"))
    horizon_days = row.get("horizon_days")
    if horizon_days is None:
        horizon_days = _parse_horizon(row.get("horizon"))
    return {
        "id": row.get("id"),
        "local_id": row.get("local_id"),
        "ticker": row.get("ticker"),
        "symbol": row.get("symbol"),
        "usd": None if row.get("usd") is None else float(row["usd"]),
        "stop": None if stop is None else float(stop),
        "p_sube": None if row.get("p_sube") is None else float(row["p_sube"]),
        "enter_on": row.get("enter_on"),
        "horizon": row.get("horizon"),
        "horizon_days": int(horizon_days),
    }


def _sql_open_plans(store) -> list[dict]:
    with store.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(_OPEN_PLANS_SQL)
            cols = [desc[0] for desc in cur.description]
            fetched = [dict(zip(cols, rec)) for rec in cur.fetchall()]
    return [_normalize_plan(row) for row in fetched]


def _write_outcomes(rows: list[dict]) -> int | None:
    """Escribe lab.outcome. Usa insert_outcomes si existe; si no, connect()."""
    store = _import_lab_store()
    if not _db_ready(store):
        logger.info("outcomes sin Postgres (%d filas)", len(rows))
        return None
    fn = getattr(store, "insert_outcomes", None)
    if callable(fn):
        return int(fn(rows))
    payload = []
    for row in rows:
        if row.get("entry_plan_id") is None:
            continue
        payload.append({
            "entry_plan_id": int(row["entry_plan_id"]),
            "scored_at": None,
            "realized_return": row.get("return_pct"),
            "hit_stop": bool(row.get("hit_stop")),
            "note": row.get("note"),
        })
    if not payload:
        return 0
    with store.connect() as conn:
        with conn.cursor() as cur:
            cur.executemany(_OUTCOME_SQL, payload)
    return len(payload)


def _read_plans_to_score(as_of: date) -> tuple[list[dict], bool]:
    """Devuelve (planes, desde_db). El filtro de horizonte lo aplica el job."""
    store = _import_lab_store()
    if _db_ready(store):
        fn = getattr(store, "plans_ready_to_score", None)
        if callable(fn):
            try:
                raw = fn(as_of)
            except TypeError:
                raw = fn()
            return [_normalize_plan(row) for row in (raw or [])], True
        return _sql_open_plans(store), True
    return [_normalize_plan(row) for row in _read_jsonl(PLANS_PATH)], False


# ── precios ──────────────────────────────────────────────────────────────────


def _ohlc_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        level0 = list(frame.columns.get_level_values(0))
        level1 = list(frame.columns.get_level_values(-1))
        if "Close" in level0:
            frame.columns = level0
        elif "Close" in level1:
            frame.columns = level1
        else:
            frame.columns = level0
    return frame


def _bar_from_frame(df: pd.DataFrame, session: date) -> dict | None:
    if df is None or df.empty:
        return None
    frame = _ohlc_frame(df)
    idx = pd.to_datetime(frame.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    frame = frame.copy()
    frame.index = idx
    day = frame.loc[frame.index.normalize() == pd.Timestamp(session)]
    if day.empty:
        return None
    row = day.iloc[-1]
    close = _num(row.get("Close"))
    if close is None or close <= 0:
        return None
    return {
        "open": _num(row.get("Open")),
        "high": _num(row.get("High")),
        "low": _num(row.get("Low")),
        "close": close,
        "volume": _num(row.get("Volume")) or 0.0,
    }


def _parquet_bar(symbol: str, session: date) -> dict | None:
    path = _market_path(symbol)
    if not path.exists():
        return None
    try:
        return _bar_from_frame(pd.read_parquet(path), session)
    except Exception as exc:
        logger.info("parquet %s: %s", symbol, _redact(str(exc)))
        return None


def _yf_download(symbols: list[str], start: date, end: date) -> pd.DataFrame:
    import yfinance as yf

    tickers = symbols if len(symbols) > 1 else symbols[0]
    raw = yf.download(
        tickers=tickers,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=False,
        progress=False,
        threads=True,
        group_by="ticker",
    )
    return raw if isinstance(raw, pd.DataFrame) else pd.DataFrame()


def _slice_symbol(raw: pd.DataFrame, symbol: str, many: bool) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    if not many:
        return raw
    if not isinstance(raw.columns, pd.MultiIndex):
        return raw
    level0 = raw.columns.get_level_values(0)
    if symbol not in set(level0):
        return pd.DataFrame()
    return raw[symbol].dropna(how="all")


def _yf_bars(symbols: list[str], session: date) -> dict[str, dict]:
    if not symbols:
        return {}
    start = session - timedelta(days=7)
    found: dict[str, dict] = {}
    chunk = 40
    for offset in range(0, len(symbols), chunk):
        batch = symbols[offset : offset + chunk]
        try:
            raw = _yf_download(batch, start, session)
        except Exception as exc:
            logger.warning("yfinance falló (%d símbolos): %s", len(batch), _redact(str(exc)))
            continue
        many = len(batch) > 1
        for symbol in batch:
            bar = _bar_from_frame(_slice_symbol(raw, symbol, many), session)
            if bar:
                found[symbol] = bar
    return found


def _session_prices(session: date) -> list[dict]:
    index = _universe_index()
    wanted = [(meta["ticker"], meta["symbol"]) for meta in index.values()]
    wanted.append(("USDMXN", "MXN=X"))
    rows = []
    missing = []
    for ticker, symbol in wanted:
        bar = _parquet_bar(symbol, session)
        source = "parquet"
        if bar is None:
            missing.append((ticker, symbol))
            continue
        rows.append(_price_row(ticker, symbol, session, bar, source))
    fetched = _yf_bars([symbol for _ticker, symbol in missing], session)
    for ticker, symbol in missing:
        bar = fetched.get(symbol)
        if bar is None:
            logger.info("sin barra de %s para %s", session.isoformat(), symbol)
            continue
        rows.append(_price_row(ticker, symbol, session, bar, "yfinance"))
    return rows


def _moneda(symbol: str) -> str:
    if symbol.endswith(".MX") or symbol in {"MXN=X"}:
        return "MXN"
    return "USD"


def _price_row(ticker: str, symbol: str, session: date, bar: dict, source: str) -> dict:
    return {
        "ticker": ticker,
        "symbol": symbol,
        "trade_date": session,
        "open": bar["open"],
        "high": bar["high"],
        "low": bar["low"],
        "close": bar["close"],
        "volume": bar["volume"],
        "moneda": _moneda(symbol),
        "source": source,
    }


def _history(symbol: str, start: date, end: date) -> pd.DataFrame:
    path = _market_path(symbol)
    if path.exists():
        try:
            frame = pd.read_parquet(path)
            idx = pd.to_datetime(frame.index)
            if getattr(idx, "tz", None) is not None:
                idx = idx.tz_localize(None)
            frame = frame.copy()
            frame.index = idx
            sl = frame.loc[
                (frame.index.normalize() >= pd.Timestamp(start))
                & (frame.index.normalize() <= pd.Timestamp(end))
            ]
            if not sl.empty and "Close" in sl.columns:
                return sl
        except Exception as exc:
            logger.info("historial parquet %s: %s", symbol, _redact(str(exc)))
    try:
        raw = _yf_download([symbol], start, end)
    except Exception as exc:
        logger.info("historial yfinance %s: %s", symbol, _redact(str(exc)))
        return pd.DataFrame()
    frame = _slice_symbol(raw, symbol, many=False)
    if frame.empty:
        return frame
    idx = pd.to_datetime(frame.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    frame = frame.copy()
    frame.index = idx
    return frame.loc[
        (frame.index.normalize() >= pd.Timestamp(start))
        & (frame.index.normalize() <= pd.Timestamp(end))
    ]


# ── noticias ─────────────────────────────────────────────────────────────────


def _article_published(article) -> date | None:
    try:
        published = article.published_dt.astimezone(MX).date()
    except Exception:
        return None
    if published.year < 2000:
        return None
    return published


def _rows_from_articles(session: date) -> list[dict]:
    """Artículos del día para lab.news_daily (title, source, sentiment, url)."""
    try:
        from src.ml.news import _articles_in_window, _score_keywords, _scrape_articles
    except Exception as exc:
        logger.warning("src.ml.news no importó: %s", _redact(str(exc)))
        return []
    rows: list[dict] = []
    for ticker, meta in _universe_index().items():
        try:
            articles = _scrape_articles(ticker, meta["symbol"], days_back=2, use_gdelt=False)
        except Exception as exc:
            logger.warning("noticias %s: %s", ticker, _redact(str(exc)))
            continue
        try:
            window = _articles_in_window(articles, 2)
        except Exception:
            window = articles
        for article in window:
            published = _article_published(article)
            if published != session:
                continue
            sentiment = getattr(article, "av_sentiment", None)
            if sentiment is None:
                try:
                    sentiment = _score_keywords(article.text())
                except Exception:
                    sentiment = None
            title = (getattr(article, "title", "") or "").strip() or None
            rows.append({
                "ticker": ticker,
                "trade_date": session,
                "title": title,
                "source": getattr(article, "source", None),
                "sentiment": None if sentiment is None else float(sentiment),
                "url": getattr(article, "url", None) or None,
            })
    return rows


def _rows_from_feature_cache(session: date) -> list[dict]:
    """Si el scrape no dejó artículos, una fila por ticker desde el cache de features."""
    path = ROOT / "data" / "news" / f"{session.isoformat()}.json"
    if not path.exists():
        try:
            from scripts.fetch_news import run_fetch_news

            result = run_fetch_news(days_back=2, universe=None, gdelt=False, save_cache=True, verbose=False)
            dated = (result.get("paths") or {}).get("dated")
            if dated:
                path = Path(dated)
        except Exception as exc:
            logger.warning("scripts/fetch_news.py no corrió: %s", _redact(str(exc)))
            return []
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("records") or payload.get("rows") or []
    rows = []
    for rec in payload:
        ticker = rec.get("ticker")
        if not ticker:
            continue
        sources = rec.get("sources")
        if isinstance(sources, list):
            sources = ",".join(str(item) for item in sources)
        rows.append({
            "ticker": ticker,
            "trade_date": session,
            "title": None,
            "source": sources,
            "sentiment": _num(rec.get("news_sent_score")),
            "url": None,
        })
    return rows


def _fetch_news_rows(session: date) -> list[dict]:
    rows = _rows_from_articles(session)
    if rows:
        return rows
    logger.info("sin artículos del día; uso el cache de features si existe")
    return _rows_from_feature_cache(session)


def run_ingest(session: date | None = None) -> int:
    session = session or _today_mx()
    weekend = _is_weekend(session)
    logger.info("ingest %s weekend=%s", session.isoformat(), weekend)
    prices: list[dict] = []
    if not weekend:
        prices = _session_prices(session)
        written = _write_prices(prices)
        logger.info("precios %d (postgres=%s)", len(prices), written)
        if not prices:
            logger.error("día hábil sin precios")
            return 1
    news = _fetch_news_rows(session)
    written_news = _write_news(news)
    logger.info("noticias %d (postgres=%s)", len(news), written_news)
    if weekend and not news:
        logger.error("fin de semana sin noticias")
        return 1
    _emit({
        "mode": "ingest",
        "session": session.isoformat(),
        "weekend": weekend,
        "prices": len(prices),
        "news": len(news),
    })
    return 0


# ── decide ───────────────────────────────────────────────────────────────────


def _load_book():
    try:
        from src.agent.book import Book
    except Exception as exc:
        logger.warning("src.agent.book no disponible: %s", _redact(str(exc)))
        return None
    return Book(BOOK_PATH)


def _policy():
    from src.agent.book import BUDGET_USD
    from src.agent.policy import MAX_NAMES, MAX_TICKET, MIN_TICKET

    return BUDGET_USD, MAX_NAMES, MIN_TICKET, MAX_TICKET


def _ranking(slug: str, horizonte: str = DEFAULT_HORIZON) -> tuple[list[dict], int]:
    path = SIGNALS / f"{slug}.json"
    if not path.exists():
        return [], HORIZON_BDAYS[horizonte]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.info("señal %s: %s", slug, _redact(str(exc)))
        return [], HORIZON_BDAYS[horizonte]
    block = (payload.get("by_horizon") or {}).get(horizonte) or {}
    horizon = int(block.get("horizon_d") or HORIZON_BDAYS.get(horizonte, 20))
    top = list(block.get("top5") or payload.get("top5") or [])
    return top, horizon


def _p_sube(symbol: str) -> float | None:
    if not _market_path(symbol).exists():
        return None
    try:
        from src.agent.woe_obs import explain_ticker

        out = explain_ticker(symbol)
    except Exception as exc:
        logger.info("p_sube %s: %s", symbol, _redact(str(exc)))
        return None
    if not isinstance(out, dict) or not out.get("ok"):
        return None
    val = out.get("p_sube_5d")
    return None if val is None else round(float(val), 4)


def _fx_close(session: date) -> float | None:
    bar = _parquet_bar("MXN=X", session) or _yf_bars(["MXN=X"], session).get("MXN=X")
    if not bar:
        return None
    return bar["close"]


def _range_stop(symbol: str, close: float) -> float:
    """1.5 veces el rango medio de 14 sesiones. Si no hay historia, 2% del close."""
    path = _market_path(symbol)
    if not path.exists():
        return close * 0.98
    try:
        frame = pd.read_parquet(path, columns=["High", "Low", "Close"]).tail(15)
    except Exception:
        return close * 0.98
    if len(frame) < 5 or "High" not in frame.columns:
        return close * 0.98
    span = (pd.to_numeric(frame["High"], errors="coerce") - pd.to_numeric(frame["Low"], errors="coerce")).mean()
    if span is None or pd.isna(span) or span <= 0:
        return close * 0.98
    stop = close - 1.5 * float(span)
    if stop <= 0 or stop >= close:
        return close * 0.98
    return stop


def _quote(meta: dict, session: date, fx: float | None) -> dict | None:
    bar = _parquet_bar(meta["symbol"], session)
    if bar is None:
        bar = _yf_bars([meta["symbol"]], session).get(meta["symbol"])
    if bar is None:
        return None
    close = bar["close"]
    stop_local = _range_stop(meta["symbol"], close)
    if _in_mxn(meta["symbol"], meta["moneda"]):
        if not fx or fx <= 0:
            logger.info("sin MXN=X para %s", meta["ticker"])
            return None
        px = close / fx
        stop = stop_local / fx
    else:
        px = close
        stop = stop_local
    return {"entry_px": round(px, 4), "stop": round(stop, 4)}


def _candidates(session: date) -> list[dict]:
    index = _universe_index()
    fx = _fx_close(session)
    seen: set[str] = set()
    out: list[dict] = []
    for slug in ("bmv", "usa", "etfs", "cripto", "fibras"):
        top, horizon = _ranking(slug)
        for pick in top:
            ticker = str(pick.get("ticker") or "").upper()
            if not ticker or ticker in seen:
                continue
            meta = index.get(ticker)
            if meta is None or not _tradable(ticker):
                continue
            seen.add(ticker)
            quote = _quote(meta, session, fx)
            if quote is None:
                continue
            out.append({
                "ticker": ticker,
                "symbol": meta["symbol"],
                "usd": TICKET_USD,
                "stop": quote["stop"],
                "p_sube": None,
                "horizon_days": horizon or meta["horizon_days"],
                "entry_px": quote["entry_px"],
                "score": pick.get("score"),
            })
    return out


def _parse_json_obj(text: str) -> dict:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end < start:
        raise ValueError("la respuesta no trae JSON")
    return json.loads(raw[start : end + 1])


def _call_llm(context: dict) -> tuple[dict, str] | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    import httpx

    model = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-5-5")
    user = (
        "Reparte el libro para el siguiente día hábil. Long only. "
        "Máximo 3 nombres con dólares. Cada nombre lleva 0, o entre 10 y 25. "
        "La suma de los dólares más el efectivo es 50. "
        "No cambies el tamaño de una posición con menos de 5 días salvo que el precio haya tocado el stop. "
        "Responde solo un JSON con allocations (ticker, usd, stop_usd, horizon), cash_usd y motivo.\n"
        + json.dumps(context, ensure_ascii=False, default=_json_default)
    )
    response = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 2000,
            "system": "Eres el agente inversor. Respondes únicamente JSON.",
            "messages": [{"role": "user", "content": user}],
        },
        timeout=90,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Anthropic respondió {response.status_code}")
    body = response.json()
    content = "".join(
        block.get("text", "") for block in body.get("content") or [] if block.get("type") == "text"
    )
    return _parse_json_obj(content), body.get("model") or model


def _with_open_positions(book, candidates: list[dict], session: date) -> list[dict]:
    """Mete las posiciones del libro al frente, para que el plan no las pierda."""
    if book is None:
        return candidates
    have = {c["ticker"] for c in candidates}
    index = _universe_index()
    fx = _fx_close(session)
    extra = []
    for ticker in book.positions:
        if ticker in have:
            continue
        meta = index.get(ticker)
        if meta is None:
            continue
        quote = _quote(meta, session, fx)
        if quote is None:
            continue
        extra.append({
            "ticker": ticker,
            "symbol": meta["symbol"],
            "usd": round(float(book.positions[ticker]["usd"]), 2),
            "stop": quote["stop"],
            "p_sube": None,
            "horizon_days": meta["horizon_days"],
            "entry_px": quote["entry_px"],
            "score": None,
        })
    return extra + candidates


def _attach_p_sube(rows: list[dict], limit: int = 8) -> None:
    for row in rows[:limit]:
        if row.get("p_sube") is None and row.get("symbol"):
            row["p_sube"] = _p_sube(row["symbol"])


def _plan_from_agent(book, enter_on: date):
    """Claude elige tools (modelos, IV, horizontes) y cierra con la asignación."""
    from src.agent.book import Book
    from src.agent.investor import AgenteInversor

    agent_book = book if book is not None else Book()
    result = AgenteInversor(agent_book).act(apply=False)
    if not result.get("ok"):
        logger.warning("agente con tools no cerró: %s", result.get("error"))
        return None
    asig = result.get("asignacion") or {}
    index = _universe_index()
    fx = _fx_close(enter_on) or _fx_close(_today_mx())
    rows = []
    for alloc in asig.get("allocations") or []:
        ticker = str(alloc.get("ticker") or "").upper()
        meta = index.get(ticker)
        if meta is None:
            continue
        quote = _quote(meta, enter_on, fx) or _quote(meta, _today_mx(), fx)
        if quote is None:
            continue
        days = HORIZON_BDAYS.get(str(alloc.get("horizon") or ""), meta["horizon_days"])
        rows.append({
            "ticker": ticker,
            "symbol": meta["symbol"],
            "usd": round(float(alloc["usd"]), 2),
            "stop": quote["stop"],
            "p_sube": None,
            "enter_on": enter_on.isoformat(),
            "horizon_days": days,
            "entry_px": quote["entry_px"],
        })
    return rows, asig.get("motivo") or "tools", float(asig.get("cash_usd") or 0), result.get("modelo")


def _plan_from_llm(book, candidates: list[dict], enter_on: date) -> tuple[list[dict], str, float, str] | None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    by_ticker = {c["ticker"]: c for c in candidates}
    held = []
    if book is not None:
        for ticker, pos in book.positions.items():
            held.append({
                "ticker": ticker,
                "usd": pos.get("usd"),
                "stop_usd": pos.get("stop_usd"),
                "dias": book.days_open(ticker),
            })
    context = {
        "enter_on": enter_on.isoformat(),
        "libro": book.snapshot() if book is not None else None,
        "posiciones": held,
        "candidatos": [
            {
                "ticker": c["ticker"],
                "stop": c["stop"],
                "p_sube": c["p_sube"],
                "score": c["score"],
                "entry_px": c["entry_px"],
                "horizon_days": c["horizon_days"],
            }
            for c in candidates[:8]
        ],
    }
    try:
        parsed = _call_llm(context)
    except Exception as exc:
        logger.warning("xAI no respondió: %s", _redact(str(exc)))
        return None
    if parsed is None:
        return None
    proposal, model = parsed
    try:
        from src.agent.policy import PoliticaInversor

        prices = {c["ticker"]: float(c["entry_px"]) for c in candidates}
        guarded = PoliticaInversor().guard(proposal, book, prices)
    except Exception as exc:
        logger.warning("guard del plan: %s", _redact(str(exc)))
        return None
    if not guarded.get("ok"):
        logger.warning("el plan del LLM no pasó el guard: %s", guarded.get("error"))
        return None
    rows = []
    for alloc in guarded["allocations"]:
        known = by_ticker.get(alloc["ticker"])
        if known is None:
            continue
        stop = alloc.get("stop_usd")
        if stop is None:
            stop = known["stop"]
        horizon_key = str(alloc.get("horizon") or DEFAULT_HORIZON)
        rows.append({
            "ticker": alloc["ticker"],
            "symbol": known["symbol"],
            "usd": round(float(alloc["usd"]), 2),
            "stop": round(float(stop), 4),
            "p_sube": known.get("p_sube"),
            "enter_on": enter_on.isoformat(),
            "horizon_days": known["horizon_days"] if horizon_key not in HORIZON_BDAYS else HORIZON_BDAYS[horizon_key],
            "entry_px": known["entry_px"],
        })
    motivo = guarded.get("motivo") or "plan xAI"
    return rows, motivo, float(guarded["cash_usd"]), model


def _plan_from_rules(book, candidates: list[dict], enter_on: date) -> tuple[list[dict], str, float]:
    budget, max_names, min_ticket, max_ticket = _policy()
    index = _universe_index()
    rows: list[dict] = []
    if book is not None:
        for ticker, pos in book.positions.items():
            meta = index.get(ticker)
            if meta is None:
                continue
            known = next((c for c in candidates if c["ticker"] == ticker), None)
            stop = pos.get("stop_usd")
            if stop is None and known:
                stop = known["stop"]
            if stop is None:
                continue
            rows.append({
                "ticker": ticker,
                "symbol": meta["symbol"],
                "usd": round(float(pos["usd"]), 2),
                "stop": round(float(stop), 4),
                "p_sube": None if known is None else known.get("p_sube"),
                "enter_on": enter_on.isoformat(),
                "horizon_days": meta["horizon_days"] if known is None else known["horizon_days"],
                "entry_px": None if known is None else known.get("entry_px"),
            })
    invested = round(sum(r["usd"] for r in rows), 2)
    cash = round(budget - invested, 2)
    slots = max_names - len(rows)
    held = {r["ticker"] for r in rows}
    for cand in candidates:
        if slots <= 0 or cash < min_ticket:
            break
        if cand["ticker"] in held:
            continue
        if cand.get("p_sube") is not None and cand["p_sube"] < P_SUBE_MIN:
            continue
        usd = round(min(max_ticket, max(min_ticket, min(TICKET_USD, cash))), 2)
        if usd < min_ticket or usd - cash > 0.05:
            continue
        rows.append({
            "ticker": cand["ticker"],
            "symbol": cand["symbol"],
            "usd": usd,
            "stop": cand["stop"],
            "p_sube": cand.get("p_sube"),
            "enter_on": enter_on.isoformat(),
            "horizon_days": cand["horizon_days"],
            "entry_px": cand["entry_px"],
        })
        cash = round(cash - usd, 2)
        slots -= 1
        held.add(cand["ticker"])
    motivo = "sin ANTHROPIC_API_KEY; plan por libro y ranking, sin LLM"
    return rows, motivo, cash


def _cs_frames(session: date) -> dict[str, pd.DataFrame]:
    """Unos 15 meses de barras del núcleo, el satélite y MXN=X: parquet local,
    y lo que falte (en Cloud Run, todo) de yfinance en lotes."""
    from src.agent import core_satellite as cs

    start = session - timedelta(days=460)
    wanted = {name: cs.symbol_of(name) for name in cs.CORE}
    wanted.update(cs.satellite_symbols())
    wanted[cs.FX_SYMBOL] = cs.FX_SYMBOL
    frames: dict[str, pd.DataFrame] = {}
    missing: dict[str, str] = {}
    for name, symbol in wanted.items():
        frame = _history(symbol, start, session) if _market_path(symbol).exists() else pd.DataFrame()
        if frame.empty:
            missing[name] = symbol
        else:
            frames[name] = _ohlc_frame(frame)
    symbols = list(missing.values())
    for offset in range(0, len(symbols), 40):
        batch = symbols[offset : offset + 40]
        try:
            raw = _yf_download(batch, start, session)
        except Exception as exc:
            logger.warning("yfinance falló (%d símbolos): %s", len(batch), _redact(str(exc)))
            continue
        for name, symbol in missing.items():
            if symbol not in batch:
                continue
            frame = _ohlc_frame(_slice_symbol(raw, symbol, len(batch) > 1)).dropna(how="all")
            if frame.empty or "Close" not in frame.columns:
                continue
            idx = pd.to_datetime(frame.index)
            frame.index = idx.tz_localize(None) if getattr(idx, "tz", None) is not None else idx
            frames[name] = frame.sort_index()
    return frames


def _plan_from_core_satellite(book, session: date, enter_on: date) -> tuple[list[dict], str, float, str | None, dict]:
    """La regla pone los pesos; Haiku solo puede vetar nombres del satélite.
    El gemelo sin veto queda en raw para medir lo que aporta el LLM."""
    from src.agent import core_satellite as cs

    budget = _policy()[0]
    frames = _cs_frames(session)
    if "SPY" not in frames or cs.FX_SYMBOL not in frames:
        raise RuntimeError("sin historia de SPY o MXN=X para la regla")
    asof = pd.Timestamp(session)
    inputs = cs.Inputs.load()
    held = [t for t in (book.positions if book is not None else {}) if t not in cs.CORE]
    twin = cs.decide(frames, asof, inputs, equity_usd=budget, held_satellite=held)
    decision, model = twin, None
    if os.environ.get("ANTHROPIC_API_KEY"):
        from src.agent.veto import llm_veto

        decision = cs.decide(frames, asof, inputs, equity_usd=budget, held_satellite=held, veto=llm_veto)
        model = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-5-5")
    fx = cs.fx_on(frames, asof)
    rows = []
    for name, weight in decision.weights.items():
        symbol = cs.symbol_of(name)
        close = float(frames[name]["Close"].loc[:asof].dropna().iloc[-1])
        px = close / fx if cs.is_mx(symbol) else close
        stop = None if name in cs.CORE else round(px * (1 - cs.CATASTROPHE_DD), 4)
        rows.append({
            "ticker": name,
            "symbol": symbol,
            "usd": round(weight * budget, 2),
            "stop": stop,
            "p_sube": None,
            "enter_on": enter_on.isoformat(),
            "horizon_days": HORIZON_BDAYS["4s"],
            "entry_px": round(px, 4),
        })
    cash = round(budget - sum(r["usd"] for r in rows), 2)
    extra = {
        "regla": cs.config(),
        "gemelo": twin.weights,
        "vetos": decision.vetoed,
        "candidatos": decision.candidates,
        "tendencia": decision.trend,
    }
    return rows, decision.motivo, cash, model, extra


def _public_plan(row: dict) -> dict:
    return {
        "ticker": row["ticker"],
        "usd": row["usd"],
        "stop": row["stop"],
        "p_sube": row.get("p_sube"),
        "enter_on": row["enter_on"],
    }


def _horizontes(session: date) -> dict:
    """Tres horizontes del libro de 50. Si falla, el decide sigue sin ellos."""
    try:
        from src.agent.horizons import libro
        import pandas as pd
        return {
            "todo": libro(pd.Timestamp(session), "todo"),
            "bmv": libro(pd.Timestamp(session), "bmv"),
        }
    except Exception as exc:
        logger.warning("horizontes: %s", exc)
        return {}


def run_decide(session: date | None = None, force: bool = False) -> int:
    """Una decisión al mes: al cierre de la última sesión, para la apertura siguiente.
    Los demás días el libro no se toca."""
    from src.agent.core_satellite import is_rebalance_day

    session = session or _today_mx()
    enter_on = _next_bday(session)
    if not force and not is_rebalance_day(session, enter_on):
        logger.info("decide: %s no es cierre de mes, el libro no se toca", session.isoformat())
        return 0
    book = _load_book()
    try:
        plans, motivo, cash, model, extra = _plan_from_core_satellite(book, session, enter_on)
    except Exception as exc:
        logger.error("la regla núcleo/satélite falló: %s", _redact(str(exc)))
        return 1
    llm_used = model is not None
    decision = {
        "decided_on": session.isoformat(),
        "enter_on": enter_on.isoformat(),
        "llm_used": llm_used,
        "model": model,
        "motivo": motivo,
        "cash_usd": cash,
        "book": None if book is None else book.snapshot(),
        "horizontes": _horizontes(session),
        **extra,
    }
    for row in plans:
        _emit(_public_plan(row))
    if not plans:
        _emit({
            "ticker": None,
            "usd": 0,
            "stop": None,
            "p_sube": None,
            "enter_on": enter_on.isoformat(),
            "cash_usd": cash,
            "motivo": motivo,
        })
    try:
        decision_id = _write_decision(decision, plans)
    except Exception as exc:
        logger.error("no se escribió la decisión: %s", _redact(str(exc)))
        return 1
    if decision_id is None and plans:
        local_decision = str(uuid.uuid4())
        for row in plans:
            _append_jsonl(PLANS_PATH, {
                **row,
                "local_id": local_decision + ":" + row["ticker"],
                "decision_local": local_decision,
                "decided_on": session.isoformat(),
            })
    logger.info("decide enter_on=%s nombres=%d llm=%s", enter_on.isoformat(), len(plans), llm_used)
    return 0


# ── score ────────────────────────────────────────────────────────────────────


def _usd_window(symbol: str, in_mxn: bool, start: date, end: date, cache: dict) -> pd.DataFrame:
    key = (symbol, start.isoformat(), end.isoformat(), in_mxn)
    if key in cache:
        return cache[key]
    frame = _history(symbol, start, end)
    if frame.empty or "Close" not in frame.columns:
        cache[key] = pd.DataFrame()
        return cache[key]
    out = pd.DataFrame(index=frame.index)
    out["close"] = pd.to_numeric(frame["Close"], errors="coerce")
    low = frame["Low"] if "Low" in frame.columns else frame["Close"]
    out["low"] = pd.to_numeric(low, errors="coerce")
    if in_mxn:
        fx_key = ("MXN=X", start.isoformat(), end.isoformat())
        if fx_key not in cache:
            fx = _history("MXN=X", start, end)
            cache[fx_key] = fx
        fx = cache[fx_key]
        if fx is None or fx.empty or "Close" not in fx.columns:
            cache[key] = pd.DataFrame()
            return cache[key]
        fx_close = pd.to_numeric(fx["Close"], errors="coerce").rename("fx")
        joined = out.join(fx_close, how="left")
        joined["fx"] = joined["fx"].ffill()
        joined = joined.dropna(subset=["close", "fx"])
        joined = joined[joined["fx"] > 0]
        joined["close"] = joined["close"] / joined["fx"]
        joined["low"] = joined["low"] / joined["fx"]
        out = joined[["close", "low"]]
    cache[key] = out.dropna(subset=["close"])
    return cache[key]


def _score_one(plan: dict, as_of: date, cache: dict) -> dict | None:
    enter = _as_date(plan["enter_on"])
    horizon = int(plan.get("horizon_days") or _parse_horizon(plan.get("horizon")))
    exit_on = _add_bdays(enter, horizon)
    if as_of < exit_on:
        return None
    index = _universe_index()
    meta = index.get(plan["ticker"])
    symbol = plan.get("symbol") or (None if meta is None else meta["symbol"])
    if not symbol:
        logger.info("plan sin símbolo: %s", plan.get("ticker"))
        return None
    moneda = "USD" if meta is None else meta["moneda"]
    window = _usd_window(symbol, _in_mxn(symbol, moneda), enter, exit_on, cache)
    if window.empty:
        logger.info("sin precios para calificar %s %s→%s", plan["ticker"], enter, exit_on)
        return None
    entry_raw = _num(window.iloc[0]["close"])
    exit_raw = _num(window.iloc[-1]["close"])
    if not entry_raw or entry_raw <= 0 or exit_raw is None:
        return None
    entry_px = round(entry_raw, 4)
    exit_px = round(exit_raw, 4)
    stop = _num(plan.get("stop"))
    if stop is not None:
        stop = round(stop, 4)
    low = _num(window["low"].min())
    hit_stop = bool(stop is not None and low is not None and low <= stop)
    realized = stop if hit_stop else exit_px
    ret = (realized - entry_px) / entry_px
    usd = float(plan.get("usd") or 0)
    pnl = round(usd * ret, 4)
    plan_id = plan.get("id")
    if plan_id is None:
        plan_id = plan.get("local_id")
    note = (
        f"entrada={entry_px} close={exit_px} stop={stop} "
        f"pnl_usd={pnl} exit_on={exit_on.isoformat()}"
    )
    return {
        "entry_plan_id": plan.get("id"),
        "local_id": plan.get("local_id"),
        "ticker": plan["ticker"],
        "scored_on": as_of.isoformat(),
        "enter_on": enter.isoformat(),
        "exit_on": exit_on.isoformat(),
        "entry_px": entry_px,
        "exit_px": exit_px,
        "stop": stop,
        "hit_stop": hit_stop,
        "return_pct": round(ret, 6),
        "pnl_usd": pnl,
        "note": note,
        "_id": plan_id,
    }


def _already_scored_local() -> set:
    return {row.get("local_id") or row.get("_id") for row in _read_jsonl(OUTCOMES_PATH)}


def run_score(session: date | None = None) -> int:
    session = session or _today_mx()
    try:
        plans, from_db = _read_plans_to_score(session)
    except Exception as exc:
        logger.error("no se leyeron planes: %s", _redact(str(exc)))
        return 1
    done_local = set() if from_db else _already_scored_local()
    cache: dict = {}
    card = []
    failed = 0
    due = 0
    for plan in plans:
        if not from_db and (plan.get("local_id") in done_local):
            continue
        enter = _as_date(plan["enter_on"])
        horizon = int(plan.get("horizon_days") or _parse_horizon(plan.get("horizon")))
        if session < _add_bdays(enter, horizon):
            continue
        due += 1
        scored = _score_one(plan, session, cache)
        if scored is None:
            failed += 1
            continue
        card.append(scored)
    if card and from_db:
        payload = [{k: v for k, v in row.items() if k != "_id" and k != "local_id"} for row in card]
        try:
            _write_outcomes(payload)
        except Exception as exc:
            logger.error("no se escribió lab.outcome: %s", _redact(str(exc)))
            return 1
    if card and not from_db:
        for row in card:
            _append_jsonl(OUTCOMES_PATH, row)
    if not card:
        _emit({"scorecard": [], "as_of": session.isoformat(), "due": due})
    for row in card:
        _emit({
            "ticker": row["ticker"],
            "enter_on": row["enter_on"],
            "exit_on": row["exit_on"],
            "entrada": row["entry_px"],
            "close_futuro": row["exit_px"],
            "stop": row["stop"],
            "toco_stop": row["hit_stop"],
            "retorno": row["return_pct"],
            "pnl_usd": row["pnl_usd"],
        })
    logger.info("score as_of=%s due=%d scored=%d failed=%d", session.isoformat(), due, len(card), failed)
    if due and failed == due:
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Job diario Cloud Run del laboratorio mercado")
    parser.add_argument("--mode", required=True, choices=("ingest", "decide", "score"))
    parser.add_argument("--force", action="store_true", help="decide aunque no sea cierre de mes")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    _load_env()
    _align_db_env()
    if args.mode == "ingest":
        return run_ingest()
    if args.mode == "decide":
        return run_decide(force=args.force)
    return run_score()


if __name__ == "__main__":
    sys.exit(main())
