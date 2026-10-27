"""Herramientas que Claude puede pedir. Cada una es un subagente de observación."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.agent.base import ObservationAgent
from src.agent.book import Book
from src.agent.woe_obs import explain_ticker
from src.data.universes import UNIVERSES
from src.ml.risk_engine import atr_stop

ROOT = Path(__file__).resolve().parents[2]
SIGNALS = ROOT / "data" / "signals"
MARKET = ROOT / "data" / "market"

SLUGS = {cfg["slug"]: (name, cfg) for name, cfg in UNIVERSES.items()}


def find_ticker(ticker: str) -> tuple[str, dict, str] | None:
    key = ticker.upper().strip()
    for name, cfg in UNIVERSES.items():
        if key in cfg["tickers"]:
            return key, cfg, cfg["tickers"][key]
    return None


def _signal(slug: str) -> dict:
    path = SIGNALS / f"{slug}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _last_close(symbol: str) -> float | None:
    path = MARKET / f"{symbol}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=["Close"])
    if df.empty:
        return None
    return float(df["Close"].iloc[-1])


def usd_mxn() -> float | None:
    for name in ("MXN=X.parquet", "MXN=X.MX.parquet"):
        path = MARKET / name
        if path.exists():
            df = pd.read_parquet(path, columns=["Close"])
            if not df.empty:
                return float(df["Close"].iloc[-1])
    return None


def price_usd(ticker: str) -> dict:
    found = find_ticker(ticker)
    if not found:
        return {"ok": False, "error": f"{ticker} no está en los universos"}
    name, cfg, sym = found
    px = _last_close(sym)
    slug = cfg["slug"]
    if px is None:
        sig = _signal(slug)
        for block in (sig.get("by_horizon") or {}).values():
            for p in block.get("top5") or []:
                if p["ticker"] == name and p.get("precio_actual"):
                    px = float(p["precio_actual"])
                    break
    if px is None:
        return {"ok": False, "error": f"sin precio para {name}", "symbol": sym}

    moneda = cfg["moneda"]
    fx = usd_mxn()
    en_mxn = moneda == "MXN" or sym.endswith(".MX")
    if en_mxn:
        if not fx:
            return {"ok": False, "error": "falta MXN=X para pasar el precio a dólares", "precio": px}
        usd = px / fx
    else:
        usd = px
    return {
        "ok": True,
        "ticker": name,
        "symbol": sym,
        "slug": slug,
        "precio": round(px, 4),
        "moneda": "MXN" if en_mxn else "USD",
        "usd_mxn": None if fx is None else round(fx, 4),
        "precio_usd": round(usd, 4),
    }


class AgenteLibro(ObservationAgent):
    name = "ver_libro"
    description = "Efectivo y posiciones abiertas del libro de 50 dólares."

    def __init__(self, book: Book):
        self.book = book

    def observe(self, **params) -> dict:
        return {"ok": True, **self.book.snapshot()}

    def tool_schema(self) -> dict:
        return _schema(self.name, self.description, {})


class AgenteRanking(ObservationAgent):
    name = "ver_ranking"
    description = (
        "Orden del modelo cross-seccional ya calculado. "
        "No lo recalcula: lee data/signals. El puesto 1 es el de mayor score."
    )

    def observe(self, universo: str = "bmv", horizonte: str = "4s", **params) -> dict:
        slug = universo.lower().strip()
        if slug not in SLUGS:
            return {"ok": False, "error": f"universo {universo}. Usa: {', '.join(SLUGS)}"}
        sig = _signal(slug)
        if not sig:
            return {"ok": False, "error": f"no hay data/signals/{slug}.json"}
        by_h = sig.get("by_horizon") or {}
        block = by_h.get(horizonte) or by_h.get("4s") or {}
        consensus = {c["ticker"]: c for c in sig.get("consensus") or []}
        picks = []
        for i, p in enumerate(block.get("top5") or [], start=1):
            cons = consensus.get(p["ticker"]) or {}
            picks.append({
                "puesto": i,
                "ticker": p["ticker"],
                "score": p.get("score"),
                "razones": p.get("razones") or [],
                "precio": p.get("precio_actual"),
                "horizontes": cons.get("n_horizontes"),
            })
        return {
            "ok": True,
            "universo": slug,
            "horizonte": horizonte,
            "fecha_senal": sig.get("fecha"),
            "mercado": sig.get("mercado"),
            "top": picks,
            "evitar": block.get("evitar") or [],
        }

    def tool_schema(self) -> dict:
        return _schema(self.name, self.description, {
            "universo": {"type": "string", "enum": list(SLUGS), "description": "bmv, usa, etfs, cripto o fibras"},
            "horizonte": {"type": "string", "enum": ["1s", "2s", "3s", "4s"], "description": "1s es una semana, 4s es un mes"},
        }, required=["universo"])


class AgenteClima(ObservationAgent):
    name = "ver_clima"
    description = "Semáforo del mercado y los tres números grandes: dólar, petróleo y S&P 500."

    def observe(self, **params) -> dict:
        climas = {}
        for slug in SLUGS:
            sig = _signal(slug)
            if sig.get("mercado"):
                climas[slug] = sig["mercado"]
        numeros = {}
        for label, fname, days in (
            ("usdmxn_ret5_pct", "MXN=X.parquet", 5),
            ("wti_ret5_pct", "CL=F.parquet", 5),
            ("sp500_ret5_pct", "GSPC.parquet", 5),
        ):
            path = MARKET / fname
            if not path.exists():
                continue
            close = pd.read_parquet(path, columns=["Close"])["Close"]
            if len(close) > days:
                numeros[label] = round(float(close.pct_change(days).iloc[-1]) * 100, 2)
        return {"ok": True, "por_universo": climas, "macro_5d_pct": numeros}

    def tool_schema(self) -> dict:
        return _schema(self.name, self.description, {})


class AgenteWoe(ObservationAgent):
    name = "ver_woe"
    description = (
        "Por qué este ticker: bin de cada variable, su WoE, su Information Value "
        "y la probabilidad de que suba en 5 días."
    )

    def observe(self, ticker: str = "", **params) -> dict:
        found = find_ticker(ticker)
        if not found:
            return {"ok": False, "error": f"{ticker} no está en los universos"}
        name, cfg, sym = found
        out = explain_ticker(sym)
        out["ticker"] = name
        out["universo"] = cfg["slug"]
        return out

    def tool_schema(self) -> dict:
        return _schema(self.name, self.description, {
            "ticker": {"type": "string", "description": "Nombre corto, por ejemplo WALMEX o SLV"},
        }, required=["ticker"])


class AgenteNoticias(ObservationAgent):
    name = "ver_noticias"
    description = (
        "Titulares del ticker. Por defecto usa los que ya vienen en la señal. "
        "en_vivo=true baja noticias nuevas (Yahoo, Google, DataBursátil)."
    )

    def observe(self, ticker: str = "", en_vivo: bool = False, **params) -> dict:
        found = find_ticker(ticker)
        if not found:
            return {"ok": False, "error": f"{ticker} no está en los universos"}
        name, cfg, sym = found
        if not en_vivo:
            cached = _cached_news(cfg["slug"], name)
            return {"ok": True, "ticker": name, "fuente": "senal_guardada", "titulares": cached[:5]}

        from src.api.gemini_analyst import fetch_news_for_ticker
        news = fetch_news_for_ticker(name, sym, cfg["slug"], max_items=5) or []
        return {
            "ok": True,
            "ticker": name,
            "fuente": "en_vivo",
            "titulares": [
                {"title": n.get("title"), "publisher": n.get("publisher"), "ago": n.get("ago")}
                for n in news[:5]
            ],
        }

    def tool_schema(self) -> dict:
        return _schema(self.name, self.description, {
            "ticker": {"type": "string"},
            "en_vivo": {"type": "boolean", "description": "true solo si la señal guardada no alcanza"},
        }, required=["ticker"])


class AgenteSalida(ObservationAgent):
    name = "ver_salida"
    description = "Precio de entrada aproximado y stop a 1.5 ATR. También el precio en dólares."

    def observe(self, ticker: str = "", **params) -> dict:
        quote = price_usd(ticker)
        if not quote.get("ok"):
            return quote
        found = find_ticker(ticker)
        assert found is not None
        _name, _cfg, sym = found
        from src.agent.woe_obs import latest_features
        row, _status = latest_features(sym)
        atr = None
        if row is not None and row.get("atr_pct") == row.get("atr_pct"):
            atr = float(quote["precio"]) * float(row["atr_pct"]) / 100.0
        if not atr or atr <= 0:
            atr = float(quote["precio"]) * 0.02
            atr_nota = "ATR no disponible, use 2% del precio"
        else:
            atr_nota = "ATR(14)"
        entry = round(float(quote["precio"]) * 1.0025, 4)
        stop, t1, t2 = atr_stop(entry, atr)
        fx = quote.get("usd_mxn") or 1.0
        en_mxn = quote["moneda"] == "MXN"

        def to_usd(px: float) -> float:
            return round(px / fx, 4) if en_mxn else round(px, 4)

        return {
            "ok": True,
            "ticker": quote["ticker"],
            "precio": quote["precio"],
            "moneda": quote["moneda"],
            "precio_usd": quote["precio_usd"],
            "entrada": entry,
            "stop": stop,
            "stop_usd": to_usd(stop),
            "objetivo_1": t1,
            "objetivo_2": t2,
            "atr_nota": atr_nota,
        }

    def tool_schema(self) -> dict:
        return _schema(self.name, self.description, {
            "ticker": {"type": "string"},
        }, required=["ticker"])


def _cached_news(slug: str, ticker: str) -> list[dict]:
    sig = _signal(slug)
    for block in (sig.get("by_horizon") or {}).values():
        for p in block.get("top5") or []:
            if p["ticker"] == ticker and p.get("noticias"):
                return [
                    {"title": n.get("title"), "publisher": n.get("publisher"), "ago": n.get("ago")}
                    for n in p["noticias"]
                ]
    return []


def _schema(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    schema: dict = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return {"name": name, "description": description, "input_schema": schema}


def asignacion_schema() -> dict:
    return {
        "name": "proponer_asignacion",
        "description": "Cierra el ciclo. Reparte los 50 dólares. El sistema valida la suma y los límites.",
        "input_schema": {
            "type": "object",
            "required": ["allocations"],
            "properties": {
                "allocations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["ticker", "usd"],
                        "properties": {
                            "ticker": {"type": "string"},
                            "usd": {"type": "number"},
                            "horizon": {"type": "string"},
                            "stop_usd": {"type": "number"},
                        },
                    },
                },
                "cash_usd": {"type": "number"},
                "motivo": {"type": "string"},
            },
        },
    }


def build_tools(book: Book) -> dict[str, ObservationAgent]:
    agents = [
        AgenteLibro(book),
        AgenteRanking(),
        AgenteClima(),
        AgenteWoe(),
        AgenteNoticias(),
        AgenteSalida(),
    ]
    return {a.name: a for a in agents}
