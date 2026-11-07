"""Herramientas que Claude puede pedir. Cada una es un subagente de observación."""
from __future__ import annotations

import json
from datetime import date
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
    """Carga señales de un universo sin confundir ranking con apuestas LONG.

    El historial del SuperModel solo se usa para BMV cuando el JSON declara
    universe='bmv'. Para otros universos conservamos el archivo estático.
    El ranking es informativo; solo LONG y presentes en ese ranking pueden
    autorizar una compra. Una señal caducada no se rejuvenece aquí.
    """
    history_dir = SIGNALS / "history"
    if slug == "bmv" and history_dir.exists():
        # Elegir el último snapshot *válido* de BMV, no simplemente el último nombre.
        for path in sorted(history_dir.glob("????-??-??.json"), reverse=True):
            try:
                date.fromisoformat(path.stem)
                raw = json.loads(path.read_text(encoding="utf-8"))
                if str(raw.get("universe", "")).lower() != slug:
                    continue
                if raw.get("date") != path.stem:
                    continue
                top10 = raw.get("top10_ranking")
                bets = raw.get("bets")
                if not isinstance(top10, list) or not isinstance(bets, list):
                    continue
                # El archivo del 31/oct tiene, aun declarando BMV, nombres de USA.
                # Solo consideramos tickers del catálogo local de BMV.
                permitidos = set(SLUGS[slug][1]["tickers"])
                ranked = [
                    r for r in top10
                    if isinstance(r, dict)
                    and str(r.get("ticker", "")).upper() in permitidos
                ]
                longs = {
                    str(b.get("ticker", "")).upper()
                    for b in bets if isinstance(b, dict)
                    and str(b.get("direction", "")).upper() == "LONG"
                    and b.get("horizon_days") == 20
                }
                # Este contrato exige estar en el top + en apuestas LONG.
                top = [
                    {
                        "puesto": r.get("rank", i + 1),
                        "ticker": str(r["ticker"]).upper(),
                        "score": r.get("conviction"),
                        "razones": ["ranking_historico_supermodel"],
                        "precio_actual": None,
                    }
                    for i, r in enumerate(ranked[:5])
                ]
                comprables = [r["ticker"] for r in top if r["ticker"] in longs]
                return {
                    "fecha": path.stem,
                    "mercado": {},
                    "consensus": [],
                    "by_horizon": {
                        "4s": {
                            "top5": top,
                            "comprables": comprables,
                            # El histórico no trae un campo 'evitar'. No lo inventamos.
                            "evitar": [],
                        }
                    },
                    "_fuente": f"historial_diario:{path.name}",
                    "_nota": "Solo LONG en bets y en top BMV autorizan nuevas compras. "
                             "El historial no proporciona lista evitar.",
                }
            except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                # Ignorar un snapshot malformado y revisar el siguiente más reciente.
                continue

    # En USA, ETFs, cripto y fibras se conserva su señal estática propia.
    # Si no existe historial válido de BMV, también se usa su archivo estático.
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
            return {"ok": False, "error": f"no hay señal disponible para {slug}"}
        by_h = sig.get("by_horizon") or {}
        # Un snapshot de 20 días no debe etiquetarse falsamente como 1s/2s/3s.
        if horizonte not in by_h and sig.get("_fuente", "").startswith("historial_diario:"):
            return {"ok": False, "error": f"el historial SuperModel solo publica horizonte 4s; solicitado {horizonte}"}
        block = by_h.get(horizonte) or by_h.get("4s") or {}
        consensus = {c["ticker"]: c for c in sig.get("consensus") or []}
        picks = []
        for i, item in enumerate(block.get("top5") or [], start=1):
            cons = consensus.get(item["ticker"]) or {}
            picks.append({
                "puesto": item.get("puesto", i),
                "ticker": item["ticker"],
                "score": item.get("score"),
                "razones": item.get("razones") or [],
                "precio": item.get("precio_actual"),
                "horizontes": cons.get("n_horizontes"),
            })
        out = {
            "ok": True,
            "universo": slug,
            "horizonte": horizonte,
            "fecha_senal": sig.get("fecha"),
            "mercado": sig.get("mercado"),
            "top": picks,
            "evitar": block.get("evitar") or [],
        }
        if "comprables" in block:
            out["comprables"] = [t for t in block["comprables"] if t in {p["ticker"] for p in picks}]
            out["nota"] = sig.get("_nota", "")
        return out

    def tool_schema(self) -> dict:
        return _schema(self.name, self.description, {
            "universo": {"type": "string", "enum": list(SLUGS), "description": "bmv, usa, etfs, cripto o fibras"},
            "horizonte": {"type": "string", "enum": ["1s", "2s", "3s", "4s"], "description": "1s es una semana, 4s es un mes"},
        }, required=["universo"])

    def comprables(self, resultado: dict) -> list[str]:
        # Señales históricas: autorización explícita de bets LONG ∩ top BMV.
        # Señales estáticas antiguas: mantener el contrato original (top).
        if "comprables" in resultado:
            return list(resultado["comprables"] or [])
        return [p["ticker"] for p in resultado.get("top") or []]

    def evitados(self, resultado: dict) -> list[str]:
        return list(resultado.get("evitar") or [])


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


class AgenteModelos(ObservationAgent):
    name = "ver_modelos"
    description = (
        "Information Value de las variables y el AUC del LSTM y del ranking. "
        "Léelo antes de repartir: IV bajo 0.02 no separa, AUC 0.50 es un volado."
    )

    def observe(self, **params) -> dict:
        root = ROOT / "data"

        def _iv(path: Path, key: str | None = None) -> list:
            if not path.exists():
                return []
            payload = json.loads(path.read_text())
            rows = payload.get(key) if key else payload.get("iv")
            if rows is None and "iv_summary" in payload:
                rows = payload["iv_summary"]
            return [
                {"feature": r.get("feature"), "iv": r.get("iv")}
                for r in (rows or [])
            ]

        lstm = []
        lstm_path = root / "eval" / "lstm_vs_lgbm.json"
        if lstm_path.exists():
            raw = json.loads(lstm_path.read_text())
            summary = raw.get("summary") or []
            lstm = [
                {
                    "ticker": row.get("ticker"),
                    "auc_lstm": row.get("lstm_seq_mean_auc"),
                    "auc_ranking": row.get("xs_lgbm_mean_auc"),
                    "lstm_wins": row.get("lstm_wins"),
                    "folds": [
                        {
                            "fold": fold.get("fold"),
                            "test_start": fold.get("test_start"),
                            "test_end": fold.get("test_end"),
                            "auc_lstm": fold.get("lstm_seq_auc"),
                            "auc_ranking": fold.get("xs_lgbm_auc"),
                        }
                        for fold in ((raw.get("tickers") or {}).get(row.get("ticker"), {}).get("folds") or [])
                    ],
                }
                for row in summary
            ]
        return {
            "ok": True,
            "umbral_iv": 0.02,
            "nota": "IV menor a 0.02 no distingue. AUC 0.50 es un volado. El LSTM no tiene probabilidad de hoy, solo estas pruebas.",
            "woe_patrones_20d": _iv(root / "woe_pattern.json"),
            "woe_50_extra": _iv(root / "woe_extra_iv.json"),
            "woe_valor": _iv(root / "woe_value_iv.json"),
            "woe_clasico": _iv(root / "woe_results.json"),
            "flujo": _iv(root / "eval" / "flow_iv.json"),
            "lstm_auc_5d": lstm,
        }

    def tool_schema(self) -> dict:
        return _schema(self.name, self.description, {})


class AgenteHorizontes(ObservationAgent):
    name = "ver_horizontes"
    description = (
        "Ranking completo de cada plazo: todas las emisoras con score y precio. "
        "Mañana y semana ordenan por reversión. El mes, por momentum de 12 meses. "
        "libro es solo un reparto posible. Tú eliges cualquier nombre de ranking_completo."
    )

    def observe(self, manga: str = "bmv", **params) -> dict:
        sleeve = params.get("sleeve") or manga or "bmv"
        if sleeve not in ("bmv", "todo"):
            sleeve = "bmv"
        from src.agent.horizons import libro
        asof = pd.Timestamp.today().normalize()
        # El último cierre que exista en un precio líquido.
        sample = ROOT / "data" / "market" / "WALMEX.MX.parquet"
        if sample.exists():
            idx = pd.read_parquet(sample, columns=["Close"]).index
            asof = pd.to_datetime(idx).tz_localize(None).max()
        block = libro(asof, sleeve)
        from src.agent.horizons import rank_horizon
        block["ranking_completo"] = {
            kind: rank_horizon(asof, kind, sleeve=sleeve, top_n=500)
            for kind in ("manana", "semana", "mes")
        }
        return {"ok": True, **block}

    def tool_schema(self) -> dict:
        return _schema(self.name, self.description, {
            "sleeve": {"type": "string", "enum": ["bmv", "todo"], "description": "bmv solo mexicanas, todo incluye USA y cripto"},
        })


def build_tools(book: Book) -> dict[str, ObservationAgent]:
    agents = [
        AgenteLibro(book),
        AgenteRanking(),
        AgenteClima(),
        AgenteWoe(),
        AgenteNoticias(),
        AgenteSalida(),
        AgenteModelos(),
        AgenteHorizontes(),
    ]
    return {a.name: a for a in agents}
