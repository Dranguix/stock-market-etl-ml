"""
Features de ex-dividendo para el modelo BMV.

Fuente: DataBursátil /v2/emisoras → campo 'dividendos'
Almacén: data/dividends/dividends.json (actualizado semanalmente)

Features computadas (por ticker, por día):
    dias_a_ex_date  — días al próximo ex-date (negativo = ya pasó, positivo = futuro)
    ex_date_window  — 1 si el día está en ventana [-3, +10] alrededor del ex-date más cercano
    div_yield_fwd   — dividendo por acción anualizado / precio de cierre del día

El ex-date effect en BMV es robusto: las acciones caen mecánicamente en el
ex-date y frecuentemente se recuperan en los 5-10 días siguientes (PEAD),
especialmente en tickers de alta liquidez.

Uso:
    from src.ml.dividend_features import (
        fetch_all_dividends, build_dividend_panel,
        add_dividend_features, DIVIDEND_FEATURE_COLS,
    )
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Rutas ─────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent.parent
DIVIDENDS_DIR  = _ROOT / "data" / "dividends"
DIVIDENDS_FILE = DIVIDENDS_DIR / "dividends.json"

# ── Columnas de salida ─────────────────────────────────────────────────────────
DIVIDEND_FEATURE_COLS: list[str] = [
    "dias_a_ex_date",   # días al próximo ex-date (entero, NaN si no hay)
    "ex_date_window",   # 1.0 si en ventana [-3, +10] del ex-date
    "div_yield_fwd",    # dividendo por acción / precio (% estimado, point-in-time)
]

# Ventana de efecto ex-dividendo
_EX_PRE_DAYS  = 3    # días antes del ex-date que el mercado empieza a descontar
_EX_POST_DAYS = 10   # días post ex-date donde suele haber rebote


# ── 1. Fetch de dividendos desde DataBursátil ─────────────────────────────────

def _parse_dividend_events(serie_data: dict) -> list[dict]:
    """Extrae lista de eventos de dividendo de un bloque de serie."""
    divs = serie_data.get("dividendos") or {}
    events: list[dict] = []

    reciente = divs.get("reciente") or {}
    for fecha_pago, info in reciente.items():
        if not isinstance(info, dict):
            continue
        ex_date = info.get("fechaexcupon") or fecha_pago
        events.append({
            "pago_date": fecha_pago,
            "ex_date":   ex_date,
            "pago":      float(info.get("pago", 0) or 0),
            "tipo":      info.get("tipo", ""),
            "divisa":    info.get("divisa", "MXN"),
            "source":    "reciente",
        })

    historico = divs.get("historico") or {}
    for fecha_pago, info in historico.items():
        if not isinstance(info, dict):
            continue
        try:
            dt_pago = datetime.strptime(fecha_pago, "%Y-%m-%d").date()
            dt_ex   = dt_pago - timedelta(days=1)
        except ValueError:
            dt_ex = date.fromisoformat(fecha_pago) - timedelta(days=1)
        events.append({
            "pago_date": fecha_pago,
            "ex_date":   dt_ex.strftime("%Y-%m-%d"),
            "pago":      float(info.get("pago", 0) or 0),
            "tipo":      info.get("tipo", ""),
            "divisa":    "MXN",
            "source":    "historico",
        })

    return sorted(events, key=lambda e: e["ex_date"])


def _build_api_lookup(raw_all: dict) -> dict[str, dict]:
    """
    Construye lookup: full_ticker_str → serie_data.

    La respuesta de /v2/emisoras?mercado=local tiene estructura:
        { "GFNORTE": { "O": {serie_data}, ... }, "AMX": { "B": {...}, "L": {...} }, ... }

    Generamos todas las combinaciones emisora+serie (con y sin separador),
    e.g. "GFNORTEO", "GFNORTE_O", "GFNORTE O", etc.
    """
    lookup: dict[str, dict] = {}
    for emisora, series_dict in raw_all.items():
        if not isinstance(series_dict, dict):
            continue
        for serie, serie_data in series_dict.items():
            if not isinstance(serie_data, dict):
                continue
            # Distintas formas en que el ticker puede aparecer en nuestros sistemas:
            # GFNORTE+O → "GFNORTEO", AMX+B → "AMXB", FEMSA+UBD → "FEMSAUBD"
            full = emisora + serie.replace("*", "")
            lookup[full] = serie_data
            lookup[emisora] = serie_data          # sin serie (fallback)
            lookup[emisora + "_" + serie] = serie_data
    return lookup


def fetch_all_dividends(
    tickers: list[str],
    token: str | None = None,
    *,
    timeout: int = 60,
    sleep: float = 0.0,
    force_refresh: bool = False,
) -> dict[str, list[dict]]:
    """
    Obtiene dividendos de /v2/emisoras (una sola llamada, mercado=local completo).

    La API devuelve estructura {emisora: {serie: data}} — se resuelve la
    correspondencia con nuestros tickers internos via concatenación emisora+serie.

    Cachea en DIVIDENDS_FILE por 7 días.
    """
    if not force_refresh and DIVIDENDS_FILE.exists():
        age_days = (date.today() - date.fromtimestamp(DIVIDENDS_FILE.stat().st_mtime)).days
        if age_days <= 7:
            logger.info("Usando caché de dividendos (%d días de antigüedad)", age_days)
            return json.loads(DIVIDENDS_FILE.read_text())

    if token is None:
        token = os.environ.get("DATABURSATIL_API_TOKEN", "")
    if not token:
        logger.warning("DATABURSATIL_API_TOKEN no definido — dividendos sin fetch")
        return {}

    import requests

    # ── Una sola llamada para todo el mercado local ───────────────────────────
    if sleep > 0:
        time.sleep(sleep)
    try:
        r = requests.get(
            "https://api.databursatil.com/v2/emisoras",
            params={"token": token, "mercado": "local"},
            timeout=timeout,
        )
        r.raise_for_status()
        raw_all = r.json()
        logger.info("Emisoras locales descargadas: %d entradas, %.1f KiB",
                    len(raw_all), len(r.content) / 1024)
    except Exception as exc:
        logger.error("Error fetching all emisoras: %s", exc)
        return {}

    # ── Construir lookup full_ticker → serie_data ─────────────────────────────
    lookup = _build_api_lookup(raw_all)

    # ── También usar el mapeo BMV_EMISORA_SERIE si está disponible ────────────
    try:
        from src.data.bmv_catalog import BMV_EMISORA_SERIE
        # BMV_EMISORA_SERIE: "GFNORTEO" → "GFNORTEO", "AMXL" → "AMXB", etc.
    except ImportError:
        BMV_EMISORA_SERIE = {}

    result: dict[str, list[dict]] = {}
    for ticker in tickers:
        # Intentar resolución: ticker interno → API ticker
        api_ticker = BMV_EMISORA_SERIE.get(ticker, ticker).replace("*", "")
        serie_data = lookup.get(api_ticker) or lookup.get(ticker)
        if serie_data is None:
            logger.debug("Sin match dividendos para %s (api_ticker=%s)", ticker, api_ticker)
            continue
        events = _parse_dividend_events(serie_data)
        if events:
            result[ticker] = events
            logger.debug("Dividendos %s: %d eventos", ticker, len(events))

    DIVIDENDS_DIR.mkdir(parents=True, exist_ok=True)
    DIVIDENDS_FILE.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    logger.info("Dividendos guardados para %d/%d tickers", len(result), len(tickers))
    return result


# ── 2. Panel de features de dividendo ─────────────────────────────────────────

def build_dividend_panel(
    dividends: dict[str, list[dict]],
    date_index: pd.DatetimeIndex,
    close_panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Construye un panel (date × ticker) con DIVIDEND_FEATURE_COLS.

    Parameters
    ----------
    dividends   : salida de fetch_all_dividends
    date_index  : rango de fechas para el panel
    close_panel : DataFrame de precios de cierre (date × ticker) para div_yield_fwd

    Returns
    -------
    DataFrame con MultiIndex (date, ticker) o columnas (ticker_feature).
    """
    tickers = list(dividends.keys())
    dates   = pd.to_datetime(date_index)

    records = []
    for ticker in tickers:
        events = dividends.get(ticker, [])
        if not events:
            for dt in dates:
                records.append({
                    "date": dt, "ticker": ticker,
                    "dias_a_ex_date": np.nan,
                    "ex_date_window": 0.0,
                    "div_yield_fwd":  np.nan,
                })
            continue

        # Extraer ex_dates y pagos como arrays numpy
        ex_dates_arr = pd.to_datetime([e["ex_date"] for e in events])
        pagos        = np.array([e["pago"] for e in events], dtype=float)

        for dt in dates:
            # ── dias_a_ex_date: días al ex-date más cercano (futuro o reciente) ──
            deltas_arr = (ex_dates_arr - dt).days.values  # numpy int64 array
            # Prefiere ex-dates no muy pasados (hasta 30 días antes)
            future_mask = deltas_arr >= -30
            if future_mask.any():
                sub_deltas = deltas_arr[future_mask]
                sub_pagos  = pagos[future_mask]
                idx_nearest   = int(np.abs(sub_deltas).argmin())
                delta_nearest = int(sub_deltas[idx_nearest])
                pago_nearest  = float(sub_pagos[idx_nearest])
            else:
                # Todos muy viejos: usar el más reciente
                delta_nearest = int(deltas_arr[-1])
                pago_nearest  = float(pagos[-1])

            # ── ex_date_window ─────────────────────────────────────────────────
            in_window = float(-_EX_PRE_DAYS <= delta_nearest <= _EX_POST_DAYS)

            # ── div_yield_fwd ──────────────────────────────────────────────────
            div_yield = np.nan
            if close_panel is not None and ticker in close_panel.columns:
                price_col = close_panel[ticker]
                price = price_col.reindex([dt], method="ffill").iloc[0]
                if price and price > 0 and pago_nearest > 0:
                    # Anualizamos: WALMEX paga ~2 veces al año
                    freq_estimate = 2.0  # conservador; en producción calcular de historial
                    div_yield = (pago_nearest * freq_estimate) / price * 100

            records.append({
                "date":           dt,
                "ticker":         ticker,
                "dias_a_ex_date": float(delta_nearest),
                "ex_date_window": in_window,
                "div_yield_fwd":  div_yield,
            })

    if not records:
        return pd.DataFrame()

    panel = pd.DataFrame(records).set_index(["date", "ticker"])
    return panel


# ── 3. Merge en DataFrame de ticker ──────────────────────────────────────────

def add_dividend_features(
    feat_df: pd.DataFrame,
    ticker: str,
    div_panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Añade DIVIDEND_FEATURE_COLS a un DataFrame de features de un ticker.

    Parameters
    ----------
    feat_df   : DataFrame con DatetimeIndex
    ticker    : clave del ticker
    div_panel : salida de build_dividend_panel (MultiIndex date×ticker)

    Returns
    -------
    feat_df con columnas de dividendo añadidas (NaN si no hay datos).
    """
    for col in DIVIDEND_FEATURE_COLS:
        feat_df[col] = np.nan

    if div_panel is None or div_panel.empty:
        return feat_df

    try:
        if ticker not in div_panel.index.get_level_values("ticker"):
            return feat_df
        ticker_data = div_panel.xs(ticker, level="ticker")
        idx = pd.DatetimeIndex(feat_df.index)
        ticker_data = ticker_data.reindex(idx, method="ffill")
        for col in DIVIDEND_FEATURE_COLS:
            if col in ticker_data.columns:
                feat_df[col] = ticker_data[col].values
    except Exception as exc:
        logger.debug("add_dividend_features(%s): %s", ticker, exc)

    return feat_df


# ── 4. Helper rápido: div events como list para un ticker ─────────────────────

def get_upcoming_ex_dates(
    ticker: str,
    as_of: date | None = None,
    dividends: dict[str, list[dict]] | None = None,
    n: int = 3,
) -> list[dict]:
    """
    Devuelve los próximos n ex-dates del ticker desde as_of.
    Útil para el InsightsBundle y reportes.
    """
    if as_of is None:
        as_of = date.today()
    if dividends is None:
        if DIVIDENDS_FILE.exists():
            dividends = json.loads(DIVIDENDS_FILE.read_text())
        else:
            return []
    events = dividends.get(ticker, [])
    upcoming = [
        e for e in events
        if date.fromisoformat(e["ex_date"]) >= as_of
    ]
    return sorted(upcoming, key=lambda e: e["ex_date"])[:n]
