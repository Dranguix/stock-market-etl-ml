"""Valor sin contar las acciones en circulación.

earnings_yield_ttm  utilidad por acción de los últimos 4 reportes / precio
div_yield_ttm       dividendos por acción de los últimos 365 días / precio
eps_surprise        (reportado − estimado) / |estimado| del último anuncio
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ttm_eps_on_prices(close: pd.Series, events: pd.DataFrame) -> pd.Series:
    """Suma de los últimos 4 EPS ya anunciados, alineada al cierre. Sin look-ahead."""
    if events is None or events.empty or "reported" not in events.columns:
        return pd.Series(np.nan, index=close.index)
    ev = events.dropna(subset=["reported"]).copy()
    ev["announced"] = pd.to_datetime(ev["announced"]).dt.tz_localize(None).dt.normalize()
    ev = ev.sort_values("announced").drop_duplicates("announced", keep="last")
    ev["ttm"] = ev["reported"].rolling(4, min_periods=4).sum()
    # El número se puede usar al día siguiente del anuncio.
    ev["available"] = ev["announced"] + pd.Timedelta(days=1)
    left = pd.DataFrame({"date": pd.to_datetime(close.index).tz_localize(None)})
    right = ev[["available", "ttm"]].rename(columns={"available": "date"})
    merged = pd.merge_asof(left.sort_values("date"), right.sort_values("date"), on="date", direction="backward")
    out = pd.Series(merged["ttm"].to_numpy(), index=close.index)
    return out / close.replace(0, np.nan)


def dividend_yield_on_prices(close: pd.Series, dividends: list[dict]) -> pd.Series:
    """Suma de dividendos por acción con ex-date en los 365 días previos, entre el precio."""
    if not dividends:
        return pd.Series(np.nan, index=close.index)
    rows = []
    for item in dividends:
        ex = item.get("ex_date") or item.get("pago_date")
        pago = item.get("pago")
        if not ex or not pago:
            continue
        rows.append((pd.Timestamp(ex), float(pago)))
    if not rows:
        return pd.Series(np.nan, index=close.index)
    div = pd.Series({d: p for d, p in rows}).sort_index()
    div = div.groupby(level=0).sum()
    daily = div.reindex(pd.to_datetime(close.index).tz_localize(None)).fillna(0.0)
    # El índice de precios es de días hábiles: 252 sesiones ≈ un año.
    ttm = daily.rolling(252, min_periods=1).sum()
    return ttm / close.replace(0, np.nan)


def surprise_on_prices(close: pd.Series, events: pd.DataFrame) -> pd.Series:
    if events is None or events.empty:
        return pd.Series(np.nan, index=close.index)
    ev = events.dropna(subset=["surprise"]).copy()
    if ev.empty:
        return pd.Series(np.nan, index=close.index)
    ev["announced"] = pd.to_datetime(ev["announced"]).dt.tz_localize(None).dt.normalize()
    ev = ev.sort_values("announced").drop_duplicates("announced", keep="last")
    ev["available"] = ev["announced"] + pd.Timedelta(days=1)
    left = pd.DataFrame({"date": pd.to_datetime(close.index).tz_localize(None)})
    right = ev[["available", "surprise"]].rename(columns={"available": "date"})
    merged = pd.merge_asof(left.sort_values("date"), right.sort_values("date"), on="date", direction="backward")
    return pd.Series(merged["surprise"].to_numpy(), index=close.index)
