"""Tres horizontes para un solo libro de 50 dólares.

Mañana y semana: reversión (se compran los que más cayeron).
Mes: momentum de 12 meses, saltando el último mes.
Si un nombre sale en dos horizontes, se queda en el más largo.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MARKET = ROOT / "data" / "market"
SKIP = {"MXN=X", "CL=F", "GSPC"}

# dólares del libro de 50
BUDGET = {"manana": 10.0, "semana": 15.0, "mes": 25.0}
SLOTS = {"manana": 1, "semana": 1, "mes": 2}
HOLD_DAYS = {"manana": 1, "semana": 5, "mes": 20}


def _symbols(sleeve: str) -> list[str]:
    out = []
    for path in sorted(MARKET.glob("*.parquet")):
        stem = path.name.replace(".parquet", "")
        if stem in SKIP or "=" in stem:
            continue
        if sleeve == "bmv" and not stem.endswith(".MX"):
            continue
        out.append(stem)
    return out


def _close(stem: str) -> pd.Series:
    series = pd.read_parquet(MARKET / f"{stem}.parquet", columns=["Close"])["Close"]
    series = pd.to_numeric(series, errors="coerce")
    series.index = pd.to_datetime(series.index).tz_localize(None)
    return series.sort_index()


def _score(series: pd.Series, asof: pd.Timestamp, kind: str) -> float | None:
    hist = series.loc[:asof].dropna()
    if kind == "manana":
        if len(hist) < 2 or hist.iloc[-2] <= 0:
            return None
        return -(hist.iloc[-1] / hist.iloc[-2] - 1)
    if kind == "semana":
        if len(hist) < 6 or hist.iloc[-6] <= 0:
            return None
        return -(hist.iloc[-1] / hist.iloc[-6] - 1)
    if len(hist) < 252 or hist.iloc[-252] <= 0 or hist.iloc[-21] <= 0:
        return None
    return hist.iloc[-21] / hist.iloc[-252] - 1


def rank_horizon(asof: pd.Timestamp, kind: str, sleeve: str = "todo", top_n: int = 5) -> list[dict]:
    """Mayor score = más atractivo. En corto, score alto = cayó más."""
    rows = []
    for stem in _symbols(sleeve):
        series = _close(stem)
        if asof not in series.index:
            continue
        score = _score(series, asof, kind)
        if score is None:
            continue
        rows.append({
            "ticker": stem.replace(".MX", ""),
            "symbol": stem,
            "score": round(float(score), 4),
            "precio": round(float(series.loc[asof]), 4),
        })
    rows.sort(key=lambda r: -r["score"])
    for i, row in enumerate(rows[:top_n], start=1):
        row["puesto"] = i
    return rows[:top_n]


def libro(asof: pd.Timestamp, sleeve: str = "todo") -> dict:
    """Un solo libro de 50: 10 mañana, 15 semana, 25 mes, sin repetir nombre."""
    lists = {kind: rank_horizon(asof, kind, sleeve=sleeve, top_n=8) for kind in HOLD_DAYS}
    used: set[str] = set()
    planes = {}
    for kind in ("mes", "semana", "manana"):
        picks = []
        budget = BUDGET[kind]
        slot = round(budget / SLOTS[kind], 2)
        for row in lists[kind]:
            if row["ticker"] in used:
                continue
            if len(picks) >= SLOTS[kind]:
                break
            used.add(row["ticker"])
            picks.append({**row, "usd": slot, "horizonte": kind, "dias": HOLD_DAYS[kind]})
        planes[kind] = picks
    return {
        "asof": str(asof.date()),
        "sleeve": sleeve,
        "listas": {k: lists[k][:5] for k in lists},
        "libro": planes,
    }
