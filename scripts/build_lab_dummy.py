#!/usr/bin/env python3
"""Datos dummy del esquema lab, para probar el dominio Lab/Trading sin Postgres.

Escribe data/lab/dummy/<tabla>.jsonl con las columnas exactas de
sql/03_lab_schema.sql: decision, entry_plan, outcome y price_daily.
No son decisiones reales.

Cuatro cierres de mes (jul-oct 2026) con la regla núcleo-satélite:
  - jul: sin cambios, satélite ALSEA + FUNO11.
  - ago: NAFTRAC debajo de su tendencia, el núcleo se queda a la mitad.
  - sep: reselección trimestral. Haiku veta KIMBERA y entra FIBRAPL;
    el gemelo (la regla sin veto) se queda con KIMBERA.
  - oct: sin llave de Anthropic, la regla corre sola. Sin calificar todavía.

Los precios son caminatas aleatorias con semilla fija. outcome se calcula como
run_score de scripts/gcp_daily_job.py: cierre de entrada a cierre de salida en
USD, y el stop cuenta si el mínimo del día lo toca.

    python scripts/build_lab_dummy.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.core_satellite import CATASTROPHE_DD, CORE_BEAR_CUT, config  # noqa: E402

OUT = ROOT / "data" / "lab" / "dummy"
BUDGET_USD = 50.0
HORIZON = "20d"
HORIZON_DAYS = 20
AS_OF = date(2026, 11, 6)
FX = "USDMXN"

# ticker: (moneda, precio inicial, volatilidad diaria, deriva diaria por mes)
SERIES = {
    "SPY": ("USD", 640.0, 0.008, {8: 0.0008, 9: 0.0006, 10: 0.0009, 11: 0.0005}),
    "NAFTRAC": ("MXN", 62.0, 0.009, {8: -0.0012, 9: 0.0010, 10: 0.0007, 11: 0.0004}),
    "ALSEA": ("MXN", 50.0, 0.016, {8: 0.0015, 9: 0.0010, 10: 0.0012, 11: 0.0008}),
    "FUNO11": ("MXN", 33.0, 0.012, {8: 0.0005, 9: -0.0110, 10: 0.0002, 11: 0.0002}),
    "KIMBERA": ("MXN", 36.0, 0.013, {8: 0.0004, 9: 0.0002, 10: -0.0045, 11: 0.0}),
    "FIBRAPL": ("MXN", 83.0, 0.011, {8: 0.0006, 9: 0.0005, 10: 0.0015, 11: 0.0006}),
    FX: ("MXN", 18.60, 0.004, {8: 0.0001, 9: -0.0002, 10: 0.0001, 11: 0.0}),
}

# session_date: (pesos de la regla con veto, pesos del gemelo, vetos, candidatos, tendencia, llm)
DECISIONES = [
    (
        date(2026, 7, 31),
        {"SPY": 0.5, "NAFTRAC": 0.25, "ALSEA": 0.125, "FUNO11": 0.125},
        None,
        {},
        [],
        {"SPY": True, "NAFTRAC": True, "ALSEA": True, "FUNO11": True},
        True,
    ),
    (
        date(2026, 8, 31),
        {"SPY": 0.5, "NAFTRAC": 0.25 * CORE_BEAR_CUT, "ALSEA": 0.125, "FUNO11": 0.125},
        None,
        {},
        [],
        {"SPY": True, "NAFTRAC": False, "ALSEA": True, "FUNO11": True},
        True,
    ),
    (
        date(2026, 9, 30),
        {"SPY": 0.5, "NAFTRAC": 0.25, "ALSEA": 0.125, "FIBRAPL": 0.125},
        {"SPY": 0.5, "NAFTRAC": 0.25, "ALSEA": 0.125, "KIMBERA": 0.125},
        {"KIMBERA": "recorte_distribucion: titulares sobre un recorte del dividendo trimestral"},
        ["KIMBERA", "FIBRAPL"],
        {"SPY": True, "NAFTRAC": True, "ALSEA": True, "KIMBERA": True, "FIBRAPL": True},
        True,
    ),
    (
        date(2026, 10, 30),
        {"SPY": 0.5, "NAFTRAC": 0.25, "ALSEA": 0.125, "FIBRAPL": 0.125},
        None,
        {},
        [],
        {"SPY": True, "NAFTRAC": True, "ALSEA": True, "FIBRAPL": True},
        False,
    ),
]


def _bdays(start: date, end: date) -> list[date]:
    out, day = [], start
    while day <= end:
        if day.weekday() < 5:
            out.append(day)
        day += timedelta(days=1)
    return out


def _add_bdays(day: date, n: int) -> date:
    left = n
    while left > 0:
        day += timedelta(days=1)
        if day.weekday() < 5:
            left -= 1
    return day


def _ts(day: date, hhmm: str) -> str:
    return f"{day.isoformat()}T{hhmm}:00-06:00"


def build_prices(days: list[date]) -> dict[str, dict[date, dict]]:
    bars: dict[str, dict[date, dict]] = {}
    for offset, (ticker, (moneda, start, vol, drift)) in enumerate(SERIES.items()):
        rng = np.random.default_rng(20260731 + offset)
        close = start
        bars[ticker] = {}
        for day in days:
            prev = close
            close = prev * math.exp(drift.get(day.month, 0.0) + vol * rng.standard_normal())
            open_ = prev * (1 + vol * 0.3 * rng.standard_normal())
            high = max(open_, close) * (1 + abs(rng.standard_normal()) * vol * 0.5)
            low = min(open_, close) * (1 - abs(rng.standard_normal()) * vol * 0.5)
            bars[ticker][day] = {
                "ticker": ticker,
                "trade_date": day.isoformat(),
                "open": round(open_, 4),
                "high": round(high, 4),
                "low": round(low, 4),
                "close": round(close, 4),
                "volume": float(int(rng.integers(200_000, 5_000_000))),
                "moneda": moneda,
                "source": "dummy",
                "loaded_at": _ts(day, "18:30"),
            }
    return bars


def _usd(bars: dict, ticker: str, day: date, col: str) -> float:
    px = bars[ticker][day][col]
    return px / bars[FX][day]["close"] if SERIES[ticker][0] == "MXN" else px


def _motivo(weights: dict, satelite: list[str], vetos: dict, reselect: bool) -> str:
    nucleo = ", ".join(f"{k} {weights.get(k, 0) * 100:.0f}%" for k in ("SPY", "NAFTRAC"))
    return (
        f"núcleo {nucleo}; satélite {', '.join(satelite) or 'efectivo'}"
        + (f"; vetados {', '.join(vetos)}" if vetos else "")
        + ("; reselección trimestral" if reselect else "; solo tendencia y deriva")
    )


def build(as_of: date = AS_OF) -> dict[str, list[dict]]:
    days = _bdays(date(2026, 7, 30), as_of)
    bars = build_prices(days)
    decisiones, planes, resultados = [], [], []
    for d_id, (session, weights, gemelo, vetos, candidatos, tendencia, llm) in enumerate(DECISIONES, start=1):
        enter_on = _add_bdays(session, 1)
        satelite = [t for t in weights if t not in ("SPY", "NAFTRAC")]
        plan_rows = []
        for ticker, weight in weights.items():
            stop = None
            if ticker in satelite:
                stop = round(_usd(bars, ticker, session, "close") * (1 - CATASTROPHE_DD), 4)
            plan_rows.append({
                "ticker": ticker,
                "usd": round(weight * BUDGET_USD, 2),
                "horizon": HORIZON,
                "stop_usd": stop,
                "p_sube": None,
                "enter_on": enter_on.isoformat(),
            })
        cash = round(BUDGET_USD - sum(r["usd"] for r in plan_rows), 2)
        decisiones.append({
            "id": d_id,
            "decided_at": _ts(session, "19:15"),
            "session_date": session.isoformat(),
            "model": "claude-haiku-5-5" if llm else "rules",
            "cash_usd": cash,
            "motivo": _motivo(weights, satelite, vetos, bool(candidatos)),
            "raw": {
                "llm_used": llm,
                "enter_on": enter_on.isoformat(),
                "book": None,
                "plans": plan_rows,
                "regla": config(),
                "gemelo": gemelo or weights,
                "vetos": vetos,
                "candidatos": candidatos,
                "tendencia": tendencia,
            },
        })
        for row in plan_rows:
            p_id = len(planes) + 1
            planes.append({"id": p_id, "decision_id": d_id, **row})
            exit_on = _add_bdays(enter_on, HORIZON_DAYS)
            if exit_on > as_of:
                continue
            window = [d for d in days if enter_on <= d <= exit_on]
            ticker = row["ticker"]
            entry = round(_usd(bars, ticker, window[0], "close"), 4)
            exit_px = round(_usd(bars, ticker, window[-1], "close"), 4)
            low = min(_usd(bars, ticker, d, "low") for d in window)
            stop = row["stop_usd"]
            hit = bool(stop is not None and low <= stop)
            ret = ((stop if hit else exit_px) - entry) / entry
            pnl = round(row["usd"] * ret, 4)
            resultados.append({
                "id": len(resultados) + 1,
                "entry_plan_id": p_id,
                "scored_at": _ts(exit_on, "20:00"),
                "realized_return": round(ret, 6),
                "hit_stop": hit,
                "note": f"entrada={entry} close={exit_px} stop={stop} pnl_usd={pnl} exit_on={exit_on.isoformat()}",
            })
    precios = [bars[t][d] for t in SERIES for d in days]
    return {"decision": decisiones, "entry_plan": planes, "outcome": resultados, "price_daily": precios}


def write(tablas: dict[str, list[dict]], out: Path = OUT) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for tabla, rows in tablas.items():
        text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        (out / f"{tabla}.jsonl").write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    tablas = build()
    write(tablas, args.out)
    for tabla, rows in tablas.items():
        print(f"{tabla}: {len(rows)} filas")


if __name__ == "__main__":
    main()
