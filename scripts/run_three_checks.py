#!/usr/bin/env python3
"""Tres pruebas: plan mensual, momentum 12-1, y WoE del flujo de órdenes.

    python scripts/run_three_checks.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ml.woe import compute_woe

SKIP = {"MXN=X", "CL=F", "GSPC"}
COST = 0.0055  # ida y vuelta


def _symbols() -> list[str]:
    out = []
    for path in sorted((ROOT / "data" / "market").glob("*.parquet")):
        stem = path.name.replace(".parquet", "")
        if stem in SKIP or "=" in stem:
            continue
        out.append(stem)
    return out


def _close(stem: str) -> pd.Series:
    df = pd.read_parquet(ROOT / "data" / "market" / f"{stem}.parquet", columns=["Close"])
    s = pd.to_numeric(df["Close"], errors="coerce")
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s.sort_index()


def momentum_12_1() -> None:
    closes = {s: _close(s) for s in _symbols()}
    # Calendario del nombre con más historia reciente.
    base = max(closes.values(), key=len)
    month_ends = base.groupby(base.index.to_period("M")).tail(1).index
    month_ends = month_ends[month_ends >= pd.Timestamp("2015-01-01")]
    equity = 50.0
    equity_net = 50.0
    n = 0
    recent = 50.0
    recent_start = pd.Timestamp("2025-09-01")
    for i in range(len(month_ends) - 1):
        t0, t1 = month_ends[i], month_ends[i + 1]
        scores = []
        for stem, series in closes.items():
            if t0 not in series.index or t1 not in series.index:
                continue
            hist = series.loc[:t0]
            if len(hist) < 260:
                continue
            past = hist.iloc[-252]
            skip = hist.iloc[-21]
            if past <= 0 or skip <= 0:
                continue
            mom = skip / past - 1
            entry = series.loc[series.index > t0]
            if entry.empty:
                continue
            px0 = float(entry.iloc[0])
            px1 = float(series.loc[t1])
            if px0 <= 0:
                continue
            scores.append((mom, px1 / px0 - 1))
        if len(scores) < 5:
            continue
        scores.sort(key=lambda x: -x[0])
        ret = float(np.mean([r for _, r in scores[:5]]))
        equity *= 1 + ret
        equity_net *= 1 + ret - COST
        n += 1
        if t0 >= recent_start:
            recent *= 1 + ret
    print("MOMENTUM 12-1  top 5, un mes, salta el último mes")
    print(f"  meses {n}  bruto ${equity:.2f}  neto de 0.55% ${equity_net:.2f}  (desde $50)")
    print(f"  solo desde sep 2025, bruto ${recent:.2f}")


def flow_iv() -> None:
    from src.ml.flow_features import compute_flow_panel

    stems = _symbols()
    short = {s.replace(".MX", ""): s for s in stems}
    panel = compute_flow_panel(tickers=list(short), close_panel=None)
    if panel.empty:
        print("FLUJO  sin datos")
        return
    panel["date"] = pd.to_datetime(panel["date"]).dt.tz_localize(None)
    closes = {s: _close(s) for s in short.values()}
    rows = []
    for _, row in panel.iterrows():
        stem = short.get(row["ticker"])
        series = closes.get(stem) if stem else None
        if series is None or row["date"] not in series.index:
            continue
        i = series.index.get_loc(row["date"])
        if isinstance(i, slice) or i + 20 >= len(series):
            continue
        fwd = float(series.iloc[i + 20] / series.iloc[i] - 1)
        rows.append({
            "date": row["date"],
            "y": 1.0 if fwd > 0 else 0.0,
            "flow_imbalance": row["flow_imbalance"],
            "flow_concentration": row["flow_concentration"],
            "flow_persistence": row["flow_persistence"],
        })
    df = pd.DataFrame(rows)
    if df.empty:
        print("FLUJO  no cruzó con precios")
        return
    dates = np.sort(df["date"].unique())
    cut = dates[int(len(dates) * 0.8)]
    train = df[df["date"] < cut]
    print(f"FLUJO  filas {len(df)}  train {len(train)}  corte {pd.Timestamp(cut).date()}")
    for feat in ("flow_imbalance", "flow_concentration", "flow_persistence"):
        fw = compute_woe(train[feat], train["y"], n_bins=6)
        print(f"  IV {fw.iv:.4f}  {feat}")


def monthly_claude() -> None:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    os.environ.pop("LAB_DATABASE_URL", None)
    os.environ.pop("DATABASE_URL", None)
    import httpx
    from scripts import gcp_daily_job as job

    anchor = _close("AAPL")
    bdays = list(anchor.index)
    bdays = [d for d in bdays if d >= pd.Timestamp("2025-09-01") and d <= pd.Timestamp("2026-09-12")]
    signals = bdays[::20]

    def plan(cands, enter_on):
        context = {"enter_on": str(enter_on.date()), "candidatos": [
            {"ticker": c["ticker"], "stop": c["stop"], "p_sube": c["p_sube"], "entry_px": c["entry_px"]}
            for c in cands[:8]
        ]}
        user = (
            "Reparte 50 dólares para un mes. Long only. Máximo 3 nombres, entre 10 y 25. "
            "Responde solo JSON {\"allocations\":[{\"ticker\",\"usd\"}],\"cash_usd\"}.\n"
            + json.dumps(context, default=job._json_default)
        )
        r = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": os.environ.get("ANTHROPIC_MODEL", "claude-haiku-5-5"), "max_tokens": 1500,
                  "messages": [{"role": "user", "content": user}]},
            timeout=90,
        )
        body = r.json()
        text = "".join(b.get("text", "") for b in body.get("content") or [] if b.get("type") == "text")
        return job._parse_json_obj(text)

    equity = 50.0
    print("PLAN MENSUAL  Claude, se mantiene 20 días hábiles")
    for ts in signals:
        signal_day = ts.date()
        cands = job._candidates(signal_day)
        if not cands:
            continue
        job._attach_p_sube(cands)
        by = {c["ticker"]: c for c in cands}
        try:
            proposal = plan(cands, ts + pd.Timedelta(days=1))
        except Exception as exc:
            print(f"  {signal_day} fallo {exc}")
            continue
        pnl = 0.0
        names = []
        scale = equity / 50.0
        for a in (proposal.get("allocations") or [])[:3]:
            t = str(a.get("ticker", "")).upper()
            c = by.get(t)
            usd = float(a.get("usd") or 0) * scale
            if c is None or usd < 10:
                continue
            series = _close(c["symbol"])
            if ts not in series.index:
                continue
            i = series.index.get_loc(ts)
            if i + 20 >= len(series):
                continue
            px0 = float(series.iloc[i + 1])
            px1 = float(series.iloc[i + 20])
            if px0 <= 0:
                continue
            part = usd * (px1 / px0 - 1)
            pnl += part
            names.append(f"{t}{part:+.2f}")
        if not names:
            print(f"  {signal_day} sin cruce")
            continue
        equity += pnl
        print(f"  {signal_day}  {pnl:+.2f}  libro {equity:.2f}  {', '.join(names)}")
    print(f"  final ${equity:.2f}")


if __name__ == "__main__":
    momentum_12_1()
    print("---")
    flow_iv()
    print("---")
    monthly_claude()
