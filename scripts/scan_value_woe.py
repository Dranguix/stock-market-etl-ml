#!/usr/bin/env python3
"""IV de rendimiento de ganancias, dividendo y sorpresa de EPS.

EPS: Yahoo Finance, Reported EPS, usable al día siguiente del anuncio.
Dividendo: data/dividends/dividends.json, por acción.

    python scripts/scan_value_woe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ml.value_features import dividend_yield_on_prices, surprise_on_prices, ttm_eps_on_prices
from src.ml.woe import compute_woe

MARKET = ROOT / "data" / "market"
DIVIDENDS = ROOT / "data" / "dividends" / "dividends.json"
EPS_CACHE = ROOT / "data" / "eps" / "reported_eps.json"
OUT = ROOT / "data" / "woe_value_iv.json"
SKIP = {"MXN=X", "CL=F", "GSPC"}
FEATURES = ["earnings_yield_ttm", "div_yield_ttm", "eps_surprise"]


def _symbol_to_yf(stem: str) -> str:
    return stem


def _load_eps_cache() -> dict:
    if not EPS_CACHE.exists():
        return {}
    return json.loads(EPS_CACHE.read_text())


def _fetch_eps(symbols: list[str], cache: dict) -> dict:
    import yfinance as yf

    EPS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    for i, sym in enumerate(symbols, start=1):
        if sym in cache:
            continue
        try:
            ed = yf.Ticker(sym).get_earnings_dates(limit=80)
        except Exception:
            cache[sym] = []
            continue
        rows = []
        if ed is not None and len(ed):
            for ts, row in ed.iterrows():
                reported = row.get("Reported EPS")
                estimate = row.get("EPS Estimate")
                if pd.isna(reported):
                    continue
                surprise = None
                if estimate is not None and not pd.isna(estimate) and abs(float(estimate)) > 1e-9:
                    surprise = (float(reported) - float(estimate)) / abs(float(estimate))
                ts = pd.Timestamp(ts)
                if ts.tzinfo is not None:
                    ts = ts.tz_convert("UTC").tz_localize(None)
                announced = ts.date().isoformat()
                rows.append({
                    "announced": announced,
                    "reported": float(reported),
                    "surprise": surprise,
                })
        cache[sym] = rows
        if i % 10 == 0:
            EPS_CACHE.write_text(json.dumps(cache))
            print(f"  eps {i}/{len(symbols)}", flush=True)
    EPS_CACHE.write_text(json.dumps(cache))
    return cache


def _events(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["announced", "reported", "surprise"])
    return pd.DataFrame(rows)


def main() -> None:
    dividends = json.loads(DIVIDENDS.read_text()) if DIVIDENDS.exists() else {}
    symbols = []
    for path in sorted(MARKET.glob("*.parquet")):
        stem = path.name.replace(".parquet", "")
        if stem in SKIP or "=" in stem:
            continue
        symbols.append(stem)
    cache = _fetch_eps(symbols, _load_eps_cache())

    frames = []
    for stem in symbols:
        path = MARKET / f"{stem}.parquet"
        px = pd.read_parquet(path, columns=["Close"])
        if len(px) < 260:
            continue
        close = px["Close"]
        short = stem.replace(".MX", "")
        divs = dividends.get(short) or dividends.get(stem) or []
        ev = _events(cache.get(stem) or [])
        feat = pd.DataFrame({
            "earnings_yield_ttm": ttm_eps_on_prices(close, ev),
            "div_yield_ttm": dividend_yield_on_prices(close, divs),
            "eps_surprise": surprise_on_prices(close, ev),
        }, index=close.index)
        fwd = close.shift(-20) / close - 1
        feat["y"] = (fwd > 0).astype(float).where(fwd.notna())
        feat["date"] = pd.to_datetime(feat.index).tz_localize(None)
        frames.append(feat.reset_index(drop=True))
    panel = pd.concat(frames, ignore_index=True)
    dates = np.sort(panel["date"].dropna().unique())
    cut = dates[int(len(dates) * 0.8)]
    train = panel[panel["date"] < cut]
    rows = []
    for feat in FEATURES:
        fw = compute_woe(train[feat], train["y"], n_bins=6)
        covered = int(train[feat].notna().sum())
        rows.append({
            "feature": feat,
            "iv": round(fw.iv, 5),
            "n_bins": fw.n_bins,
            "n_train_nonnull": covered,
        })
    rows.sort(key=lambda r: -r["iv"])
    payload = {
        "horizon_days": 20,
        "cutoff": str(pd.Timestamp(cut).date()),
        "n_train_rows": int(len(train)),
        "iv": rows,
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"corte {payload['cutoff']}  filas {payload['n_train_rows']}")
    for row in rows:
        print(f"  IV {row['iv']:.4f}  n {row['n_train_nonnull']}  {row['feature']}")
    print(f"con IV >= 0.02: {sum(r['iv'] >= 0.02 for r in rows)}")


if __name__ == "__main__":
    main()
