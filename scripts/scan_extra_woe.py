#!/usr/bin/env python3
"""Mide Information Value de las 50 variables extra. Blanco: sube en 20 días.

    python scripts/scan_extra_woe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ml.extra_features import EXTRA_FEATURES, extra_frame
from src.ml.woe import compute_woe

MARKET = ROOT / "data" / "market"
OUT = ROOT / "data" / "woe_extra_iv.json"
SKIP = {"MXN=X", "CL=F", "GSPC"}


def main() -> None:
    frames = []
    for path in sorted(MARKET.glob("*.parquet")):
        stem = path.name.replace(".parquet", "")
        if stem in SKIP or "=" in stem:
            continue
        raw = pd.read_parquet(path)
        if len(raw) < 280 or "Close" not in raw.columns:
            continue
        feat = extra_frame(raw, horizon=20)
        feat["date"] = pd.to_datetime(feat.index).tz_localize(None)
        frames.append(feat.reset_index(drop=True))
    panel = pd.concat(frames, ignore_index=True)
    dates = np.sort(panel["date"].dropna().unique())
    cut = dates[int(len(dates) * 0.8)]
    train = panel[panel["date"] < cut]
    rows = []
    for feat in EXTRA_FEATURES:
        fw = compute_woe(train[feat], train["y"], n_bins=6)
        rows.append({"feature": feat, "iv": round(fw.iv, 5), "n_bins": fw.n_bins})
    rows.sort(key=lambda r: -r["iv"])
    payload = {
        "horizon_days": 20,
        "cutoff": str(pd.Timestamp(cut).date()),
        "n_train": int(len(train)),
        "n_features": len(rows),
        "iv": rows,
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"train {payload['n_train']}  corte {payload['cutoff']}  variables {len(rows)}")
    print("top 15:")
    for row in rows[:15]:
        print(f"  {row['iv']:.4f}  {row['feature']}")
    fuertes = [r for r in rows if r["iv"] >= 0.02]
    print(f"con IV >= 0.02: {len(fuertes)}")


if __name__ == "__main__":
    main()
