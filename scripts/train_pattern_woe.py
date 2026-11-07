#!/usr/bin/env python3
"""Entrena un WoE único (todas las acciones juntas) con patrones de 3, 20 y 60 días.

Blanco: el cierre sube en los próximos 20 días hábiles.
El ajuste usa el 80% de fechas más viejas. El 20% final solo se mira al calificar.

    python scripts/train_pattern_woe.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ml.pattern_features import PAT3_LABELS, PATTERN_FEATURES, pattern_frame
from src.ml.woe import compute_woe

MARKET = ROOT / "data" / "market"
OUT = ROOT / "data" / "woe_pattern.json"
HORIZON = 20
SKIP_NAMES = {"MXN=X", "CL=F", "GSPC"}


def _load_panel() -> pd.DataFrame:
    frames = []
    for path in sorted(MARKET.glob("*.parquet")):
        stem = path.name.replace(".parquet", "")
        if stem in SKIP_NAMES or "=" in stem:
            continue
        df = pd.read_parquet(path, columns=["Close"])
        if len(df) < 120:
            continue
        feat = pattern_frame(df["Close"], horizon=HORIZON)
        feat["ticker"] = stem
        feat["date"] = pd.to_datetime(feat.index).tz_localize(None)
        frames.append(feat.reset_index(drop=True))
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.dropna(subset=PATTERN_FEATURES + ["y"])
    return panel


def _cat_woe(x: pd.Series, y: pd.Series) -> dict:
    df = pd.DataFrame({"x": x.astype(int), "y": y.astype(int)}).dropna()
    n1 = int((df["y"] == 1).sum())
    n0 = int((df["y"] == 0).sum())
    k = df["x"].nunique()
    bins = []
    iv = 0.0
    for code, label in PAT3_LABELS.items():
        part = df[df["x"] == code]
        e1 = int((part["y"] == 1).sum())
        e0 = int((part["y"] == 0).sum())
        p1 = (e1 + 0.5) / (n1 + 0.5 * k)
        p0 = (e0 + 0.5) / (n0 + 0.5 * k)
        woe = math.log(p1 / p0)
        share1 = e1 / n1 if n1 else 0
        share0 = e0 / n0 if n0 else 0
        iv += (share1 - share0) * woe
        rate = e1 / len(part) if len(part) else None
        bins.append({
            "code": code,
            "label": label,
            "n": int(len(part)),
            "event_rate": None if rate is None else round(rate, 4),
            "woe": round(woe, 4),
        })
    return {"feature": "pat3", "iv": round(iv, 5), "bins": bins}


def _sigmoid(z: float) -> float:
    z = max(-20.0, min(20.0, z))
    return 1 / (1 + math.exp(-z))


def _lookup_cont(value: float, bins: list[dict]) -> float:
    for b in bins:
        if value >= b["lo"] and value < b["hi"]:
            return b["woe"]
    if not bins:
        return 0.0
    if value < bins[0]["lo"]:
        return bins[0]["woe"]
    return bins[-1]["woe"]


def main() -> None:
    panel = _load_panel()
    dates = np.sort(panel["date"].unique())
    cut = dates[int(len(dates) * 0.8)]
    train = panel[panel["date"] < cut]
    test = panel[panel["date"] >= cut]
    base = float(train["y"].mean())

    saved = []
    for feat in PATTERN_FEATURES:
        if feat == "pat3":
            saved.append(_cat_woe(train["pat3"], train["y"]))
            continue
        fw = compute_woe(train[feat], train["y"], n_bins=6)
        saved.append({
            "feature": feat,
            "iv": round(fw.iv, 5),
            "bins": [
                {
                    "label": b.bin_label,
                    "lo": b.lo,
                    "hi": b.hi,
                    "n": b.n,
                    "event_rate": round(b.event_rate, 4),
                    "woe": round(b.woe, 4),
                }
                for b in fw.bins
            ],
        })

    by_name = {row["feature"]: row for row in saved}
    used = [r["feature"] for r in saved if r["iv"] >= 0.01 and r["bins"]]
    logit_base = math.log(base / (1 - base))

    def score(df: pd.DataFrame) -> np.ndarray:
        woe_sum = np.zeros(len(df))
        for feat in used:
            spec = by_name[feat]
            if feat == "pat3":
                table = {b["code"]: b["woe"] for b in spec["bins"]}
                woe_sum += df["pat3"].astype(int).map(table).fillna(0).to_numpy()
            else:
                woe_sum += df[feat].map(lambda v, bins=spec["bins"]: _lookup_cont(float(v), bins)).to_numpy()
        return 1 / (1 + np.exp(-np.clip(logit_base + woe_sum, -20, 20)))

    p = score(test)
    y = test["y"].to_numpy()
    brier = float(np.mean((p - y) ** 2))
    brier_base = float(np.mean((base - y) ** 2))
    payload = {
        "horizon_days": HORIZON,
        "cutoff": str(pd.Timestamp(cut).date()),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_tickers": int(panel["ticker"].nunique()),
        "event_rate_train": round(base, 4),
        "features_in_score": used,
        "iv": [{k: r[k] for k in ("feature", "iv")} for r in sorted(saved, key=lambda r: -r["iv"])],
        "test": {
            "brier": round(brier, 4),
            "brier_tasa_base": round(brier_base, 4),
            "p_cuando_sube": round(float(p[y == 1].mean()), 4) if (y == 1).any() else None,
            "p_cuando_no": round(float(p[y == 0].mean()), 4) if (y == 0).any() else None,
        },
        "features": saved,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"tickers {payload['n_tickers']}  train {payload['n_train']}  test {payload['n_test']}  corte {payload['cutoff']}")
    print(f"tasa base train {base:.3f}")
    for row in payload["iv"]:
        print(f"  IV {row['iv']:.4f}  {row['feature']}")
    print(
        f"test  p|sube {payload['test']['p_cuando_sube']}  "
        f"p|no {payload['test']['p_cuando_no']}  "
        f"brier {brier:.4f}  (base {brier_base:.4f})"
    )
    print(f"guardado en {OUT}")


if __name__ == "__main__":
    main()
