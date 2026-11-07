"""WoE de hoy para un ticker, usando data/woe_results.json y src/ml/woe.py.

La probabilidad es el sigmoide del scorecard:
    σ( logit(tasa_base) + Σ WoE_i )
solo con variables de IV >= 0.02, que es el piso de 'señal débil' en woe.py.

Si aparece data/woe_lr.json con intercept y coefficients, esos beta sustituyen
la suma simple. Ese archivo lo puede dejar el proceso que está completando datos.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.ml.features import add_macro_features, compute_features

ROOT = Path(__file__).resolve().parents[2]
WOE_PATH = ROOT / "data" / "woe_results.json"
LR_PATH = ROOT / "data" / "woe_lr.json"
MARKET = ROOT / "data" / "market"
IV_FLOOR = 0.02

MACRO_FILES = {
    "USDMXN": "MXN=X.parquet",
    "NAFTRAC": "NAFTRACISHRS.MX.parquet",
    "WTI": "CL=F.parquet",
    "SP500": "GSPC.parquet",
}


def sigmoid(x: float) -> float:
    x = max(-20.0, min(20.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def scorecard_probability(woes: list[float], base_rate: float) -> float:
    base = min(max(base_rate, 1e-4), 1 - 1e-4)
    logit = math.log(base / (1 - base)) + sum(woes)
    return sigmoid(logit)


def bin_of(value: float, bins: list[dict]) -> dict | None:
    if not bins or value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    for b in bins:
        if value >= b["lo"] and value < b["hi"]:
            return b
    if value < bins[0]["lo"]:
        return bins[0]
    return bins[-1]


@lru_cache(maxsize=1)
def load_woe() -> dict:
    if not WOE_PATH.exists():
        return {}
    return json.loads(WOE_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_lr() -> dict | None:
    if not LR_PATH.exists():
        return None
    return json.loads(LR_PATH.read_text(encoding="utf-8"))


def _read_ohlcv(symbol: str) -> pd.DataFrame | None:
    path = MARKET / f"{symbol}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    need = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
    if "Close" not in need:
        return None
    return df[need].dropna(subset=["Close"])


def _macro() -> dict[str, pd.DataFrame]:
    out = {}
    for key, fname in MACRO_FILES.items():
        path = MARKET / fname
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if "Close" in df.columns:
            out[key] = df
    return out


def latest_features(symbol: str) -> tuple[pd.Series | None, str]:
    raw = _read_ohlcv(symbol)
    if raw is None or len(raw) < 60:
        return None, f"sin parquet usable en data/market/{symbol}.parquet"
    feat = compute_features(raw)
    macro = _macro()
    if macro:
        feat = add_macro_features(feat, macro)
    return feat.iloc[-1], "ok"


def explain_ticker(symbol: str) -> dict:
    """Tabla de bins de hoy + probabilidad del scorecard."""
    table = load_woe()
    if not table:
        return {"ok": False, "error": "falta data/woe_results.json"}

    meta = table.get("meta") or {}
    base = float(meta.get("event_rate") or 0.5)
    iv_rows = {r["feature"]: r for r in table.get("iv_summary") or []}
    bins_map = table.get("bins_by_feature") or {}

    row, status = latest_features(symbol)
    if row is None:
        return {"ok": False, "error": status, "symbol": symbol}

    fired = []
    woes_for_p = []
    for feat, info in iv_rows.items():
        if feat not in row.index or feat not in bins_map:
            continue
        val = row[feat]
        try:
            val_f = float(val)
        except (TypeError, ValueError):
            continue
        if math.isnan(val_f):
            continue
        b = bin_of(val_f, bins_map[feat])
        if b is None:
            continue
        enters = float(info["iv"]) >= IV_FLOOR
        item = {
            "feature": feat,
            "valor": round(val_f, 4),
            "bin": b["label"],
            "woe": b["woe"],
            "iv": info["iv"],
            "fuerza": info.get("strength"),
            "entra_en_probabilidad": enters,
        }
        fired.append(item)
        if enters:
            woes_for_p.append(float(b["woe"]))

    fired.sort(key=lambda x: abs(x["woe"]) * max(x["iv"], 1e-6), reverse=True)

    lr = load_lr()
    if lr and lr.get("coefficients"):
        z = float(lr.get("intercept") or 0)
        for item in fired:
            z += float(lr["coefficients"].get(item["feature"], 0)) * item["woe"]
        prob = sigmoid(z)
        fuente = "woe_lr.json"
    else:
        prob = scorecard_probability(woes_for_p, base)
        fuente = "scorecard: sigmoide(logit(tasa base) + suma de WoE con IV>=0.02)"

    atr_pct = row.get("atr_pct")
    close = float(row["Close"]) if "Close" in row.index else None
    atr = None
    if close and atr_pct == atr_pct:
        atr = close * float(atr_pct) / 100.0

    return {
        "ok": True,
        "symbol": symbol,
        "p_sube_5d": round(prob, 4),
        "tasa_base": base,
        "fuente_probabilidad": fuente,
        "n_variables_en_probabilidad": len(woes_for_p),
        "tabla": fired[:12],
        "close": None if close is None else round(close, 4),
        "atr": None if atr is None else round(atr, 4),
    }
