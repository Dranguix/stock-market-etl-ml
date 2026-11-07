"""Patrones de largo plazo por ticker, para un WoE único de todo el mercado.

La frase completa solo se guarda a 3 días (8 combinaciones).
En 20 y 60 días se guardan resúmenes: días verdes, rachas, vueltas y
la caída desde el máximo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PAT3_LABELS = {
    0: "---",
    1: "--+",
    2: "-+-",
    3: "-++",
    4: "+--",
    5: "+-+",
    6: "++-",
    7: "+++",
}

PATTERN_FEATURES = [
    "up_days_20",
    "up_days_60",
    "racha",
    "racha_max_60",
    "drawdown_60",
    "vueltas_20",
    "giro_5_vs_60",
    "pat3",
]


def _current_streak(up: np.ndarray) -> np.ndarray:
    out = np.full(len(up), np.nan)
    streak = 0
    for i, v in enumerate(up):
        if np.isnan(v):
            streak = 0
            continue
        if v == 1:
            streak = streak + 1 if streak > 0 else 1
        else:
            streak = streak - 1 if streak < 0 else -1
        out[i] = streak
    return out


def _longest_up_streak(up: np.ndarray, window: int) -> np.ndarray:
    run = np.zeros(len(up))
    current = 0
    for i, v in enumerate(up):
        if np.isnan(v) or v == 0:
            current = 0
        else:
            current += 1
        run[i] = current
    return pd.Series(run).rolling(window, min_periods=window).max().to_numpy()


def pattern_frame(close: pd.Series, horizon: int = 20) -> pd.DataFrame:
    """Features de patrón y el blanco: 1 si el cierre sube en `horizon` días."""
    c = pd.to_numeric(close, errors="coerce")
    ret = c.pct_change()
    up = (ret > 0).astype(float)
    up = up.where(ret.notna())
    up_np = up.to_numpy(dtype=float)

    bits = up.fillna(0)
    pat3 = bits.shift(2) * 4 + bits.shift(1) * 2 + bits
    pat3 = pat3.where(up.notna() & up.shift(1).notna() & up.shift(2).notna())

    sign = np.sign(ret)
    flip = (sign != sign.shift(1)) & sign.notna() & sign.shift(1).notna() & (sign != 0) & (sign.shift(1) != 0)

    ret5 = c.pct_change(5)
    ret60 = c.pct_change(60)
    out = pd.DataFrame({
        "up_days_20": up.rolling(20, min_periods=20).sum(),
        "up_days_60": up.rolling(60, min_periods=60).sum(),
        "racha": _current_streak(up_np),
        "racha_max_60": _longest_up_streak(up_np, 60),
        "drawdown_60": c / c.rolling(60, min_periods=60).max() - 1,
        "vueltas_20": flip.astype(float).rolling(20, min_periods=20).sum(),
        "giro_5_vs_60": ret5 - ret60,
        "pat3": pat3,
    }, index=c.index)
    fwd = c.shift(-horizon) / c - 1
    out["y"] = (fwd > 0).astype(float).where(fwd.notna())
    return out
