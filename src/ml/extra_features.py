"""Cincuenta variables extra, todas a partir de OHLCV, para un WoE único.

El blanco lo pone quien entrena: aquí solo está el cálculo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EXTRA_FEATURES = [
    "ret_1", "ret_3", "ret_10", "ret_40", "ret_120", "ret_250",
    "accel_5_20", "accel_20_60", "accel_60_120",
    "vol_10", "vol_20", "vol_60", "vol_ratio_10_60", "vol_of_vol_20",
    "skew_20", "kurt_20",
    "hl_mean_10", "hl_mean_20", "range_compress",
    "gap", "close_loc", "body",
    "overnight_5", "intraday_5",
    "dist_sma10", "dist_sma100", "dist_sma200",
    "dist_low_120", "dist_high_120",
    "dd_20", "dd_120",
    "days_since_high_60", "days_since_low_60",
    "up_days_10", "up_days_120",
    "autocorr_20",
    "max_ret_20", "min_ret_20", "ret_pct_60",
    "volume_z_20", "volume_trend_20", "dollar_vol_z_20",
    "higher_high_10", "lower_low_10",
    "sign_agree_3",
    "efficiency_20", "efficiency_60",
    "tail_ratio_20",
    "pos_in_20", "bounce_5",
]


def _days_since(arr: np.ndarray, window: int, high: bool) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    for i in range(window - 1, len(arr)):
        w = arr[i - window + 1 : i + 1]
        if np.all(np.isnan(w)):
            continue
        j = int(np.nanargmax(w) if high else np.nanargmin(w))
        out[i] = window - 1 - j
    return out


def extra_frame(df: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    c = pd.to_numeric(df["Close"], errors="coerce")
    h = pd.to_numeric(df["High"], errors="coerce") if "High" in df.columns else c
    l = pd.to_numeric(df["Low"], errors="coerce") if "Low" in df.columns else c
    o = pd.to_numeric(df["Open"], errors="coerce") if "Open" in df.columns else c
    v = pd.to_numeric(df["Volume"], errors="coerce") if "Volume" in df.columns else pd.Series(0.0, index=c.index)
    ret = c.pct_change()
    up = (ret > 0).astype(float).where(ret.notna())
    span = (h - l).replace(0, np.nan)

    ret5, ret20 = c.pct_change(5), c.pct_change(20)
    ret60, ret120 = c.pct_change(60), c.pct_change(120)
    hl = (h - l) / c.replace(0, np.nan)
    hl10, hl20 = hl.rolling(10).mean(), hl.rolling(20).mean()
    vol10, vol60 = ret.rolling(10).std(), ret.rolling(60).std()
    vol20 = ret.rolling(20).std()

    worst = ret.rolling(20).apply(lambda x: np.mean(np.sort(x)[:3]), raw=True)
    best = ret.rolling(20).apply(lambda x: np.mean(np.sort(x)[-3:]), raw=True)

    out = pd.DataFrame({
        "ret_1": ret,
        "ret_3": c.pct_change(3),
        "ret_10": c.pct_change(10),
        "ret_40": c.pct_change(40),
        "ret_120": ret120,
        "ret_250": c.pct_change(250),
        "accel_5_20": ret5 - ret20,
        "accel_20_60": ret20 - ret60,
        "accel_60_120": ret60 - ret120,
        "vol_10": vol10,
        "vol_20": vol20,
        "vol_60": vol60,
        "vol_ratio_10_60": vol10 / vol60.replace(0, np.nan),
        "vol_of_vol_20": vol20.rolling(20).std(),
        "skew_20": ret.rolling(20).skew(),
        "kurt_20": ret.rolling(20).kurt(),
        "hl_mean_10": hl10,
        "hl_mean_20": hl20,
        "range_compress": hl10 / hl20.replace(0, np.nan),
        "gap": o / c.shift(1) - 1,
        "close_loc": (c - l) / span,
        "body": (c - o) / c.replace(0, np.nan),
        "overnight_5": (o / c.shift(1) - 1).rolling(5).mean(),
        "intraday_5": (c / o.replace(0, np.nan) - 1).rolling(5).mean(),
        "dist_sma10": c / c.rolling(10).mean() - 1,
        "dist_sma100": c / c.rolling(100).mean() - 1,
        "dist_sma200": c / c.rolling(200).mean() - 1,
        "dist_low_120": c / l.rolling(120).min() - 1,
        "dist_high_120": c / h.rolling(120).max() - 1,
        "dd_20": c / c.rolling(20).max() - 1,
        "dd_120": c / c.rolling(120).max() - 1,
        "days_since_high_60": _days_since(c.to_numpy(dtype=float), 60, True),
        "days_since_low_60": _days_since(c.to_numpy(dtype=float), 60, False),
        "up_days_10": up.rolling(10).sum(),
        "up_days_120": up.rolling(120).sum(),
        "autocorr_20": ret.rolling(20).corr(ret.shift(1)),
        "max_ret_20": ret.rolling(20).max(),
        "min_ret_20": ret.rolling(20).min(),
        "ret_pct_60": ret.rolling(60).rank(pct=True),
        "volume_z_20": (v - v.rolling(20).mean()) / v.rolling(20).std().replace(0, np.nan),
        "volume_trend_20": v / v.shift(20).replace(0, np.nan) - 1,
        "dollar_vol_z_20": (
            (c * v) - (c * v).rolling(20).mean()
        ) / (c * v).rolling(20).std().replace(0, np.nan),
        "higher_high_10": (h > h.shift(1)).astype(float).rolling(10).sum(),
        "lower_low_10": (l < l.shift(1)).astype(float).rolling(10).sum(),
        "sign_agree_3": (
            (np.sign(ret) == np.sign(ret20)).astype(float)
            + (np.sign(ret.shift(1)) == np.sign(ret20)).astype(float)
            + (np.sign(ret.shift(2)) == np.sign(ret20)).astype(float)
        ),
        "efficiency_20": ret20 / ret.abs().rolling(20).sum().replace(0, np.nan),
        "efficiency_60": ret60 / ret.abs().rolling(60).sum().replace(0, np.nan),
        "tail_ratio_20": worst / best.replace(0, np.nan),
        "pos_in_20": (c - l.rolling(20).min()) / (h.rolling(20).max() - l.rolling(20).min()).replace(0, np.nan),
        "bounce_5": ret5 - ret.rolling(20).min(),
    }, index=c.index)
    if len(out.columns) != 50:
        raise RuntimeError(f"se esperaban 50 variables, hay {len(out.columns)}")
    fwd = c.shift(-horizon) / c - 1
    out["y"] = (fwd > 0).astype(float).where(fwd.notna())
    return out
