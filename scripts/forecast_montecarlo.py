"""
Pronóstico a 30/60/90 días por simulación Montecarlo (abanico / fan chart).

A diferencia de `run_sin_ancla` en `forecast_horizons.py` (que compone UNA
sola trayectoria determinista con la predicción del modelo, y por eso sale
lisa), aquí se simulan MUCHOS caminos: cada día se le agrega ruido aleatorio
~N(0, vol_daily_pct) a la ganancia esperada del modelo, así que cada camino
es distinto. El modelo se ajusta UNA sola vez en el corte (mismo patrón que
`run_sin_ancla`) y luego se usa para todos los caminos.

Uso:
    python scripts/forecast_montecarlo.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.forecast_horizons import (
    TICKERS, HORIZONS, FIT_ROLLING, fit_regimen, apply_fitted,
    COLOR_REAL, COLOR_FORECAST, OUT_DIR, VOL_WINDOW,
)
from src.data.store import get as store_get
from src.ml.features import compute_features

MIN_TRAIN_ROWS = 300


def run_montecarlo(ticker_name: str, model_name: str, horizon_label: str,
                    horizon: int, n_paths: int = 60, seed: int = 0) -> dict | None:
    """Ajusta el modelo una vez en el corte y simula `n_paths` caminos de
    precio agregando ruido ~N(0, vol_daily_pct) a la ganancia predicha cada
    día — en vez de una sola línea lisa, un abanico de trayectorias."""
    symbol = TICKERS[ticker_name]
    df = store_get(symbol)
    if df is None or len(df) < MIN_TRAIN_ROWS + horizon + 10:
        return None

    feat = compute_features(df)
    cutoff_idx = len(feat) - horizon - 1
    if cutoff_idx < MIN_TRAIN_ROWS:
        return None
    cutoff_date = feat.index[cutoff_idx]
    price0 = float(df["Close"].iloc[cutoff_idx])

    daily_log_ret = np.log(df["Close"] / df["Close"].shift(1))
    vol_daily_pct = float(
        daily_log_ret.iloc[cutoff_idx - VOL_WINDOW:cutoff_idx].std() * 100
    )
    if not np.isfinite(vol_daily_pct) or vol_daily_pct <= 0:
        return None

    try:
        if model_name == "regimen_hmm":
            state = fit_regimen(df, feat, cutoff_idx)
        else:
            state = FIT_ROLLING[model_name](feat, cutoff_idx)
    except Exception as exc:
        print(f"  [montecarlo {model_name}] {ticker_name} {horizon_label}: ERROR ajuste {exc}")
        return None
    if state is None:
        return None

    # Ganancia predicha por el modelo para cada día del horizonte (misma para
    # todos los caminos; lo que cambia entre caminos es el ruido agregado).
    gains_pred = []
    for idx in range(cutoff_idx, cutoff_idx + horizon):
        try:
            g = apply_fitted(state, feat, idx)
        except Exception:
            g = 0.0
        if not np.isfinite(g):
            g = 0.0
        gains_pred.append(g)

    n_days = len(gains_pred)
    if n_days == 0:
        return None

    rng = np.random.default_rng(seed)
    paths = np.empty((n_paths, n_days))
    for p in range(n_paths):
        price = price0
        for d in range(n_days):
            noise = rng.normal(0, vol_daily_pct)
            price *= (1 + (gains_pred[d] + noise) / 100.0)
            paths[p, d] = price

    dates = feat.index[cutoff_idx + 1: cutoff_idx + 1 + n_days]
    real = df["Close"].iloc[cutoff_idx + 1: cutoff_idx + 1 + n_days].reset_index()
    real.columns = ["date", "close"]

    return {
        "ticker": ticker_name, "model": model_name, "horizon": horizon_label,
        "cutoff_date": cutoff_date, "dates": dates, "paths": paths, "real": real,
    }


def plot_montecarlo(results: list[dict], ticker: str, model: str) -> Path | None:
    """1x3 (30d/60d/90d): abanico de caminos simulados en azul transparente,
    mediana en azul sólido grueso, real en rojo grueso."""
    rows = [r for r in results if r["ticker"] == ticker and r["model"] == model]
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: list(HORIZONS).index(r["horizon"]))

    fig, axes = plt.subplots(1, len(rows), figsize=(6 * len(rows), 4.2))
    axes = np.atleast_1d(axes)

    for ax, r in zip(axes, rows):
        dates = r["dates"]
        paths = r["paths"]
        real = r["real"]

        for p in range(paths.shape[0]):
            ax.plot(dates, paths[p], color=COLOR_FORECAST, lw=0.8, alpha=0.1)

        mediana = np.median(paths, axis=0)
        ax.plot(dates, mediana, color=COLOR_FORECAST, lw=2.5, label="Mediana simulada")
        ax.plot(real["date"], real["close"], color=COLOR_REAL, lw=2.5, label="Real")

        ax.set_title(f"{r['horizon']} · corte {r['cutoff_date'].date()}", fontsize=11)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)

    axes[0].legend(fontsize=9, frameon=False)
    fig.suptitle(f"{ticker} — intento: {model} (Montecarlo, {rows[0]['paths'].shape[0]} caminos)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"montecarlo_{model}_{ticker}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main() -> None:
    ticker = "SPY"
    models = ["lightgbm_ranking", "momentum_12_1", "woe_score", "regimen_hmm"]

    print("=== Modo Montecarlo (abanico de caminos simulados) ===")
    results = []
    for model in models:
        for h_label, h_days in HORIZONS.items():
            print(f"[montecarlo {model}] {ticker} {h_label} ({h_days}d)...")
            r = run_montecarlo(ticker, model, h_label, h_days)
            if r:
                results.append(r)
                print(f"  {r['paths'].shape[0]} caminos simulados")

    print()
    for model in models:
        out = plot_montecarlo(results, ticker, model)
        if out:
            print(f"Guardado: {out}")


if __name__ == "__main__":
    main()
