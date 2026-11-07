"""
Pronóstico día a día (modo "sin ancla") + banda de incertidumbre ~90%.

Combina las dos ideas que ya existen en `scripts/forecast_horizons.py`:

  - Modo "sin ancla" (`run_sin_ancla`): el modelo se ajusta UNA vez en el
    corte y luego se evalúa día por día dentro de la ventana de validación,
    componiendo solo sus propias predicciones (`price_model *= (1 + r/100)`)
    — por eso la línea azul SÍ reacciona día a día, no es una línea recta.
  - Modo "estático" (`build_forecast_path`): banda de incertidumbre ~90%
    con ancho que escala `sqrt(t) * vol_daily` (random walk), `Z_SCORE`
    fijo.

  Aquí la banda se centra en la línea día-a-día del modo sin ancla (no en
  una línea recta con deriva lineal), para que la banda "respire" junto con
  el pronóstico en vez de ser un cono separado de lo que la línea azul
  realmente dice.

Uso:
    python scripts/forecast_banda.py
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
    COLOR_REAL, COLOR_FORECAST, COLOR_BAND, OUT_DIR, VOL_WINDOW, Z_SCORE,
)
from src.data.store import get as store_get
from src.ml.features import compute_features

MIN_TRAIN_ROWS = 300


def run_banda(ticker_name: str, model_name: str, horizon_label: str, horizon: int) -> dict | None:
    """Como `run_sin_ancla`, pero además calcula una banda ~90% alrededor
    de la línea media día-a-día (no de una línea recta con deriva lineal)."""
    symbol = TICKERS[ticker_name]
    df = store_get(symbol)
    if df is None or len(df) < MIN_TRAIN_ROWS + horizon + 10:
        print(f"  [banda {model_name}] {ticker_name} {horizon_label}: datos insuficientes")
        return None

    feat = compute_features(df)
    cutoff_idx = len(feat) - horizon - 1
    if cutoff_idx < MIN_TRAIN_ROWS:
        return None
    cutoff_date = feat.index[cutoff_idx]
    price0 = float(df["Close"].iloc[cutoff_idx])

    # Volatilidad diaria reciente (misma receta que run_one en
    # forecast_horizons.py): std de log-retornos, ventana VOL_WINDOW antes
    # del corte, en %.
    daily_log_ret = np.log(df["Close"] / df["Close"].shift(1))
    vol_daily = float(daily_log_ret.iloc[max(0, cutoff_idx - VOL_WINDOW):cutoff_idx].std())
    if not np.isfinite(vol_daily) or vol_daily <= 0:
        print(f"  [banda {model_name}] {ticker_name} {horizon_label}: volatilidad inválida")
        return None
    vol_daily_pct = vol_daily * 100.0

    try:
        if model_name == "regimen_hmm":
            state = fit_regimen(df, feat, cutoff_idx)
        else:
            state = FIT_ROLLING[model_name](feat, cutoff_idx)
    except Exception as exc:
        print(f"  [banda {model_name}] {ticker_name} {horizon_label}: ERROR ajuste {exc}")
        return None
    if state is None:
        return None

    rows = []
    price_model = price0
    t = 0
    for idx in range(cutoff_idx, cutoff_idx + horizon):
        if idx + 1 >= len(df):
            break
        price_real = float(df["Close"].iloc[idx + 1])
        try:
            gain_pred_pct = apply_fitted(state, feat, idx)
        except Exception:
            continue
        if not np.isfinite(gain_pred_pct):
            continue
        price_model *= (1 + gain_pred_pct / 100.0)
        t += 1
        width = Z_SCORE * (vol_daily_pct / 100.0) * np.sqrt(t)
        lower = price_model * np.exp(-width)
        upper = price_model * np.exp(width)
        rows.append({
            "date": feat.index[idx + 1],
            "price_model": price_model,
            "lower": lower,
            "upper": upper,
            "price_real": price_real,
        })

    if not rows:
        return None
    daily = pd.DataFrame(rows)
    return {
        "ticker": ticker_name, "model": model_name, "horizon": horizon_label,
        "cutoff_date": cutoff_date, "daily": daily,
    }


def plot_banda(results: list[dict], ticker: str, model: str) -> Path | None:
    """Una gráfica por (modelo, ticker): 1 fila x 3 columnas (30d/60d/90d).
    Real en rojo, media del modelo (sin ancla) en azul punteado, banda de
    incertidumbre ~90% alrededor de esa línea media."""
    rows = [r for r in results if r["ticker"] == ticker and r["model"] == model]
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: list(HORIZONS).index(r["horizon"]))

    fig, axes = plt.subplots(1, len(rows), figsize=(6 * len(rows), 4.2))
    axes = np.atleast_1d(axes)

    for ax, r in zip(axes, rows):
        daily = r["daily"]
        ax.fill_between(daily["date"], daily["lower"], daily["upper"],
                         color=COLOR_BAND, alpha=0.15, label="Incertidumbre ~90%", zorder=1)
        ax.plot(daily["date"], daily["price_real"], color=COLOR_REAL, lw=2.3,
                 label="Real", zorder=5)
        ax.plot(daily["date"], daily["price_model"], color=COLOR_FORECAST, lw=1.8,
                 ls="--", label="Modelo (sin ancla, día a día)", zorder=4)
        ax.set_title(f"{r['horizon']} · corte {r['cutoff_date'].date()}", fontsize=11)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)

    axes[0].legend(fontsize=9, frameon=False)
    fig.suptitle(f"{ticker} — intento: {model} (sin ancla + banda de incertidumbre)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"banda_{model}_{ticker}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main() -> None:
    ticker = "SPY"
    models = list(FIT_ROLLING.keys()) + ["regimen_hmm"]

    results = []
    for model_name in models:
        for horizon_label, horizon in HORIZONS.items():
            r = run_banda(ticker, model_name, horizon_label, horizon)
            if r is not None:
                results.append(r)

    for model_name in models:
        out_path = plot_banda(results, ticker, model_name)
        if out_path is not None:
            print(f"Guardado: {out_path}")
        else:
            print(f"Sin datos suficientes para graficar {model_name} ({ticker})")


if __name__ == "__main__":
    main()
