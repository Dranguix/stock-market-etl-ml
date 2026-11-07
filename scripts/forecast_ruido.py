"""
Variante "con ruido" de `run_sin_ancla` (ver `scripts/forecast_horizons.py`):
mismo ajuste de modelo UNA vez en el corte, misma composición día a día de
SOLO las predicciones propias del modelo (sin copiar el precio real) — pero
en cada paso se le suma ruido gaussiano calibrado a la volatilidad diaria
realizada reciente (ventana `VOL_WINDOW`, mismo patrón que `run_one`).

Por qué: `run_sin_ancla` compone el retorno esperado (la media condicional
del modelo) día tras día, así que la línea sale lisa — no hay zigzag porque
nunca se simula el ruido diario que sí tiene el mercado real. Aquí se agrega
ese ruido (mismo orden de magnitud que la volatilidad realizada) para que la
trayectoria del modelo se vea tan "viva" como la real, sin tocar la
dirección/magnitud esperada que el modelo sí pronostica.

Uso:
    python scripts/forecast_ruido.py
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
    COLOR_REAL, COLOR_FORECAST, OUT_DIR, VOL_WINDOW, MODEL_STYLES,
)
from src.data.store import get as store_get
from src.ml.features import compute_features

MIN_TRAIN_ROWS = 300


def run_ruidoso(ticker_name: str, model_name: str, horizon_label: str, horizon: int,
                 seed: int = 0) -> dict | None:
    """Como `run_sin_ancla`, pero cada paso de la composición diaria recibe
    un empujón aleatorio ~N(0, vol_daily_pct) para imitar el zigzag real."""
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
        daily_log_ret.iloc[max(0, cutoff_idx - VOL_WINDOW):cutoff_idx].std() * 100
    )
    if not np.isfinite(vol_daily_pct) or vol_daily_pct <= 0:
        return None

    try:
        if model_name == "regimen_hmm":
            state = fit_regimen(df, feat, cutoff_idx)
        else:
            state = FIT_ROLLING[model_name](feat, cutoff_idx)
    except Exception as exc:
        print(f"  [ruido {model_name}] {ticker_name} {horizon_label}: ERROR ajuste {exc}")
        return None
    if state is None:
        return None

    rng = np.random.default_rng(seed)
    rows = []
    price_model = price0
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
        ruido = rng.normal(0, vol_daily_pct)
        price_model *= (1 + (gain_pred_pct + ruido) / 100.0)
        rows.append({"date": feat.index[idx + 1], "price_model": price_model, "price_real": price_real})

    if not rows:
        return None
    daily = pd.DataFrame(rows)
    mae_model = float((daily["price_model"] - daily["price_real"]).abs().mean())
    return {
        "ticker": ticker_name, "model": model_name, "horizon": horizon_label,
        "cutoff_date": cutoff_date, "price0": price0, "daily": daily,
        "mae_model": mae_model,
    }


def plot_ruidoso(results: list[dict], ticker: str, model: str) -> Path | None:
    """1 fila x 3 columnas (30d/60d/90d): real (rojo) vs modelo con ruido
    (azul, guión), por modelo/ticker."""
    rows = [r for r in results if r["ticker"] == ticker and r["model"] == model]
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: list(HORIZONS).index(r["horizon"]))

    fig, axes = plt.subplots(1, len(rows), figsize=(6 * len(rows), 4.2))
    axes = np.atleast_1d(axes)

    for ax, r in zip(axes, rows):
        daily = r["daily"]
        ax.plot(daily["date"], daily["price_real"], color=COLOR_REAL, lw=2.3, label="Real")
        ax.plot(daily["date"], daily["price_model"], color=COLOR_FORECAST, lw=1.8,
                 ls="--", label="Modelo + ruido")
        ax.set_title(
            f"{r['horizon']} · corte {r['cutoff_date'].date()}\nMAE modelo {r['mae_model']:.2f}",
            fontsize=9.5,
        )
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)

    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"{ticker} — intento: {model} (sin ancla + ruido calibrado)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"ruido_{model}_{ticker}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main() -> None:
    ticker = "SPY"
    models = ["lightgbm_ranking", "momentum_12_1", "woe_score", "regimen_hmm"]

    results = []
    for model_name in models:
        for horizon_label, horizon in HORIZONS.items():
            r = run_ruidoso(ticker, model_name, horizon_label, horizon)
            if r is None:
                print(f"  [ruido {model_name}] {ticker} {horizon_label}: sin resultado")
                continue
            results.append(r)

    for model_name in models:
        out_path = plot_ruidoso(results, ticker, model_name)
        if out_path:
            print(f"Guardado: {out_path}")
        else:
            print(f"  [ruido {model_name}] {ticker}: nada que graficar")


if __name__ == "__main__":
    main()
