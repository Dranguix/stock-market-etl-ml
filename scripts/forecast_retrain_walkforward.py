"""
Comparación de estrategias de reentrenamiento para el modelo
"lightgbm_ranking" sobre SPY: ¿qué tanto importa reentrenar seguido en vez
de ajustar una sola vez y predecir día a día con un modelo viejo?

Contexto (ver scripts/forecast_horizons.py, modo "mesh"/"sin_ancla"): ese
script ya demostró que un modelo ajustado una sola vez en un corte fijo
(FIT_ROLLING["lightgbm_ranking"] -> fit_lightgbm) y aplicado día a día con
apply_fitted(), sin reentrenar, se va quedando "viendo un mundo viejo" con
el paso del tiempo. Aquí medimos el error (MAE en precio) de 3 estrategias
de reentrenamiento a lo largo de los últimos ~4 años de SPY:

  (a) nunca reentrena      -> 1 solo ajuste, al inicio de la ventana.
  (b) reentrena cada 60d   -> ~1 vez por trimestre.
  (c) reentrena cada 20d   -> ~1 vez por mes.

Uso:
    python scripts/forecast_retrain_walkforward.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.forecast_horizons import FIT_ROLLING, apply_fitted, MIN_TRAIN_ROWS
from src.data.store import get as store_get
from src.ml.features import compute_features

OUT_DIR = ROOT / "data" / "eval" / "forecast_demo"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TICKER = "SPY"
MODEL_NAME = "lightgbm_ranking"
TEST_WINDOW = 1000  # ~4 años de sesiones

STRATEGIES = [
    ("nunca_reentrena", None),
    ("reentrena_cada_60d", 60),
    ("reentrena_cada_20d", 20),
]

COLOR_BAR = "#1f6feb"


def main() -> None:
    df = store_get(TICKER)
    if df is None:
        raise RuntimeError(f"No se encontró {TICKER} en el store")
    feat = compute_features(df)

    n = len(feat)
    window = min(TEST_WINDOW, n - MIN_TRAIN_ROWS - 10)
    if window < 100:
        raise RuntimeError(
            f"No hay suficientes datos para una ventana de prueba razonable "
            f"(n={n}, MIN_TRAIN_ROWS={MIN_TRAIN_ROWS})"
        )
    test_start_idx = n - window - 1  # -1: necesitamos "mañana" para el último día
    test_idxs = list(range(test_start_idx, n - 1))  # idx -> predice feat idx+1

    print(f"{TICKER}: {n} sesiones totales, ventana de prueba = {len(test_idxs)} sesiones "
          f"({feat.index[test_idxs[0]].date()} -> {feat.index[test_idxs[-1]].date()})")

    fit_fn = FIT_ROLLING[MODEL_NAME]
    resumen = []

    for name, retrain_every in STRATEGIES:
        state = None
        n_reentrenos = 0
        errors = []

        for pos, idx in enumerate(test_idxs):
            do_retrain = (
                state is None
                or (retrain_every is not None and pos % retrain_every == 0)
            )
            if do_retrain:
                new_state = fit_fn(feat, idx)
                if new_state is not None:
                    state = new_state
                    n_reentrenos += 1

            if state is None:
                continue  # no hubo suficientes datos para ajustar aún

            pred_ret_pct = apply_fitted(state, feat, idx)
            price_hoy = float(feat["Close"].iloc[idx])
            price_pred = price_hoy * (1 + pred_ret_pct / 100)
            price_real = float(feat["Close"].iloc[idx + 1])
            errors.append(abs(price_pred - price_real))

        mae = float(np.mean(errors)) if errors else float("nan")
        resumen.append({"estrategia": name, "mae": mae, "n_reentrenos": n_reentrenos})
        print(f"  {name:22s} MAE={mae:8.3f}  n_reentrenos={n_reentrenos}")

    # ── Tabla en consola ──────────────────────────────────────────────────
    print("\nEstrategia              MAE ($)   # Reentrenos")
    print("-" * 48)
    for r in resumen:
        print(f"{r['estrategia']:22s} {r['mae']:8.3f}   {r['n_reentrenos']}")

    # ── JSON resumen ──────────────────────────────────────────────────────
    out_json = OUT_DIR / "retrain_resumen.json"
    out_json.write_text(json.dumps(resumen, indent=2, ensure_ascii=False))
    print(f"\nJSON guardado en {out_json}")

    # ── Gráfica de barras ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [r["estrategia"] for r in resumen]
    maes = [r["mae"] for r in resumen]
    bars = ax.bar(labels, maes, color=COLOR_BAR, width=0.55)

    for bar, mae in zip(bars, maes):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{mae:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_ylabel("MAE del precio pronosticado a 1 día ($)")
    ax.set_title(f"{TICKER} — lightgbm_ranking: error según frecuencia de reentrenamiento\n"
                 f"(ventana de prueba: últimos {len(test_idxs)} días hábiles)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.set_ylim(0, max(maes) * 1.2)

    fig.tight_layout()
    out_png = OUT_DIR / "retrain_comparacion.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Gráfica guardada en {out_png}")


if __name__ == "__main__":
    main()
