"""
Retorno esperado y riesgo a 30 días (21 sesiones) para un portafolio de
tickers de EEUU (SPY, AAPL, JPM, XOM, JNJ) usando simulación Montecarlo con
el modelo "lightgbm_ranking".

Mismo patrón que `run_montecarlo` en `scripts/forecast_montecarlo.py`: el
modelo se ajusta UNA sola vez en el corte (rolling fit de 1 día adelante) y
luego se compone día a día SOLO con su propia predicción + ruido gaussiano
calibrado a la volatilidad diaria realizada (ventana `VOL_WINDOW` antes del
corte). A diferencia de `run_montecarlo`, aquí no usamos `TICKERS` de
`forecast_horizons` (que solo tiene WALMEX/SPY) — recibimos el símbolo del
store directamente, para poder correr tickers de EEUU como AAPL/JPM/XOM/JNJ.

Uso:
    python scripts/portfolio_us.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from scripts.forecast_horizons import FIT_ROLLING, apply_fitted, VOL_WINDOW
from src.data.store import get as store_get
from src.ml.features import compute_features

MIN_TRAIN_ROWS = 300
OUT_PATH = ROOT / "data" / "eval" / "forecast_demo" / "portfolio_us.json"

TICKERS_US = ["SPY", "AAPL", "JPM", "XOM", "JNJ"]


def simular(ticker_symbol: str, horizon: int = 21, n_paths: int = 200, seed: int = 0) -> dict | None:
    """Ajusta lightgbm_ranking una vez en el corte y simula `n_paths` caminos
    de precio componiendo día a día (predicción del modelo + ruido ~N(0, vol))."""
    df = store_get(ticker_symbol)
    if df is None or len(df) < MIN_TRAIN_ROWS + horizon + 10:
        print(f"  [{ticker_symbol}] datos insuficientes")
        return None

    feat = compute_features(df)
    cutoff_idx = len(feat) - horizon - 1
    if cutoff_idx < MIN_TRAIN_ROWS:
        print(f"  [{ticker_symbol}] cutoff_idx < MIN_TRAIN_ROWS")
        return None
    price0 = float(df["Close"].iloc[cutoff_idx])

    daily_log_ret = np.log(df["Close"] / df["Close"].shift(1))
    vol_daily_pct = float(
        daily_log_ret.iloc[cutoff_idx - VOL_WINDOW:cutoff_idx].std() * 100
    )
    if not np.isfinite(vol_daily_pct) or vol_daily_pct <= 0:
        print(f"  [{ticker_symbol}] volatilidad inválida")
        return None

    try:
        state = FIT_ROLLING["lightgbm_ranking"](feat, cutoff_idx)
    except Exception as exc:
        print(f"  [{ticker_symbol}] ERROR ajuste: {exc}")
        return None
    if state is None:
        print(f"  [{ticker_symbol}] ajuste devolvió None")
        return None

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

    final_prices = paths[:, -1]
    retorno_esperado_pct = float((np.median(final_prices) / price0 - 1) * 100)
    p95 = np.percentile(final_prices, 95)
    p5 = np.percentile(final_prices, 5)
    riesgo_pct = float((p95 - p5) / price0 * 100)

    return {
        "ticker": ticker_symbol,
        "precio0": price0,
        "retorno_esperado_pct": retorno_esperado_pct,
        "riesgo_pct": riesgo_pct,
    }


def main() -> None:
    print("=== Portafolio EEUU — Montecarlo lightgbm_ranking, 30d (21 sesiones) ===")
    resultados = []
    for ticker in TICKERS_US:
        print(f"[{ticker}] simulando...")
        r = simular(ticker)
        if r:
            resultados.append(r)
            print(f"  precio0={r['precio0']:.2f}  retorno_esp={r['retorno_esperado_pct']:+.2f}%  "
                  f"riesgo={r['riesgo_pct']:.2f}%")
        else:
            print(f"  [{ticker}] FALLÓ — se omite")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    print(f"\nGuardado: {OUT_PATH}")

    print("\n{:<8} {:>12} {:>18} {:>12}".format("Ticker", "Precio0", "Retorno Esp. %", "Riesgo %"))
    for r in resultados:
        print("{:<8} {:>12.2f} {:>18.2f} {:>12.2f}".format(
            r["ticker"], r["precio0"], r["retorno_esperado_pct"], r["riesgo_pct"]))


if __name__ == "__main__":
    main()
