"""
Retorno esperado y riesgo a 30 días (21 sesiones) para 5 tickers de la BMV,
por simulación Montecarlo (mismo patrón que `forecast_montecarlo.py`:
ajuste del modelo lightgbm_ranking UNA sola vez en el corte, luego se
compone día a día solo con su propia predicción + ruido calibrado a la
volatilidad real) — adaptado para recibir el símbolo directo del store en
vez de depender de `TICKERS` de `scripts.forecast_horizons` (que hoy solo
tiene WALMEX y SPY).

Uso:
    python scripts/portfolio_bmv.py
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

from scripts.forecast_horizons import FIT_ROLLING, apply_fitted, VOL_WINDOW, OUT_DIR
from src.data.store import get as store_get
from src.ml.features import compute_features

MIN_TRAIN_ROWS = 300
# Nota: para América Móvil el store usa la clave "AMXL" (serie L), que se
# resuelve internamente a AMXB.MX en Yahoo Finance (ver src/data/bmv_catalog.py).
# "AMXB" tal cual no es una clave del store.
TICKERS_BMV = ["WALMEX", "FEMSAUBD", "GFNORTEO", "CEMEXCPO", "AMXL"]


def simular(ticker_symbol: str, horizon: int = 21, n_paths: int = 200, seed: int = 0) -> dict | None:
    """Ajusta lightgbm_ranking una vez en el corte y simula `n_paths` caminos
    de precio agregando ruido ~N(0, vol_daily_pct) a la ganancia predicha
    cada día (mismo criterio que `run_montecarlo` en forecast_montecarlo.py),
    pero recibiendo el símbolo del store directamente (sin pasar por
    `TICKERS`)."""
    df = store_get(ticker_symbol)
    if df is None or len(df) < MIN_TRAIN_ROWS + horizon + 10:
        print(f"  [{ticker_symbol}] datos insuficientes")
        return None

    feat = compute_features(df)
    cutoff_idx = len(feat) - horizon - 1
    if cutoff_idx < MIN_TRAIN_ROWS:
        print(f"  [{ticker_symbol}] cutoff_idx < MIN_TRAIN_ROWS")
        return None
    cutoff_date = feat.index[cutoff_idx]
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
    p95 = float(np.percentile(final_prices, 95))
    p5 = float(np.percentile(final_prices, 5))
    riesgo_pct = float((p95 - p5) / price0 * 100)

    return {
        "ticker": ticker_symbol,
        "precio0": price0,
        "retorno_esperado_pct": retorno_esperado_pct,
        "riesgo_pct": riesgo_pct,
        "cutoff_date": str(cutoff_date.date()),
        "n_paths": n_paths,
        "horizon": horizon,
    }


def main() -> None:
    results = []
    for ticker in TICKERS_BMV:
        print(f"[portfolio_bmv] simulando {ticker}...")
        r = simular(ticker, horizon=21, n_paths=200, seed=0)
        if r:
            results.append(r)
            print(f"  retorno_esperado={r['retorno_esperado_pct']:+.2f}%  riesgo={r['riesgo_pct']:.2f}%")
        else:
            print(f"  {ticker}: SIN RESULTADO (ver mensaje arriba)")

    print()
    print(f"{'Ticker':<12}{'Precio0':>12}{'Retorno esp. %':>18}{'Riesgo %':>14}")
    print("-" * 56)
    for r in results:
        print(f"{r['ticker']:<12}{r['precio0']:>12.2f}{r['retorno_esperado_pct']:>18.2f}{r['riesgo_pct']:>14.2f}")

    out_dir = ROOT / "data" / "eval" / "forecast_demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "portfolio_bmv.json"
    out_json = [
        {
            "ticker": r["ticker"],
            "precio0": r["precio0"],
            "retorno_esperado_pct": r["retorno_esperado_pct"],
            "riesgo_pct": r["riesgo_pct"],
        }
        for r in results
    ]
    out_path.write_text(json.dumps(out_json, indent=2, ensure_ascii=False))
    print(f"\nGuardado: {out_path}")


if __name__ == "__main__":
    main()
