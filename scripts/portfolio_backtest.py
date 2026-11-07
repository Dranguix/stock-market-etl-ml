"""Backtest walk-forward del optimizador de portafolio (demo educativa).

El optimizador en `scripts/portfolio_optimizer.py` resuelve pesos dado un
retorno/riesgo esperado a 21 sesiones por ticker, calculado una sola vez
(hoy). Este script valida esa idea hacia atras en el tiempo: en 5 fechas de
corte pasadas, recalcula retorno/riesgo esperado con el MISMO patron de
`simular()` (lightgbm_ranking + Montecarlo con ruido calibrado a volatilidad,
ver `scripts/portfolio_bmv.py` / `scripts/portfolio_us.py`), pero ajustando
el modelo EN ESE CORTE (sin look-ahead) en vez de al final de la serie.

Para cada fecha de corte y cada valor de lambda en una grilla, resuelve el
portafolio optimo (mismo SLSQP que `portfolio_optimizer.py`) y mide el
retorno REALIZADO 21 sesiones despues (precio real, no simulado). Acumula
por lambda el retorno realizado en las 5 fechas, y reporta promedio,
desviacion estandar y un "Sharpe realizado" simple (promedio/desviacion).

ALCANCE: demo educativa con N=5 fechas de corte -- no hay significancia
estadistica rigurosa aqui, solo una senal direccional de que lambda se
comporto mejor en esta muestra chica. Con 5 puntos la desviacion estandar
es muy ruidosa y el "mejor" lambda puede cambiar con otra semilla o otras
fechas.

Uso:
    python scripts/portfolio_backtest.py
"""
from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import Bounds, LinearConstraint, minimize

from scripts.forecast_horizons import FIT_ROLLING, apply_fitted, VOL_WINDOW
from src.data.store import get as store_get
from src.ml.features import compute_features

MIN_TRAIN_ROWS = 300
HORIZON = 21  # sesiones (~30 dias calendario)
N_PATHS = 80  # menos que simular() original (200) para que el backtest corra en minutos
N_CUTOFFS = 5
SPACING = 60  # sesiones entre cortes, aproximado (se usa el indice de SPY)
PESO_MAX_TICKER = 0.25
LAMBDAS = [0.005, 0.01, 0.02, 0.05, 0.1]

TICKERS_BMV = ["WALMEX", "FEMSAUBD", "GFNORTEO", "CEMEXCPO", "AMXL"]
TICKERS_US = ["SPY", "AAPL", "JPM", "XOM", "JNJ"]
TICKERS_ALL = TICKERS_BMV + TICKERS_US

OUT_DIR = ROOT / "data" / "eval" / "forecast_demo"
PNG_PATH = OUT_DIR / "lambda_backtest.png"
JSON_PATH = OUT_DIR / "lambda_backtest.json"


# ---------------------------------------------------------------------------
# 1. Fechas de corte (basadas en el indice de SPY, el ticker con mas historia)
# ---------------------------------------------------------------------------

def elegir_fechas_corte() -> list:
    df_spy = store_get("SPY")
    feat_spy = compute_features(df_spy)
    n = len(feat_spy)
    idx_max = n - 1 - HORIZON  # ultimo corte posible (deja HORIZON sesiones despues)
    idx_min = MIN_TRAIN_ROWS + (N_CUTOFFS - 1) * SPACING  # deja margen de entrenamiento
    if idx_max - (N_CUTOFFS - 1) * SPACING < idx_min:
        raise RuntimeError("No hay suficiente historia en SPY para 5 cortes espaciados")

    idxs = [idx_max - k * SPACING for k in range(N_CUTOFFS)]
    idxs = sorted(idxs)  # orden cronologico: corte mas antiguo primero
    fechas = [feat_spy.index[i] for i in idxs]
    return fechas


def mapear_cutoff_idx(ticker_symbol: str, fecha_corte) -> int | None:
    """Encuentra el indice en el feat del ticker mas cercano a `fecha_corte`."""
    df = store_get(ticker_symbol)
    if df is None:
        return None
    pos = df.index.get_indexer([fecha_corte], method="nearest")[0]
    if pos < 0:
        return None
    return int(pos)


# ---------------------------------------------------------------------------
# 2. Simulacion Montecarlo en un corte arbitrario (mismo patron que
#    scripts/portfolio_bmv.py::simular, parametrizado por cutoff_idx)
# ---------------------------------------------------------------------------

def simular_en_corte(ticker_symbol: str, cutoff_idx: int, horizon: int = HORIZON,
                      n_paths: int = N_PATHS, seed: int = 0) -> dict | None:
    df = store_get(ticker_symbol)
    if df is None or len(df) < MIN_TRAIN_ROWS + horizon + 10:
        return None
    if cutoff_idx < MIN_TRAIN_ROWS or cutoff_idx + horizon >= len(df):
        return None

    feat = compute_features(df)
    price0 = float(df["Close"].iloc[cutoff_idx])

    daily_log_ret = np.log(df["Close"] / df["Close"].shift(1))
    vol_daily_pct = float(
        daily_log_ret.iloc[cutoff_idx - VOL_WINDOW:cutoff_idx].std() * 100
    )
    if not np.isfinite(vol_daily_pct) or vol_daily_pct <= 0:
        return None

    try:
        state = FIT_ROLLING["lightgbm_ranking"](feat, cutoff_idx)
    except Exception as exc:
        print(f"    [{ticker_symbol}] ERROR ajuste: {exc}")
        return None
    if state is None:
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

    # retorno realizado: precio real cutoff_idx+horizon vs cutoff_idx
    price_fwd = float(df["Close"].iloc[cutoff_idx + horizon])
    retorno_realizado_pct = float((price_fwd / price0 - 1) * 100)

    return {
        "ticker": ticker_symbol,
        "precio0": price0,
        "retorno_esperado_pct": retorno_esperado_pct,
        "riesgo_pct": riesgo_pct,
        "retorno_realizado_pct": retorno_realizado_pct,
    }


# ---------------------------------------------------------------------------
# 3. Optimizador (misma logica que scripts/portfolio_optimizer.py)
# ---------------------------------------------------------------------------

def optimizar(retornos: np.ndarray, riesgos: np.ndarray, lam: float) -> np.ndarray:
    n = len(retornos)

    def objetivo_negativo(w: np.ndarray) -> float:
        return -(np.sum(w * retornos) - lam * np.sum((w ** 2) * (riesgos ** 2)))

    def grad_negativo(w: np.ndarray) -> np.ndarray:
        return -(retornos - 2.0 * lam * w * (riesgos ** 2))

    bounds = Bounds(lb=np.zeros(n), ub=np.full(n, PESO_MAX_TICKER))
    restriccion_presupuesto = LinearConstraint(np.ones((1, n)), lb=-np.inf, ub=1.0)
    w0 = np.full(n, min(PESO_MAX_TICKER, 1.0 / n) / 2.0)

    resultado = minimize(
        objetivo_negativo, w0, jac=grad_negativo, method="SLSQP",
        bounds=bounds, constraints=[restriccion_presupuesto],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not resultado.success:
        raise RuntimeError(f"SLSQP no convergio: {resultado.message}")
    return np.clip(resultado.x, 0.0, PESO_MAX_TICKER)


# ---------------------------------------------------------------------------
# 4. Backtest principal
# ---------------------------------------------------------------------------

def main() -> None:
    fechas_corte = elegir_fechas_corte()
    print("Fechas de corte elegidas (walk-forward, indice de SPY):")
    for f in fechas_corte:
        print(f"  - {f.date()}")
    print()

    # por lambda, lista de retornos realizados (uno por fecha de corte)
    retornos_por_lambda = {lam: [] for lam in LAMBDAS}
    retornos_eq = []  # referencia: igualmente ponderado
    detalle_fechas = []

    for i, fecha_corte in enumerate(fechas_corte):
        print(f"=== Corte {i + 1}/{len(fechas_corte)}: {fecha_corte.date()} ===")
        activos = []
        for ticker in TICKERS_ALL:
            cutoff_idx = mapear_cutoff_idx(ticker, fecha_corte)
            if cutoff_idx is None:
                print(f"  [{ticker}] no se pudo mapear la fecha de corte, se omite")
                continue
            r = simular_en_corte(ticker, cutoff_idx)
            if r is None:
                print(f"  [{ticker}] simulacion fallida, se omite")
                continue
            activos.append(r)
            print(f"  [{ticker}] esperado={r['retorno_esperado_pct']:+.2f}%  "
                  f"riesgo={r['riesgo_pct']:.2f}%  realizado={r['retorno_realizado_pct']:+.2f}%")

        if len(activos) < 2:
            print(f"  corte {fecha_corte.date()}: muy pocos activos validos, se omite del backtest")
            continue

        tickers_ok = [a["ticker"] for a in activos]
        retornos_esp = np.array([a["retorno_esperado_pct"] for a in activos])
        riesgos = np.array([a["riesgo_pct"] for a in activos])
        retornos_real = np.array([a["retorno_realizado_pct"] for a in activos])
        n = len(activos)

        # referencia igualmente ponderada
        w_eq = np.full(n, 1.0 / n)
        ret_eq_realizado = float(np.sum(w_eq * retornos_real))
        retornos_eq.append(ret_eq_realizado)

        corte_detalle = {
            "fecha_corte": str(fecha_corte.date()),
            "tickers": tickers_ok,
            "retorno_esperado_pct": retornos_esp.tolist(),
            "riesgo_pct": riesgos.tolist(),
            "retorno_realizado_pct": retornos_real.tolist(),
            "retorno_realizado_igualmente_ponderado_pct": ret_eq_realizado,
            "lambdas": {},
        }

        for lam in LAMBDAS:
            w_opt = optimizar(retornos_esp, riesgos, lam)
            ret_realizado = float(np.sum(w_opt * retornos_real))
            retornos_por_lambda[lam].append(ret_realizado)
            corte_detalle["lambdas"][str(lam)] = {
                "pesos": {t: float(w) for t, w in zip(tickers_ok, w_opt)},
                "retorno_realizado_pct": ret_realizado,
            }
            print(f"  lambda={lam:<6} retorno_realizado_portafolio={ret_realizado:+.2f}%")

        detalle_fechas.append(corte_detalle)
        print()

    # ---------------------------------------------------------------
    # 5. Resumen por lambda: promedio, desviacion, Sharpe realizado
    # ---------------------------------------------------------------
    resumen = {}
    for lam in LAMBDAS:
        datos = np.array(retornos_por_lambda[lam])
        if len(datos) == 0:
            continue
        promedio = float(np.mean(datos))
        desviacion = float(np.std(datos, ddof=1)) if len(datos) > 1 else 0.0
        sharpe = float(promedio / desviacion) if desviacion > 0 else float("nan")
        resumen[lam] = {
            "n_cortes": len(datos),
            "retornos_realizados_pct": datos.tolist(),
            "promedio_pct": promedio,
            "desviacion_pct": desviacion,
            "sharpe_realizado": sharpe,
        }

    datos_eq = np.array(retornos_eq)
    promedio_eq = float(np.mean(datos_eq)) if len(datos_eq) else float("nan")
    desviacion_eq = float(np.std(datos_eq, ddof=1)) if len(datos_eq) > 1 else 0.0
    sharpe_eq = float(promedio_eq / desviacion_eq) if desviacion_eq > 0 else float("nan")

    # mejor lambda por Sharpe (ignorando NaN)
    lambdas_con_sharpe = {l: v["sharpe_realizado"] for l, v in resumen.items()
                          if not math.isnan(v["sharpe_realizado"])}
    mejor_lambda = max(lambdas_con_sharpe, key=lambdas_con_sharpe.get) if lambdas_con_sharpe else None

    # ---------------------------------------------------------------
    # 6. Tabla en consola
    # ---------------------------------------------------------------
    print("=" * 72)
    print(f"Backtest walk-forward del optimizador (N={len(detalle_fechas)} cortes)")
    print("=" * 72)
    print(f"{'Lambda':<10}{'Retorno prom. (%)':>20}{'Desv. (%)':>14}{'Sharpe realizado':>20}")
    for lam in LAMBDAS:
        if lam not in resumen:
            continue
        r = resumen[lam]
        marca = "  <-- mejor" if lam == mejor_lambda else ""
        print(f"{lam:<10}{r['promedio_pct']:>20.3f}{r['desviacion_pct']:>14.3f}"
              f"{r['sharpe_realizado']:>20.3f}{marca}")
    print("-" * 72)
    print(f"{'Igual ponderado':<10}{promedio_eq:>20.3f}{desviacion_eq:>14.3f}{sharpe_eq:>20.3f}  (referencia)")
    print()
    if mejor_lambda is not None:
        print(f"Mejor lambda por Sharpe realizado: {mejor_lambda}")
    print("NOTA: N=5 fechas de corte es una muestra muy chica -- esto es una")
    print("senal direccional para una demo educativa, NO un resultado con")
    print("significancia estadistica. Otra semilla u otras fechas podrian")
    print("dar un lambda 'ganador' distinto.")

    # ---------------------------------------------------------------
    # 7. Grafica: lambda vs Sharpe realizado
    # ---------------------------------------------------------------
    lams_plot = [lam for lam in LAMBDAS if lam in resumen]
    sharpes_plot = [resumen[lam]["sharpe_realizado"] for lam in lams_plot]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x_pos = np.arange(len(lams_plot))
    ax.bar(x_pos, sharpes_plot, color="#1f6feb", width=0.6, zorder=3)
    if not math.isnan(sharpe_eq):
        ax.axhline(sharpe_eq, color="#888888", linestyle="--", linewidth=1.2,
                    label=f"Igualmente ponderado (Sharpe={sharpe_eq:.2f})", zorder=2)
        ax.legend(loc="best", frameon=False)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(l) for l in lams_plot])
    ax.set_xlabel("Lambda (aversion al riesgo)")
    ax.set_ylabel("Sharpe realizado (promedio / desviacion, N=5 cortes)")
    ax.set_title("Backtest walk-forward: lambda vs Sharpe realizado")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------
    # 8. JSON resumen
    # ---------------------------------------------------------------
    salida = {
        "nota_alcance": (
            "Demo educativa con N=5 fechas de corte walk-forward. No hay "
            "significancia estadistica rigurosa; es solo una senal direccional "
            "de que lambda se comporto mejor en esta muestra chica."
        ),
        "horizon_sesiones": HORIZON,
        "n_paths_montecarlo": N_PATHS,
        "fechas_corte": [str(f.date()) for f in fechas_corte],
        "lambdas_evaluados": LAMBDAS,
        "mejor_lambda_por_sharpe": mejor_lambda,
        "resumen_por_lambda": {str(l): v for l, v in resumen.items()},
        "referencia_igualmente_ponderado": {
            "n_cortes": len(datos_eq),
            "retornos_realizados_pct": datos_eq.tolist(),
            "promedio_pct": promedio_eq,
            "desviacion_pct": desviacion_eq,
            "sharpe_realizado": sharpe_eq,
        },
        "detalle_por_corte": detalle_fechas,
    }
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)

    print()
    print(f"Guardado: {PNG_PATH}")
    print(f"Guardado: {JSON_PATH}")


if __name__ == "__main__":
    main()
