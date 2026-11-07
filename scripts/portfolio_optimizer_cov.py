"""Optimizador de portafolio con MATRIZ DE COVARIANZA REAL (version II).

El optimizador original (`portfolio_optimizer.py`) asume que los 10 tickers
(5 BMV + 5 US) son INDEPENDIENTES entre si: el riesgo del portafolio es solo
una suma de varianzas ponderadas, sin covarianza cruzada. Esa aproximacion es
optimista cuando dos tickers estan correlacionados (ej. WALMEX y GFNORTEO,
ambas BMV y sensibles al mismo ciclo economico mexicano): diversificar entre
ellas aporta menos reduccion de riesgo real de lo que el modelo independiente
sugiere.

Este script usa la covarianza REAL estimada de los retornos diarios
historicos de cada ticker (ultimos ~2 anios, via `src.data.store`):

    maximizar   w^T r - lambda * (w^T Sigma w)
    sujeto a    sum(w_i) <= 1
                0 <= w_i <= 0.25

donde r es el vector de retorno esperado a 21 dias (%, tomado tal cual de
los JSON de pronostico `portfolio_bmv.json` / `portfolio_us.json`, no se
recalcula) y Sigma es la matriz de covarianza de retornos a 21 dias en
unidades de %^2, estimada escalando la covarianza DIARIA historica por 21
(asumiendo retornos diarios iid: Var(suma) = 21 * Var(diario)).

METODO: igual que el optimizador original, SLSQP via
scipy.optimize.minimize -- el objetivo sigue siendo cuadratico concavo en w
(Sigma es semidefinida positiva por construccion, al venir de una matriz de
covarianza muestral), con las mismas restricciones lineales simples.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, minimize

sys.path.insert(0, str(Path("/Users/rod/Projects/stock-market")))
from src.data.store import get as store_get  # noqa: E402

BASE_DIR = Path("/Users/rod/Projects/stock-market/data/eval/forecast_demo")
BMV_PATH = BASE_DIR / "portfolio_bmv.json"
US_PATH = BASE_DIR / "portfolio_us.json"
SIN_COV_PATH = BASE_DIR / "portfolio_optimo.json"
PNG_PATH = BASE_DIR / "correlacion_tickers.png"
JSON_PATH = BASE_DIR / "portfolio_optimo_cov.json"

# mapeo ticker de los JSON de pronostico -> nombre que entiende store_get()
STORE_NAME = {
    "WALMEX": "WALMEX",
    "FEMSAUBD": "FEMSAUBD",
    "GFNORTEO": "GFNORTEO",
    "CEMEXCPO": "CEMEXCPO",
    "AMXL": "AMXL",
    "SPY": "SPY",
    "AAPL": "AAPL",
    "JPM": "JPM",
    "XOM": "XOM",
    "JNJ": "JNJ",
}

PESO_MAX_TICKER = 0.25
LAMBDA_DEFAULT = 0.02
HORIZONTE_DIAS = 21
VENTANA_ANIOS = 2


def cargar_tickers() -> list[dict]:
    with open(BMV_PATH, encoding="utf-8") as f:
        bmv = json.load(f)
    with open(US_PATH, encoding="utf-8") as f:
        us = json.load(f)
    return bmv + us


def cargar_retornos_diarios(tickers: list[str]) -> pd.DataFrame:
    """Retornos log diarios de cada ticker, alineados por fecha, ultimos
    ~2 anios de historia disponible, sin NaN."""
    series = {}
    for t in tickers:
        nombre_store = STORE_NAME[t]
        df = store_get(nombre_store)
        if df is None or "Close" not in df.columns:
            raise RuntimeError(f"No hay datos en el store para {t} ({nombre_store})")
        close = df["Close"].dropna()
        corte = close.index[-1] - pd.Timedelta(days=int(365.25 * VENTANA_ANIOS))
        close = close[close.index >= corte]
        ret_log = np.log(close / close.shift(1))
        series[t] = ret_log

    retornos = pd.DataFrame(series).dropna(how="any")
    if len(retornos) < 30:
        raise RuntimeError(
            f"Muy pocas observaciones alineadas ({len(retornos)}) para estimar covarianza"
        )
    return retornos


def matriz_covarianza_horizonte_pct(retornos_diarios: pd.DataFrame) -> pd.DataFrame:
    """Covarianza diaria -> covarianza a HORIZONTE_DIAS dias, en unidades de %^2."""
    cov_diaria = retornos_diarios.cov()  # fraccion^2 por dia
    cov_horizonte_frac = cov_diaria * HORIZONTE_DIAS  # iid: Var(suma)=21*Var(diaria)
    cov_horizonte_pct = cov_horizonte_frac * (100.0 ** 2)  # fraccion^2 -> %^2
    return cov_horizonte_pct


def retorno_riesgo_portafolio(pesos: np.ndarray, retornos: np.ndarray, Sigma: np.ndarray) -> tuple[float, float]:
    retorno = float(np.sum(pesos * retornos))
    riesgo = float(np.sqrt(max(pesos @ Sigma @ pesos, 0.0)))
    return retorno, riesgo


def optimizar(retornos: np.ndarray, Sigma: np.ndarray, lam: float) -> np.ndarray:
    n = len(retornos)

    def objetivo_negativo(w: np.ndarray) -> float:
        return -(np.sum(w * retornos) - lam * (w @ Sigma @ w))

    def grad_negativo(w: np.ndarray) -> np.ndarray:
        return -(retornos - 2.0 * lam * (Sigma @ w))

    bounds = Bounds(lb=np.zeros(n), ub=np.full(n, PESO_MAX_TICKER))
    restriccion_presupuesto = LinearConstraint(np.ones((1, n)), lb=-np.inf, ub=1.0)
    w0 = np.full(n, min(PESO_MAX_TICKER, 1.0 / n) / 2.0)

    resultado = minimize(
        objetivo_negativo,
        w0,
        jac=grad_negativo,
        method="SLSQP",
        bounds=bounds,
        constraints=[restriccion_presupuesto],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not resultado.success:
        raise RuntimeError(f"SLSQP no convergio: {resultado.message}")

    return np.clip(resultado.x, 0.0, PESO_MAX_TICKER)


def graficar_correlacion(cov_pct: pd.DataFrame, tickers: list[str]) -> None:
    std = np.sqrt(np.diag(cov_pct.values))
    corr = cov_pct.values / np.outer(std, std)
    corr = np.clip(corr, -1.0, 1.0)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="YlGnBu", vmin=-1.0, vmax=1.0)

    ax.set_xticks(range(len(tickers)))
    ax.set_yticks(range(len(tickers)))
    ax.set_xticklabels(tickers, rotation=45, ha="right")
    ax.set_yticklabels(tickers)

    for i in range(len(tickers)):
        for j in range(len(tickers)):
            valor = corr[i, j]
            color = "white" if abs(valor) > 0.6 else "#222222"
            ax.text(j, i, f"{valor:.2f}", ha="center", va="center", fontsize=8, color=color)

    ax.set_title("Correlacion de retornos diarios — 10 tickers (ultimos ~2 anios)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Correlacion")

    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=150)
    plt.close(fig)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lam", "--lambda", dest="lam", type=float, default=LAMBDA_DEFAULT,
        help=f"coeficiente de aversion al riesgo (default {LAMBDA_DEFAULT})",
    )
    args = parser.parse_args()
    lam = args.lam

    activos = cargar_tickers()
    tickers = [a["ticker"] for a in activos]
    retornos = np.array([a["retorno_esperado_pct"] for a in activos])
    n = len(activos)

    print(f"Cargando historia diaria de {n} tickers desde el store...")
    retornos_diarios = cargar_retornos_diarios(tickers)
    print(f"  {len(retornos_diarios)} dias alineados (de {retornos_diarios.index[0].date()} "
          f"a {retornos_diarios.index[-1].date()})")

    Sigma_pct = matriz_covarianza_horizonte_pct(retornos_diarios)
    Sigma_pct = Sigma_pct[tickers].loc[tickers]  # reordenar igual que `tickers`
    Sigma = Sigma_pct.values

    graficar_correlacion(Sigma_pct, tickers)

    # --- portafolio optimo CON covarianza ---
    w_cov = optimizar(retornos, Sigma, lam)
    ret_cov, riesgo_cov = retorno_riesgo_portafolio(w_cov, retornos, Sigma)

    # --- comparacion contra el resultado SIN covarianza (ya calculado) ---
    with open(SIN_COV_PATH, encoding="utf-8") as f:
        sin_cov = json.load(f)
    pesos_sin_cov = sin_cov["pesos"]
    ret_sin_cov = sin_cov["retorno_portafolio_pct"]
    # riesgo real (con covarianza) del portafolio SIN covarianza, para comparar peras con peras
    w_sin_cov_arr = np.array([pesos_sin_cov.get(t, 0.0) for t in tickers])
    _, riesgo_sin_cov_real = retorno_riesgo_portafolio(w_sin_cov_arr, retornos, Sigma)
    riesgo_sin_cov_reportado = sin_cov["riesgo_portafolio_pct"]

    print()
    print("=" * 72)
    print(f"Pesos por ticker — CON covarianza vs SIN covarianza (lambda={lam})")
    print("=" * 72)
    print(f"{'Ticker':<10}{'Con cov (%)':>14}{'Sin cov (%)':>14}")
    orden = sorted(tickers, key=lambda t: -dict(zip(tickers, w_cov))[t])
    for t in orden:
        w_c = dict(zip(tickers, w_cov))[t] * 100
        w_s = pesos_sin_cov.get(t, 0.0) * 100
        if w_c > 1.0 or w_s > 1.0:
            print(f"{t:<10}{w_c:>13.1f}%{w_s:>13.1f}%")

    print()
    print("=" * 72)
    print("Comparacion de portafolios (retorno y riesgo a 21 dias)")
    print("=" * 72)
    print(f"{'Portafolio':<40}{'Retorno (%)':>14}{'Riesgo (%)':>14}")
    print(f"{'Optimo CON covarianza real':<40}{ret_cov:>14.2f}{riesgo_cov:>14.2f}")
    print(f"{'Optimo SIN covarianza (reportado)':<40}{ret_sin_cov:>14.2f}{riesgo_sin_cov_reportado:>14.2f}")
    print(f"{'  (mismo portafolio, riesgo real con cov)':<40}{ret_sin_cov:>14.2f}{riesgo_sin_cov_real:>14.2f}")

    pesos_dict = {t: float(w) for t, w in zip(tickers, w_cov)}
    salida = {
        "pesos": pesos_dict,
        "retorno_portafolio_pct": ret_cov,
        "riesgo_portafolio_pct": riesgo_cov,
        "lambda": lam,
    }
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)

    print()
    print(f"Guardado: {PNG_PATH}")
    print(f"Guardado: {JSON_PATH}")


if __name__ == "__main__":
    main()
