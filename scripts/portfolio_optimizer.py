"""Optimizador de portafolio tipo Markowitz/knapsack (demo educativa).

Carga los pronosticos de retorno/riesgo a 30 dias (simulados con Monte Carlo
sobre un modelo LightGBM calibrado) de 10 tickers (5 BMV + 5 US) y resuelve:

    maximizar   sum(w_i * retorno_i) - lambda * sum(w_i^2 * riesgo_i^2)
    sujeto a    sum(w_i) <= 1
                0 <= w_i <= 0.25   (tope de 25% del presupuesto por ticker)

donde w_i es la fraccion del presupuesto de $50 invertida en el ticker i.

NOTA SOBRE EL RIESGO: no tenemos matriz de covarianza entre los 10 tickers
(no se simularon trayectorias conjuntas), asi que esta demo asume
independencia entre activos. El termino de riesgo es entonces una suma de
varianzas ponderadas, sin terminos de covarianza cruzada -- una
aproximacion simple y conservadora, no un Markowitz completo.

METODO ELEGIDO: scipy.optimize.minimize con SLSQP. El objetivo es una
funcion cuadratica diferenciable (concava en w, porque el termino de
riesgo es -lambda*w^2 con lambda>0) con restricciones lineales simples,
que es exactamente el caso de uso de SLSQP (constrained, gradiente
numerico esta bien dada la escala pequena del problema: 10 variables).
Una busqueda en grid tipo knapsack discreto (pesos en pasos de 5%) tambien
funcionaria, pero con 10 activos y tope de 25% el espacio de combinaciones
discretas crece rapido y SLSQP da el optimo continuo exacto sin ese costo.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import LinearConstraint, Bounds, minimize

BASE_DIR = Path("/Users/rod/Projects/stock-market/data/eval/forecast_demo")
BMV_PATH = BASE_DIR / "portfolio_bmv.json"
US_PATH = BASE_DIR / "portfolio_us.json"
PNG_PATH = BASE_DIR / "portfolio_frontera.png"
JSON_PATH = BASE_DIR / "portfolio_optimo.json"

PESO_MAX_TICKER = 0.25  # tope de concentracion por ticker (25% del presupuesto)
LAMBDA_DEFAULT = 0.5  # coeficiente de aversion al riesgo


def cargar_tickers() -> list[dict]:
    with open(BMV_PATH, encoding="utf-8") as f:
        bmv = json.load(f)
    with open(US_PATH, encoding="utf-8") as f:
        us = json.load(f)
    return bmv + us


def retorno_riesgo_portafolio(pesos: np.ndarray, retornos: np.ndarray, riesgos: np.ndarray) -> tuple[float, float]:
    """Retorno = suma ponderada. Riesgo = raiz de la suma ponderada al
    cuadrado de riesgo^2 (misma aproximacion de independencia / sin covarianza)."""
    retorno = float(np.sum(pesos * retornos))
    riesgo = float(math.sqrt(np.sum((pesos**2) * (riesgos**2))))
    return retorno, riesgo


def optimizar(retornos: np.ndarray, riesgos: np.ndarray, lam: float = LAMBDA_DEFAULT) -> np.ndarray:
    n = len(retornos)

    def objetivo_negativo(w: np.ndarray) -> float:
        # minimizamos el negativo del objetivo (equivalente a maximizar)
        return -(np.sum(w * retornos) - lam * np.sum((w**2) * (riesgos**2)))

    def grad_negativo(w: np.ndarray) -> np.ndarray:
        return -(retornos - 2.0 * lam * w * (riesgos**2))

    bounds = Bounds(lb=np.zeros(n), ub=np.full(n, PESO_MAX_TICKER))
    # sum(w_i) <= 1
    restriccion_presupuesto = LinearConstraint(np.ones((1, n)), lb=-np.inf, ub=1.0)

    # punto inicial factible: reparticion uniforme pequena
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

    w = np.clip(resultado.x, 0.0, PESO_MAX_TICKER)
    return w


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
    riesgos = np.array([a["riesgo_pct"] for a in activos])
    n = len(activos)

    # --- portafolio optimo ---
    w_opt = optimizar(retornos, riesgos, lam)
    ret_opt, riesgo_opt = retorno_riesgo_portafolio(w_opt, retornos, riesgos)

    # --- portafolio de referencia: igualmente ponderado (10% cada uno) ---
    w_eq = np.full(n, 1.0 / n)
    ret_eq, riesgo_eq = retorno_riesgo_portafolio(w_eq, retornos, riesgos)

    # --- portafolio de referencia: solo el mejor retorno esperado (100%) ---
    idx_mejor = int(np.argmax(retornos))
    w_best = np.zeros(n)
    w_best[idx_mejor] = 1.0
    ret_best, riesgo_best = retorno_riesgo_portafolio(w_best, retornos, riesgos)

    # --- tabla en consola ---
    print("=" * 60)
    print(f"Portafolio OPTIMO (lambda={lam}) — pesos > 1%")
    print("=" * 60)
    print(f"{'Ticker':<10}{'Peso (%)':>12}")
    for t, w in sorted(zip(tickers, w_opt), key=lambda x: -x[1]):
        if w > 0.01:
            print(f"{t:<10}{w * 100:>11.1f}%")

    print()
    print("=" * 60)
    print("Comparacion de portafolios (retorno y riesgo a 30 dias)")
    print("=" * 60)
    print(f"{'Portafolio':<28}{'Retorno (%)':>14}{'Riesgo (%)':>14}")
    print(f"{'Optimo (penalizado)':<28}{ret_opt:>14.2f}{riesgo_opt:>14.2f}")
    print(f"{'Igualmente ponderado':<28}{ret_eq:>14.2f}{riesgo_eq:>14.2f}")
    print(f"{'Solo mejor retorno (' + tickers[idx_mejor] + ')':<28}{ret_best:>14.2f}{riesgo_best:>14.2f}")

    # --- grafica scatter ---
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.scatter(riesgos, retornos, color="#1f6feb", s=70, zorder=3, label="Tickers individuales")
    for t, r, v in zip(tickers, riesgos, retornos):
        ax.annotate(t, (r, v), textcoords="offset points", xytext=(6, 4), fontsize=9, color="#333333")

    ax.scatter([riesgo_opt], [ret_opt], color="#C0392B", s=260, zorder=5,
               edgecolor="white", linewidth=1.2, label="Portafolio optimo")
    ax.scatter([riesgo_eq], [ret_eq], color="#888888", s=160, zorder=4,
               edgecolor="white", linewidth=1.0, label="Igualmente ponderado")

    ax.set_xlabel("Riesgo a 30 dias (%) — banda ~90%, ancho como % del precio")
    ax.set_ylabel("Retorno esperado a 30 dias (%)")
    ax.set_title("Frontera riesgo-retorno: 10 tickers y portafolios de referencia")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=150)
    plt.close(fig)

    # --- JSON de salida ---
    pesos_dict = {t: float(w) for t, w in zip(tickers, w_opt)}
    salida = {
        "pesos": pesos_dict,
        "retorno_portafolio_pct": ret_opt,
        "riesgo_portafolio_pct": riesgo_opt,
        "lambda": lam,
    }
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)

    print()
    print(f"Guardado: {PNG_PATH}")
    print(f"Guardado: {JSON_PATH}")


if __name__ == "__main__":
    main()
