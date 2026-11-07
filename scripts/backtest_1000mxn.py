"""Backtest con DINERO REAL: ¿qué pasa si de verdad metes $1,000 MXN al
portafolio óptimo de scripts/portfolio_optimizer_cov.py?

Dos capas, porque el problema NO es solo el costo de transacción:

1. "Lote mínimo" — BMV/GBM solo deja comprar ACCIONES ENTERAS, con un
   mínimo ~100 MXN por orden (ver docs/SISTEMA_END_TO_END.md). Con $1,000
   repartidos en 10 tickers (algunos a 25%, otros a <1%), varias posiciones
   del peso "óptimo" son IMPOSIBLES de comprar: no hay ni para 1 acción.
   Los tickers en USD (AAPL, JNJ aquí) convierten a pesos a tipo de cambio
   real — y sus precios por acción son altísimos vs una asignación de
   $250 MXN.

2. "Costos" — comisión + slippage round-trip. BMV: 0.55% documentado en
   articles/knowledge/agent/bmv_market_rules.yaml (KB-003). USA/FX: no hay
   dato real medido en este repo todavía — se usa un estimado de 1.0%
   round-trip (comisión + spread FX) y se marca EXPLÍCITAMENTE como
   supuesto a validar con el broker real, no un número medido.

Luego corre el walk-forward de 5 fechas de corte (mismo patrón que
scripts/portfolio_backtest.py) pero SOLO sobre lo que de verdad se puede
comprar con $1,000, neto de costos, comparado contra el retorno "de papel"
(ignorando lote mínimo y costos) para cuantificar la brecha.

Uso:
    python scripts/backtest_1000mxn.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src.data.store import get as store_get

CAPITAL_MXN = 1000.0
MIN_ORDEN_MXN = 100.0  # docs/SISTEMA_END_TO_END.md: "mínimo de ~100 MXN por orden"

COSTO_ROUNDTRIP_BMV = 0.0055   # KB-003, medido/estimado retail BMV
COSTO_ROUNDTRIP_USA = 0.010    # SUPUESTO sin validar: comisión + spread FX, no hay dato real

TICKERS_USD = {"AAPL", "JPM", "XOM", "JNJ"}  # SPY también, pero no entra con peso>0 aquí

PORTFOLIO_PATH = ROOT / "data" / "eval" / "forecast_demo" / "portfolio_optimo_cov.json"
OUT_JSON = ROOT / "data" / "eval" / "forecast_demo" / "backtest_1000mxn.json"


def fx_mxn_por_usd() -> float:
    df = store_get("USDMXN")
    return float(df["Close"].iloc[-1])


def precio_mxn(ticker: str, fx: float, df=None) -> float:
    if df is None:
        df = store_get(ticker)
    precio = float(df["Close"].iloc[-1])
    return precio * fx if ticker in TICKERS_USD else precio


def paso1_lote_minimo() -> list[dict]:
    """¿Cuánto de cada ticker se puede REALMENTE comprar con $1,000 MXN,
    dado que solo se compran acciones enteras?"""
    with open(PORTFOLIO_PATH, encoding="utf-8") as f:
        data = json.load(f)
    pesos = {t: w for t, w in data["pesos"].items() if w > 0.001}

    fx = fx_mxn_por_usd()
    filas = []
    for ticker, peso in sorted(pesos.items(), key=lambda kv: -kv[1]):
        objetivo_mxn = CAPITAL_MXN * peso
        precio = precio_mxn(ticker, fx)
        n_acciones = int(objetivo_mxn // precio)
        gastado_mxn = n_acciones * precio
        comprable = n_acciones >= 1 and gastado_mxn >= MIN_ORDEN_MXN
        filas.append({
            "ticker": ticker,
            "peso_objetivo_pct": round(peso * 100, 2),
            "objetivo_mxn": round(objetivo_mxn, 2),
            "precio_mxn": round(precio, 2),
            "moneda_origen": "USD" if ticker in TICKERS_USD else "MXN",
            "acciones_enteras_comprables": n_acciones,
            "gastado_mxn": round(gastado_mxn, 2),
            "comprable": comprable,
        })
    return filas, fx


def imprimir_paso1(filas: list[dict], fx: float) -> dict[str, float]:
    print(f"\n=== PASO 1: lote mínimo con ${CAPITAL_MXN:.0f} MXN (FX={fx:.2f} MXN/USD) ===\n")
    total_gastado = 0.0
    pesos_reales: dict[str, float] = {}
    for f in filas:
        estado = "OK" if f["comprable"] else "NO ALCANZA"
        print(
            f"  {f['ticker']:10} peso obj {f['peso_objetivo_pct']:5.1f}% "
            f"(${f['objetivo_mxn']:7.2f} MXN) | precio ${f['precio_mxn']:9.2f} MXN "
            f"({f['moneda_origen']}) | {f['acciones_enteras_comprables']} acciones "
            f"= ${f['gastado_mxn']:7.2f} MXN  [{estado}]"
        )
        if f["comprable"]:
            total_gastado += f["gastado_mxn"]
            pesos_reales[f["ticker"]] = f["gastado_mxn"]

    pct_desplegado = total_gastado / CAPITAL_MXN * 100
    print(
        f"\n  Total desplegado de verdad: ${total_gastado:.2f} MXN de ${CAPITAL_MXN:.0f} "
        f"({pct_desplegado:.1f}%). Resto queda en efectivo sin invertir por el lote mínimo."
    )
    no_comprables = [f["ticker"] for f in filas if not f["comprable"]]
    if no_comprables:
        print(f"  Tickers con peso >0 pero IMPOSIBLES de comprar: {', '.join(no_comprables)}")
    return pesos_reales


def paso2_costos_y_backtest(pesos_reales_mxn: dict[str, float]) -> dict:
    """Retorno realizado a 21 sesiones, bruto vs neto de comisión+slippage,
    SOLO sobre lo que de verdad se compró (paso 1), en 5 fechas de corte."""
    from scripts.forecast_horizons import MIN_TRAIN_ROWS
    from src.ml.features import compute_features

    HORIZON = 21
    N_CUTOFFS = 5
    SPACING = 60

    tickers = list(pesos_reales_mxn)
    total_comprado = sum(pesos_reales_mxn.values())
    pesos_norm = {t: v / total_comprado for t, v in pesos_reales_mxn.items()}

    dfs = {t: store_get(t) for t in tickers}
    feats = {t: compute_features(dfs[t]) for t in tickers}

    n_min = min(len(feats[t]) for t in tickers)
    idx_max = n_min - 1 - HORIZON
    idx_min = MIN_TRAIN_ROWS + (N_CUTOFFS - 1) * SPACING
    if idx_max - (N_CUTOFFS - 1) * SPACING < idx_min:
        raise RuntimeError("No hay suficiente historia para 5 cortes espaciados")
    cortes_idx = sorted(idx_max - k * SPACING for k in range(N_CUTOFFS))

    print(f"\n=== PASO 2: retorno realizado 21 sesiones, bruto vs neto de costos ===\n")
    resultados = []
    for idx in cortes_idx:
        fecha = feats[tickers[0]].index[idx]
        ret_portafolio_bruto = 0.0
        for t in tickers:
            feat_t = feats[t]
            p0 = float(feat_t["Close"].iloc[idx])
            p1 = float(feat_t["Close"].iloc[idx + HORIZON])
            ret_t = (p1 / p0 - 1)
            ret_portafolio_bruto += pesos_norm[t] * ret_t

        costo_round_trip = sum(
            pesos_norm[t] * (COSTO_ROUNDTRIP_USA if t in TICKERS_USD else COSTO_ROUNDTRIP_BMV)
            for t in tickers
        )
        ret_neto = ret_portafolio_bruto - costo_round_trip
        resultados.append({
            "fecha": str(fecha.date()),
            "ret_bruto_pct": round(ret_portafolio_bruto * 100, 3),
            "costo_pct": round(costo_round_trip * 100, 3),
            "ret_neto_pct": round(ret_neto * 100, 3),
        })
        print(
            f"  corte {fecha.date()}: bruto {ret_portafolio_bruto*100:+.2f}%  "
            f"- costo {costo_round_trip*100:.2f}%  = neto {ret_neto*100:+.2f}%"
        )

    import numpy as np
    brutos = np.array([r["ret_bruto_pct"] for r in resultados])
    netos = np.array([r["ret_neto_pct"] for r in resultados])
    print(
        f"\n  Promedio bruto: {brutos.mean():+.2f}%  |  Promedio neto: {netos.mean():+.2f}%  "
        f"|  Costo promedio se come: {brutos.mean() - netos.mean():.2f} pp por ciclo de 21 sesiones"
    )
    return {"pesos_usados_pct": {t: round(v * 100, 1) for t, v in pesos_norm.items()}, "cortes": resultados}


def main() -> None:
    filas, fx = paso1_lote_minimo()
    pesos_reales_mxn = imprimir_paso1(filas, fx)

    if not pesos_reales_mxn:
        print("\nNingún ticker del portafolio óptimo es comprable con $1,000 MXN. Fin.")
        return

    resultado_backtest = paso2_costos_y_backtest(pesos_reales_mxn)

    salida = {
        "capital_mxn": CAPITAL_MXN,
        "min_orden_mxn": MIN_ORDEN_MXN,
        "fx_mxn_por_usd": round(fx, 4),
        "paso1_lote_minimo": filas,
        "paso2_backtest": resultado_backtest,
        "nota_honestidad": (
            "Costo BMV (0.55% round-trip) viene de bmv_market_rules.yaml (KB-003, "
            "'validar con broker real'). Costo USA/FX (1.0% round-trip) es un SUPUESTO "
            "no medido — falta dato real del broker (GBM SIC u otro) antes de confiar en "
            "este número. El lote mínimo (acciones enteras, ~100 MXN/orden) SÍ es una regla "
            "documentada del mercado, no un supuesto."
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)
    print(f"\nGuardado: {OUT_JSON}")


if __name__ == "__main__":
    main()
