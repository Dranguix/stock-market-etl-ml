"""Núcleo pasivo + satélite factorial. Los pesos los pone esta regla, no el LLM.

Diseño fijado una sola vez, por razones económicas y antes de ver resultados.
Cambiar cualquiera de estas constantes es una prueba nueva: suma uno al N de
data/eval/trial_log.jsonl y sube la barra del Deflated Sharpe.

  Núcleo 75%: SPY 50% + NAFTRAC 25%. La brecha contra SPY era exposición,
    no puntería; el núcleo captura el mercado.
  Satélite 25%: emisoras BMV + FIBRAs, equiponderado, 2-3 nombres (lo que
    deje el mínimo de ~100 MXN por orden con un libro de 50 USD).
    Score = rango de rentabilidad (margen bruto) 30% + valor (earnings
    yield, o dividend yield si no hay UPA) 30% + baja volatilidad 25% +
    momentum 12-1 escalado por volatilidad 15%.
    Histéresis: entra en el top 20%, sale solo al caer debajo del top 40%.
    Se reselecciona cada trimestre (ene/abr/jul/oct).
  Filtro de tendencia mensual (SMA de 10 meses ≈ 210 sesiones, banda 1%):
    un nombre del satélite debajo de su media sale a efectivo; un ETF del
    núcleo debajo de su media se queda a la mitad.
  Stop de catástrofe solo en el satélite: cierre 15% debajo del máximo
    desde la entrada, se vende en la apertura siguiente. Nada intramensual.
  Banda de deriva: solo se rebalancea un peso que se movió más de 5 pp, y
    se opera hasta el borde de la banda, no hasta el objetivo.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from src.agent.bmv_report import (
    _bars,
    _div_yield_ttm,
    _earnings_yield_ttm,
    _fundamentals_asof,
    _load_dividends,
    _load_fundamentals,
    _load_reported_eps,
)

CORE = {"SPY": ("SPY", 0.50), "NAFTRAC": ("NAFTRAC.MX", 0.25)}
SAT_WEIGHT = 0.25
SAT_MAX_NAMES = 3
FACTOR_WEIGHTS = {"rentabilidad": 0.30, "valor": 0.30, "baja_vol": 0.25, "momentum": 0.15}
ENTER_TOP = 0.20
EXIT_TOP = 0.40
TREND_DAYS = 210
TREND_BAND = 0.01
CORE_BEAR_CUT = 0.5
CATASTROPHE_DD = 0.15
DRIFT_BAND = 0.05
MIN_ORDER_MXN = 100.0
RESELECT_MONTHS = {1, 4, 7, 10}
TREND_MEMORY_MONTHS = 12
FX_SYMBOL = "MXN=X"


def config() -> dict:
    """Todo lo que define la regla, para el hash de la bitácora de pruebas."""
    return {
        "core": {k: w for k, (_, w) in CORE.items()},
        "sat_weight": SAT_WEIGHT, "sat_max_names": SAT_MAX_NAMES,
        "factor_weights": FACTOR_WEIGHTS, "enter_top": ENTER_TOP, "exit_top": EXIT_TOP,
        "trend_days": TREND_DAYS, "trend_band": TREND_BAND, "core_bear_cut": CORE_BEAR_CUT,
        "catastrophe_dd": CATASTROPHE_DD, "drift_band": DRIFT_BAND,
        "min_order_mxn": MIN_ORDER_MXN, "reselect_months": sorted(RESELECT_MONTHS),
        "trend_memory_months": TREND_MEMORY_MONTHS,
    }


def satellite_symbols() -> dict[str, str]:
    """Emisoras BMV + FIBRAs. Sin cripto (es una sola apuesta de beta a BTC, no
    un factor) ni acciones USA (eso ya lo da SPY)."""
    from src.data.universes import UNIVERSES

    out: dict[str, str] = {}
    for cfg in UNIVERSES.values():
        if cfg.get("slug") in ("bmv", "fibras"):
            out.update(cfg["tickers"])
    out.pop("NAFTRAC", None)
    return out


def is_mx(symbol: str) -> bool:
    return symbol.endswith(".MX")


@dataclass
class Inputs:
    fund_df: pd.DataFrame | None = None
    eps_all: dict = field(default_factory=dict)
    divs_all: dict = field(default_factory=dict)

    @classmethod
    def load(cls) -> "Inputs":
        return cls(_load_fundamentals(), _load_reported_eps(), _load_dividends())


def load_frames() -> dict[str, pd.DataFrame]:
    """Barras OHLC por nombre corto: núcleo, satélite y el tipo de cambio."""
    frames: dict[str, pd.DataFrame] = {}
    for name, (sym, _) in CORE.items():
        df = _bars(sym)
        if df is not None:
            frames[name] = df
    for name, sym in satellite_symbols().items():
        df = _bars(sym)
        if df is not None and len(df) > TREND_DAYS:
            frames[name] = df
    fx = _bars(FX_SYMBOL)
    if fx is not None:
        frames[FX_SYMBOL] = fx
    return frames


def symbol_of(name: str) -> str:
    if name in CORE:
        return CORE[name][0]
    return satellite_symbols().get(name, name)


def _rank(values: dict[str, float | None]) -> dict[str, float]:
    """Rango percentil centrado en 0 (-0.5..0.5). Un faltante queda neutro."""
    known = pd.Series({k: v for k, v in values.items() if v is not None and np.isfinite(v)}, dtype=float)
    if len(known) < 3:
        return {k: 0.0 for k in values}
    ranked = known.rank(pct=True) - 0.5
    return {k: float(ranked.get(k, 0.0)) for k in values}


def factor_table(frames: dict[str, pd.DataFrame], asof: pd.Timestamp, inputs: Inputs) -> pd.DataFrame:
    """Una fila por emisora del satélite con sus cuatro factores y el score.
    Solo usa datos con fecha <= asof (fundamentales por available_date)."""
    raw: dict[str, dict] = {}
    for name, sym in satellite_symbols().items():
        df = frames.get(name)
        if df is None:
            continue
        close = df["Close"].loc[:asof].dropna()
        if len(close) < 253 or close.index[-1] < asof - pd.Timedelta(days=10):
            continue
        px = float(close.iloc[-1])
        rets = close.pct_change().tail(252)
        vol = float(rets.std() * np.sqrt(252))
        if not np.isfinite(vol) or vol <= 0:
            continue
        mom = float(close.iloc[-21] / close.iloc[-252] - 1) / vol
        fund = _fundamentals_asof(inputs.fund_df, name, asof)
        value = _earnings_yield_ttm(sym, px, asof, inputs.eps_all)
        if value is None:
            value = _div_yield_ttm(name, px, asof, inputs.divs_all)
        raw[name] = {
            "rentabilidad": fund.get("gross_margin"),
            "valor": value,
            "baja_vol": -vol,
            "momentum": mom,
            "close": px,
        }
    if not raw:
        return pd.DataFrame()
    ranks = {f: _rank({t: r[f] for t, r in raw.items()}) for f in FACTOR_WEIGHTS}
    rows = []
    for t, r in raw.items():
        score = sum(w * ranks[f][t] for f, w in FACTOR_WEIGHTS.items())
        rows.append({"ticker": t, **{f: r[f] for f in FACTOR_WEIGHTS}, "close": r["close"], "score": score})
    table = pd.DataFrame(rows).set_index("ticker").sort_values("score", ascending=False)
    table["rank_pct"] = np.arange(1, len(table) + 1) / len(table)
    return table


def in_trend(close: pd.Series, asof: pd.Timestamp, was_in: bool = True) -> bool:
    """Precio contra su media de 10 meses, con banda para no entrar y salir por ruido.
    Sin historia suficiente se queda como estaba."""
    hist = close.loc[:asof].dropna()
    if len(hist) < TREND_DAYS:
        return was_in
    sma = float(hist.tail(TREND_DAYS).mean())
    px = float(hist.iloc[-1])
    if was_in:
        return px >= sma * (1 - TREND_BAND)
    return px > sma * (1 + TREND_BAND)


def trend_asof(close: pd.Series, asof: pd.Timestamp) -> bool:
    """Estado de la tendencia reconstruido de los cierres de mes previos, para que
    el backtest y el job diario (que no guarda estado) decidan lo mismo."""
    hist = close.loc[:asof].dropna()
    if hist.empty:
        return True
    month_ends = hist.groupby(hist.index.to_period("M")).tail(1).index
    checks = [d for d in month_ends if d < hist.index[-1]][-TREND_MEMORY_MONTHS:] + [hist.index[-1]]
    state = True
    for day in checks:
        state = in_trend(close, day, state)
    return state


def fx_on(frames: dict[str, pd.DataFrame], asof: pd.Timestamp, col: str = "Close") -> float:
    fx = frames[FX_SYMBOL][col].loc[:asof].dropna()
    return float(fx.iloc[-1])


def min_order_usd(fx: float) -> float:
    return MIN_ORDER_MXN / fx


def n_satellite(equity_usd: float, fx: float) -> int:
    """Cuántos nombres caben en el satélite sin bajar del mínimo por orden."""
    return max(0, min(SAT_MAX_NAMES, int(equity_usd * SAT_WEIGHT // min_order_usd(fx))))


def is_rebalance_day(session: date, next_session: date) -> bool:
    """Se decide al cierre de la última sesión del mes y se ejecuta en la apertura siguiente."""
    return session.month != next_session.month


@dataclass
class Decision:
    asof: str
    weights: dict[str, float]
    satellite: list[str]
    reselected: bool
    trend: dict[str, bool]
    vetoed: dict[str, str]
    candidates: list[str]
    motivo: str

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def decide(
    frames: dict[str, pd.DataFrame],
    asof: pd.Timestamp,
    inputs: Inputs,
    equity_usd: float,
    held_satellite: list[str],
    reselect: bool | None = None,
    veto=None,
) -> Decision:
    """Pesos objetivo del libro al cierre de asof.

    veto: función opcional (candidatos, asof) -> {ticker: motivo}. Solo puede
    quitar nombres; el siguiente del ranking toma su lugar. Es el único papel
    del LLM en la asignación."""
    trend_state: dict[str, bool] = {}
    if reselect is None:
        reselect = (asof + pd.offsets.BDay(1)).month in RESELECT_MONTHS
    fx = fx_on(frames, asof)

    weights: dict[str, float] = {}
    for name, (_, w) in CORE.items():
        df = frames.get(name)
        if df is None:
            continue
        ok = trend_asof(df["Close"], asof)
        trend_state[name] = ok
        weights[name] = w if ok else w * CORE_BEAR_CUT

    table = factor_table(frames, asof, inputs)
    n = n_satellite(equity_usd, fx)
    slot_usd = equity_usd * SAT_WEIGHT / n if n else 0.0

    def affordable(t: str) -> bool:
        # BMV es en unidades enteras: una acción tiene que caber en el slot.
        px = float(table.loc[t, "close"])
        return (px / fx if is_mx(symbol_of(t)) else px) <= slot_usd

    def trending(t: str) -> bool:
        ok = trend_asof(frames[t]["Close"], asof)
        trend_state[t] = ok
        return ok

    keep, candidates = [], []
    if not table.empty and n:
        for t in held_satellite:
            if t not in table.index:
                continue
            within = table.loc[t, "rank_pct"] <= (EXIT_TOP if reselect else 1.0)
            if within and trending(t):
                keep.append(t)
        if reselect:
            candidates = [
                t for t in table.index
                if t not in keep and table.loc[t, "rank_pct"] <= ENTER_TOP and affordable(t) and trending(t)
            ]

    vetoed: dict[str, str] = {}
    if veto is not None and (keep or candidates):
        vetoed = veto(keep + candidates, asof) or {}
    keep = [t for t in keep if t not in vetoed]
    keep = sorted(keep, key=lambda t: table.loc[t, "score"], reverse=True)[:n]
    chosen = keep + [t for t in candidates if t not in vetoed][: max(0, n - len(keep))]

    for t in chosen:
        weights[t] = SAT_WEIGHT / n

    motivo = (
        f"núcleo {', '.join(f'{k} {weights.get(k, 0)*100:.0f}%' for k in CORE)}; "
        f"satélite {', '.join(chosen) or 'efectivo'}"
        + (f"; vetados {', '.join(vetoed)}" if vetoed else "")
        + ("; reselección trimestral" if reselect else "; solo tendencia y deriva")
    )
    return Decision(
        asof=str(pd.Timestamp(asof).date()),
        weights={k: round(v, 4) for k, v in weights.items()},
        satellite=chosen,
        reselected=reselect,
        trend=trend_state,
        vetoed=vetoed,
        candidates=candidates,
        motivo=motivo,
    )


def apply_drift_band(target: dict[str, float], current: dict[str, float]) -> dict[str, float]:
    """Mueve solo lo que se salió de la banda, y hasta el borde de la banda.
    Entradas y salidas completas (objetivo 0 o posición nueva) sí van completas."""
    out: dict[str, float] = {}
    for t in set(target) | set(current):
        tgt, cur = target.get(t, 0.0), current.get(t, 0.0)
        if tgt == 0.0 or cur == 0.0:
            out[t] = tgt
        elif cur > tgt + DRIFT_BAND:
            out[t] = tgt + DRIFT_BAND
        elif cur < tgt - DRIFT_BAND:
            out[t] = tgt - DRIFT_BAND
        else:
            out[t] = cur
    return {t: w for t, w in out.items() if w > 0}
