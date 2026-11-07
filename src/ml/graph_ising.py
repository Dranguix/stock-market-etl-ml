"""
Graph-Ising Market — autómata celular sobre grafo de influencia con agentes WoE.

Evolución de src/ml/ising_market.py (rejilla 2D) hacia la topología real de un
mercado: la vecindad no es geográfica sino INFORMACIONAL.

  Nodos    = tickers del universo (cada nodo agrega a "los traders de esa emisora")
  Aristas  = correlación rolling de retornos (point-in-time), top-k vecinos
  Estado   = spin continuo s_i ∈ [-1, 1] (presión neta compra/venta)
  Campo    = evidencia WoE real del día (RSI, momentum, volumen, news, flow…)

Dinámica de campo medio (relajación amortiguada):

    s_i ← (1-γ)·s_i + γ·tanh( (h_i + J·Σ_j w_ij·s_j) / T )

Calibración: J se fija cerca del punto crítico de campo medio del grafo,
J·λ_max(W)/T = crit_frac < 1 (subcrítico) — donde la susceptibilidad es máxima
y la dinámica colectiva es informativa sin congelarse en consenso trivial.

Features emergentes por nodo (GIM_FEATURE_COLS):
    gim_pressure       — spin de equilibrio (presión compradora emergente)
    gim_susceptibility — sensibilidad del nodo a shocks propios y de vecinos
    gim_cascade        — alineación ponderada con el vecindario (riesgo cascada)
    gim_phase          — fase global codificada (-1 ferro / 0 crítico / +1 para)

Anti-lookahead: los mapas WoE se ajustan solo con filas cuyo label forward ya
era observable en t (gap = horizon+1) y se re-ajustan cada `woe_refit_every`
días con ventana expansiva. El grafo usa solo retornos hasta t.

Uso:
    from src.ml.graph_ising import compute_gim_panel, GIM_FEATURE_COLS

    panel = compute_gim_panel(frames, evidence_cols=["rsi_14", "ret_5d", ...])
    # → DataFrame(date, ticker, gim_pressure, gim_susceptibility, gim_cascade, gim_phase)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.ml.ising_market import classify_phase
from src.ml.woe import FeatureWoE, compute_all_woe

logger = logging.getLogger(__name__)

GIM_FEATURE_COLS = [
    "gim_pressure",
    "gim_susceptibility",
    "gim_cascade",
    "gim_phase",
]

_PHASE_CODE = {"ferromagnetic": -1.0, "critical": 0.0, "paramagnetic": 1.0}

# Evidencia default: técnicos Kumbure + news + order flow si están disponibles
DEFAULT_EVIDENCE_COLS = [
    "rsi_14",
    "ret_5d",
    "ret_20d",
    "price_vs_sma20",
    "vol_ratio_20",
    "cs_mom_rank",
    "news_sent_score",
    "flow_imbalance",
    "flow_smart",
]


@dataclass
class GraphIsingConfig:
    corr_window: int = 63       # ventana rolling del grafo de correlación
    n_neighbors: int = 8        # top-k vecinos por nodo
    crit_frac: float = 0.90     # J·λ_max/T — subcrítico cerca de la transición
    T: float = 1.0              # temperatura (ruido de agentes)
    n_steps: int = 40           # pasos de relajación
    damping: float = 0.5        # amortiguación γ
    evidence_scale: float = 0.6 # escala del campo WoE (log-odds → campo)
    evidence_clip: float = 3.0  # clip de la suma de WoE por nodo
    woe_refit_every: int = 21   # días hábiles entre refits de WoE
    min_train_rows: int = 400   # mínimo de filas para ajustar WoE
    horizon: int = 5            # horizonte del label forward para WoE


class GraphIsingMarket:
    """Autómata Ising sobre grafo ponderado con campo de evidencia WoE."""

    def __init__(self, config: GraphIsingConfig | None = None):
        self.cfg = config or GraphIsingConfig()

    # ── Grafo de influencia ───────────────────────────────────────────────

    def build_graph(self, returns_window: pd.DataFrame) -> np.ndarray:
        """
        Matriz de pesos W (n×n) desde correlaciones de retornos.

        Solo correlaciones positivas (influencia de manada); top-k por nodo;
        filas normalizadas a suma 1 para que J controle la fuerza total.
        """
        corr = returns_window.corr().to_numpy()
        n = corr.shape[0]
        np.fill_diagonal(corr, 0.0)
        corr = np.clip(np.nan_to_num(corr), 0.0, None)

        k = min(self.cfg.n_neighbors, max(n - 1, 1))
        W = np.zeros_like(corr)
        for i in range(n):
            if corr[i].max() <= 0:
                continue
            top = np.argsort(corr[i])[-k:]
            W[i, top] = corr[i, top]

        W = np.maximum(W, W.T)  # simétrico: la influencia es mutua
        row_sums = W.sum(axis=1, keepdims=True)
        W = np.divide(W, row_sums, out=np.zeros_like(W), where=row_sums > 0)
        return W

    def coupling(self, W: np.ndarray) -> float:
        """J calibrado al punto crítico de campo medio: J·λ_max(W)/T = crit_frac."""
        if W.size == 0 or not np.any(W):
            return 0.0
        lam_max = float(np.max(np.abs(np.linalg.eigvals(W)).real))
        if lam_max <= 1e-9:
            return 0.0
        return self.cfg.crit_frac * self.cfg.T / lam_max

    # ── Dinámica ──────────────────────────────────────────────────────────

    def relax(self, W: np.ndarray, h: np.ndarray, J: float) -> np.ndarray:
        """Relajación de campo medio amortiguada hasta (casi) equilibrio."""
        T = max(self.cfg.T, 1e-8)
        g = self.cfg.damping
        s = np.tanh(h / T)
        for _ in range(self.cfg.n_steps):
            field = h + J * (W @ s)
            s_new = (1 - g) * s + g * np.tanh(field / T)
            if np.max(np.abs(s_new - s)) < 1e-6:
                s = s_new
                break
            s = s_new
        return s

    def susceptibility(self, W: np.ndarray, s: np.ndarray, J: float) -> np.ndarray:
        """
        χ_i ≈ ds_i/dh_i a segundo orden de campo medio:
        χ_i = a_i·(1 + J·Σ_j w_ij·a_j),  a_i = (1 - s_i²)/T.
        """
        a = (1.0 - s**2) / max(self.cfg.T, 1e-8)
        return a * (1.0 + J * (W @ a))

    @staticmethod
    def cascade(W: np.ndarray, s: np.ndarray) -> np.ndarray:
        """Alineación ponderada con vecinos: s_i·Σ_j w_ij·s_j ∈ [-1, 1]."""
        return s * (W @ s)


# ─────────────────────────────────────────────────────────────────────────────
# Panel point-in-time
# ─────────────────────────────────────────────────────────────────────────────

def _evidence_field(
    row_block: pd.DataFrame,
    woe_maps: dict[str, FeatureWoE],
    cols: list[str],
    cfg: GraphIsingConfig,
) -> np.ndarray:
    """Suma de WoE (log-odds) por ticker → campo externo h."""
    total = pd.Series(0.0, index=row_block.index)
    for c in cols:
        fw = woe_maps.get(c)
        if fw is None or fw.n_bins == 0 or c not in row_block.columns:
            continue
        total = total.add(fw.transform(row_block[c]).fillna(0.0), fill_value=0.0)
    h = total.clip(-cfg.evidence_clip, cfg.evidence_clip) * cfg.evidence_scale
    return h.to_numpy(dtype=float)


def compute_gim_panel(
    frames: dict[str, pd.DataFrame],
    evidence_cols: list[str] | None = None,
    config: GraphIsingConfig | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Features Graph-Ising diarias point-in-time para un universo.

    Parameters
    ----------
    frames : dict ticker → DataFrame diario con DatetimeIndex, columna Close
             y las columnas de evidencia disponibles.
    evidence_cols : columnas de evidencia; default intersección con
             DEFAULT_EVIDENCE_COLS.
    """
    cfg = config or GraphIsingConfig()
    gim = GraphIsingMarket(cfg)

    tickers = sorted(frames.keys())
    if len(tickers) < 5:
        return pd.DataFrame(columns=["date", "ticker", *GIM_FEATURE_COLS])

    closes = pd.DataFrame({t: frames[t]["Close"] for t in tickers}).sort_index()
    closes.index = pd.to_datetime(closes.index)
    if getattr(closes.index, "tz", None) is not None:
        closes.index = closes.index.tz_localize(None)
    returns = closes.pct_change(fill_method=None)

    avail_cols: set[str] = set()
    for t in tickers:
        avail_cols |= set(frames[t].columns)
    cols = [c for c in (evidence_cols or DEFAULT_EVIDENCE_COLS) if c in avail_cols]
    if not cols:
        logger.warning("graph_ising: sin columnas de evidencia disponibles")
        return pd.DataFrame(columns=["date", "ticker", *GIM_FEATURE_COLS])

    # Panel largo de evidencia: (date, ticker) → cols  [+ label forward p/ WoE]
    ev_rows = []
    for t in tickers:
        df = frames[t]
        idx = pd.to_datetime(df.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        sub = df.reindex(df.index)[[c for c in cols if c in df.columns]].copy()
        sub.index = idx
        sub["ticker"] = t
        fwd = closes[t].shift(-cfg.horizon) / closes[t] - 1.0
        sub["_y_fwd"] = (fwd.reindex(idx) > 0).astype(float).where(
            fwd.reindex(idx).notna(),
        )
        ev_rows.append(sub.reset_index(names="date"))
    ev = pd.concat(ev_rows, ignore_index=True).sort_values("date")
    ev_by_date = {d: g for d, g in ev.groupby("date")}

    all_dates = closes.index[cfg.corr_window:]
    woe_maps: dict[str, FeatureWoE] = {}
    last_refit = -10**9

    records: list[dict] = []
    for di, day in enumerate(all_dates):
        # Refit WoE (PIT: labels observables → date <= day - horizon - 1)
        if di - last_refit >= cfg.woe_refit_every:
            cutoff = day - pd.tseries.offsets.BDay(cfg.horizon + 1)
            train = ev[(ev["date"] <= cutoff) & ev["_y_fwd"].notna()]
            if len(train) >= cfg.min_train_rows:
                train = train.copy()
                train["_y"] = train["_y_fwd"].astype(int)
                woe_maps = compute_all_woe(train, cols, target_col="_y")
                last_refit = di
        if not woe_maps:
            continue

        # Grafo con retornos hasta hoy inclusive
        win = returns.loc[:day].tail(cfg.corr_window)
        active = [t for t in tickers if win[t].notna().sum() >= cfg.corr_window // 2]
        if len(active) < 5:
            continue
        W = gim.build_graph(win[active])
        J = gim.coupling(W)

        # Evidencia del día
        today = ev_by_date.get(day)
        if today is None:
            continue
        today = today.drop_duplicates(subset="ticker").set_index("ticker").reindex(active)
        h = _evidence_field(today, woe_maps, cols, cfg)
        h = np.nan_to_num(h)

        s = gim.relax(W, h, J)
        chi = gim.susceptibility(W, s, J)
        casc = gim.cascade(W, s)
        M = float(np.mean(s))
        phase = _PHASE_CODE[classify_phase(M, J, cfg.T)]

        for i, t in enumerate(active):
            records.append({
                "date": day,
                "ticker": t,
                "gim_pressure": float(s[i]),
                "gim_susceptibility": float(chi[i]),
                "gim_cascade": float(casc[i]),
                "gim_phase": phase,
            })
        if verbose and di % 100 == 0:
            print(f"  gim {day.date()}: M={M:+.3f} J={J:.3f} fase={phase:+.0f}")

    return pd.DataFrame(records, columns=["date", "ticker", *GIM_FEATURE_COLS])


def add_gim_features(
    feat_df: pd.DataFrame,
    ticker: str,
    gim_panel: pd.DataFrame | None,
) -> pd.DataFrame:
    """Merge del panel GIM a un frame diario de un ticker (index fecha)."""
    df = feat_df.copy()
    if gim_panel is None or gim_panel.empty:
        for c in GIM_FEATURE_COLS:
            df[c] = 0.0
        return df

    sub = gim_panel[gim_panel["ticker"] == ticker]
    if sub.empty:
        for c in GIM_FEATURE_COLS:
            df[c] = 0.0
        return df

    sub = sub.set_index(pd.to_datetime(sub["date"]))[GIM_FEATURE_COLS]
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    for c in GIM_FEATURE_COLS:
        df[c] = sub[c].reindex(idx).to_numpy()
        df[c] = df[c].astype(float).fillna(0.0)
    return df
