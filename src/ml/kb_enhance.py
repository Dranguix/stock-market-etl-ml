"""
Moduladores y selección de features derivados de articles/knowledge/.

Implementa políticas LR-xxx de literature_rules.yaml:
  LR-003  noticias vs momentum técnico
  LR-007  bajo desacuerdo entre modelos / feature selection Kumbure
  LR-011  WTI vs sector energía/minería MX
  LR-012  consenso multi-horizonte (data/signals/*.json)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
SIGNALS_DIR = ROOT / "data" / "signals"

# Kumbure 2022 — indicadores técnicos + macro más citados en la literatura
KUMBURE_PRIORITY_BASE_COLS: tuple[str, ...] = (
    "ret_5d",
    "ret_20d",
    "rsi_14",
    "macd_hist",
    "macd_signal",
    "cs_ret5_rank",
    "cs_ret20_rank",
    "vol_ratio_20",
    "atr_pct",
    "price_vs_sma20",
    "obv_slope",
    "wti_ret5",
    "sp500_ret5",
    "usdmxn_ret5",
    "pe_ratio",
    "profit_margin",
    "fund_net_margin",
    "news_sent_score",
    "news_sent_delta",
    "news_count_7d",
)

# LR-011 — exposición commodities / energía BMV
ENERGY_SENSITIVE_TICKERS: frozenset[str] = frozenset({
    "VISTA", "GMEXICOB", "CEMEXCPO", "GCC", "ORBIA",
})

WTI_BEARISH_THRESHOLD_PCT = -3.0
HORIZON_CONSENSUS_STRONG = 3
HORIZON_CONSENSUS_WEAK = 1


def select_priority_features(
    available: list[str],
    *,
    include_news: bool = False,
    max_cols: int = 24,
) -> list[str]:
    """KB-022 / LR-007 — subset robusto de features para ensemble y xs_lgbm."""
    base = list(KUMBURE_PRIORITY_BASE_COLS)
    if not include_news:
        base = [c for c in base if not c.startswith("news_")]

    selected = [c for c in base if c in available]
    for c in available:
        if c not in selected and len(selected) < max_cols:
            selected.append(c)
    return selected or list(available)


def select_priority_rn_cols(rn_avail: list[str], max_cols: int = 24) -> list[str]:
    """Rank-normalized subset para xs_lgbm."""
    priority_rn = [f"rn_{c}" for c in KUMBURE_PRIORITY_BASE_COLS]
    selected = [c for c in priority_rn if c in rn_avail]
    for c in rn_avail:
        if c not in selected and len(selected) < max_cols:
            selected.append(c)
    return selected or list(rn_avail)


def load_horizon_consensus(signals_dir: Path | None = None) -> dict[str, int]:
    """
    LR-012 — máximo n_horizontes por ticker entre universos (1s–4s).
    Fuente: data/signals/{bmv,usa,...}.json → consensus[]
    """
    root = signals_dir or SIGNALS_DIR
    out: dict[str, int] = {}
    if not root.exists():
        return out

    for path in sorted(root.glob("*.json")):
        if path.name == "index.json":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.debug("consensus %s: %s", path.name, exc)
            continue
        for entry in data.get("consensus") or []:
            if not isinstance(entry, dict):
                continue
            ticker = entry.get("ticker")
            n_hz = int(entry.get("n_horizontes") or 0)
            if ticker:
                out[ticker] = max(out.get(ticker, 0), n_hz)
    return out


def get_wti_ret5_pct(macro: dict[str, pd.DataFrame]) -> float | None:
    """Retorno WTI 5d en % desde panel macro cargado."""
    wti = macro.get("WTI")
    if wti is None or wti.empty or "Close" not in wti.columns:
        return None
    close = wti["Close"].dropna()
    if len(close) < 6:
        return None
    ret = close.pct_change(5).iloc[-1]
    if not np.isfinite(ret):
        return None
    return float(ret * 100)


def _models_direction_disagree(signals: dict[str, float]) -> bool:
    """xs_lgbm vs ensemble en lados opuestos de 0.5 (Ghosh-style ranking)."""
    xs = signals.get("xs_lgbm")
    ens = signals.get("ensemble")
    if xs is None or ens is None:
        return False
    return (xs > 0.5) != (ens > 0.5)


def apply_kb_modulators(
    *,
    ticker: str,
    universe_slug: str,
    conviction: float,
    disagreement: float,
    signals: dict[str, float],
    news_sent: float | None,
    news_sent_delta: float | None = None,
    regime: str = "lateral",
    n_horizontes: int = 0,
    wti_ret5_pct: float | None = None,
) -> tuple[float, list[str], dict]:
    """
    Aplica moduladores LR sobre convicción ya ajustada por régimen/noticias.

    Returns
    -------
    conviction_adj, kb_flags, kb_meta
    """
    mult = 1.0
    flags: list[str] = []
    meta: dict = {"rules_applied": []}

    # LR-012 — consenso multi-horizonte
    if n_horizontes >= HORIZON_CONSENSUS_STRONG:
        mult *= 1.08
        flags.append(f"LR-012: consenso {n_horizontes}/4 horizontes → boost")
        meta["rules_applied"].append("LR-012")
    elif n_horizontes <= HORIZON_CONSENSUS_WEAK and conviction >= 0.55:
        mult *= 0.92
        flags.append(f"LR-012: señal focalizada ({n_horizontes}/4 horizontes)")
        meta["rules_applied"].append("LR-012")

    # LR-011 — WTI vs energía/minería
    if (
        ticker in ENERGY_SENSITIVE_TICKERS
        and wti_ret5_pct is not None
        and wti_ret5_pct < WTI_BEARISH_THRESHOLD_PCT
        and conviction > 0.5
    ):
        mult *= 0.75
        flags.append(
            f"LR-011: WTI {wti_ret5_pct:+.1f}% + long {ticker} → penalización"
        )
        meta["rules_applied"].append("LR-011")
        meta["wti_ret5_pct"] = round(wti_ret5_pct, 2)

    # LR-003 — noticias contradicen momentum técnico
    if news_sent is not None and conviction >= 0.55:
        if news_sent < -0.25:
            mult *= 0.75
            flags.append(f"LR-003: news_sent {news_sent:+.2f} contradice momentum")
            meta["rules_applied"].append("LR-003")
        elif news_sent < -0.15 and (news_sent_delta or 0) < -0.1:
            mult *= 0.85
            flags.append("LR-003: deterioro reciente de sentimiento")
            meta["rules_applied"].append("LR-003")

    # LR-007 — desacuerdo entre modelos
    if disagreement > 0.25:
        penalty = min(0.35, 0.5 * (disagreement - 0.15))
        mult *= 1.0 - penalty
        flags.append(f"LR-007: desacuerdo σ={disagreement:.3f}")
        meta["rules_applied"].append("LR-007")
    elif disagreement > 0.18:
        mult *= 0.92
        flags.append(f"LR-007: desacuerdo moderado σ={disagreement:.3f}")
        meta["rules_applied"].append("LR-007")

    if _models_direction_disagree(signals):
        mult *= 0.88
        flags.append("LR-001/007: xs_lgbm vs ensemble en direcciones opuestas")
        meta["rules_applied"].append("LR-001")

    # Bear regime — no amplificar sin consenso fuerte (literature_rules agent_never_do)
    if regime.lower() == "bear" and n_horizontes < HORIZON_CONSENSUS_STRONG and conviction > 0.6:
        mult *= 0.90
        flags.append("LR-009: risk-off sin consenso multi-H → cautela")
        meta["rules_applied"].append("LR-009")

    adj = float(np.clip(conviction * mult, 0.0, 1.5))
    meta["kb_multiplier"] = round(mult, 4)
    meta["n_horizontes"] = n_horizontes
    return adj, flags, meta
