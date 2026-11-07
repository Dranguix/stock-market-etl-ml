"""
Order flow de brokers (casa compra / casa vende) — features point-in-time.

Fuente: data/microstructure/hechos/YYYY-MM-DD.parquet (scripts/backfill_hechos.py)
  columnas: date, ticker, casa, vol_buy, vol_sell, imp_buy, imp_sell, n_buy, n_sell

Nota estructural: por cada hecho hay exactamente un comprador y un vendedor,
así que el volumen agregado comprado == vendido por ticker. La señal NO está
en el volumen neto total (siempre 0) sino en la ASIMETRÍA entre casas:

  flow_imbalance     — neto de la casa dominante compradora + neto de la
                       dominante vendedora, / volumen total. Positivo = hay un
                       acumulador que domina sobre el mayor distribuidor.
  flow_concentration — HHI del lado compra menos HHI del lado venta. Positivo
                       = compra concentrada (institución acumulando de muchos
                       vendedores dispersos).
  flow_smart         — flujo neto ponderado por el "edge" histórico de cada
                       casa (qué tan predictivo fue su flujo neto del retorno
                       forward 5d, ventana rolling PIT sin lookahead).
  flow_persistence   — media móvil 5d de flow_imbalance (acumulación sostenida).

Uso:
    from src.ml.flow_features import FLOW_FEATURE_COLS, compute_flow_panel

    panel = compute_flow_panel(tickers, close_panel=closes)  # date, ticker, flow_*
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
HECHOS_DIR = ROOT / "data" / "microstructure" / "hechos"

FLOW_FEATURE_COLS = [
    "flow_imbalance",
    "flow_concentration",
    "flow_smart",
    "flow_persistence",
]

# Edge de casa: ventana rolling y gap anti-lookahead (horizonte fwd 5d + 1)
SMART_WINDOW_DAYS = 120
SMART_GAP_DAYS = 6
SMART_FWD_HORIZON = 5


def available_flow_dates(hechos_dir: Path | None = None) -> list[str]:
    d = hechos_dir or HECHOS_DIR
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.parquet"))


def load_flow_raw(
    tickers: list[str] | None = None,
    hechos_dir: Path | None = None,
) -> pd.DataFrame:
    """Concatena los parquets diarios (date, ticker, casa, vol/imp por lado)."""
    d = hechos_dir or HECHOS_DIR
    frames = []
    for path in sorted(d.glob("*.parquet")) if d.exists() else []:
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            logger.warning("flow %s: %s", path.name, exc)
            continue
        if tickers is not None:
            df = df[df["ticker"].isin(tickers)]
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(
            columns=["date", "ticker", "casa", "vol_buy", "vol_sell",
                     "imp_buy", "imp_sell", "n_buy", "n_sell"],
        )
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out


def _daily_flow_stats(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Por (date, ticker): imbalance de casa dominante, HHI compra-venta y
    netshare por casa (para flow_smart).
    """
    raw = raw.copy()
    raw["net"] = raw["vol_buy"] - raw["vol_sell"]

    g = raw.groupby(["date", "ticker"])
    vol_total = g["vol_buy"].sum()  # == vol_sell total por construcción

    net_max = g["net"].max()
    net_min = g["net"].min()
    imbalance = (net_max + net_min) / vol_total.replace(0, np.nan)

    def _hhi(col: str) -> pd.Series:
        tot = g[col].sum().replace(0, np.nan)
        share_sq = (
            raw.set_index(["date", "ticker"])[col]
            .div(tot)
            .pow(2)
        )
        return share_sq.groupby(["date", "ticker"]).sum()

    hhi_diff = _hhi("vol_buy") - _hhi("vol_sell")

    stats = pd.DataFrame({
        "flow_imbalance": imbalance,
        "flow_concentration": hhi_diff,
        "vol_total": vol_total,
    }).reset_index()
    return stats


def _casa_netshare(raw: pd.DataFrame) -> pd.DataFrame:
    """(date, ticker, casa) → netshare = (vol_buy - vol_sell) / vol total del día."""
    raw = raw.copy()
    raw["net"] = raw["vol_buy"] - raw["vol_sell"]
    tot = raw.groupby(["date", "ticker"])["vol_buy"].transform("sum")
    raw["netshare"] = raw["net"] / tot.replace(0, np.nan)
    return raw[["date", "ticker", "casa", "netshare"]].dropna(subset=["netshare"])


def _smart_flow(
    netshare: pd.DataFrame,
    close_panel: pd.DataFrame | None,
) -> pd.Series:
    """
    flow_smart por (date, ticker): Σ_casa edge_casa(t) · netshare_casa.

    edge_casa(t) = media (ponderada por |netshare|) del retorno fwd 5d de los
    tickers donde la casa fue neta compradora/vendedora, sobre la ventana
    [t-126, t-6]. El gap de 6 días garantiza que solo usa retornos ya
    realizados al momento t (sin lookahead).
    """
    idx = pd.MultiIndex.from_frame(netshare[["date", "ticker"]]).unique()
    zero = pd.Series(0.0, index=idx)
    if close_panel is None or close_panel.empty:
        return zero

    closes = close_panel.copy()
    closes.index = pd.to_datetime(closes.index)
    fwd = closes.shift(-SMART_FWD_HORIZON) / closes - 1.0

    ns = netshare.copy()
    ns["fwd_ret"] = [
        fwd.at[d, t] if (d in fwd.index and t in fwd.columns) else np.nan
        for d, t in zip(ns["date"], ns["ticker"])
    ]
    ns = ns.dropna(subset=["fwd_ret"])
    if ns.empty:
        return zero

    # Contribución diaria de cada casa a su edge (num) y su exposición (den)
    ns["num"] = ns["netshare"] * ns["fwd_ret"]
    ns["den"] = ns["netshare"].abs()
    daily = ns.groupby(["casa", "date"])[["num", "den"]].sum().reset_index()

    all_dates = sorted(netshare["date"].unique())
    date_index = pd.DatetimeIndex(all_dates)

    edges: dict[str, pd.Series] = {}
    for casa, grp in daily.groupby("casa"):
        s = grp.set_index("date")[["num", "den"]].reindex(date_index).fillna(0.0)
        num_roll = s["num"].rolling(SMART_WINDOW_DAYS, min_periods=20).sum()
        den_roll = s["den"].rolling(SMART_WINDOW_DAYS, min_periods=20).sum()
        edge = (num_roll / den_roll.replace(0, np.nan)).shift(SMART_GAP_DAYS)
        edges[casa] = edge.clip(-0.10, 0.10)

    edge_df = pd.DataFrame(edges)

    full = netshare.copy()
    full["edge"] = [
        edge_df.at[d, c] if (d in edge_df.index and c in edge_df.columns) else np.nan
        for d, c in zip(full["date"], full["casa"])
    ]
    full["contrib"] = full["edge"].fillna(0.0) * full["netshare"]
    smart = full.groupby(["date", "ticker"])["contrib"].sum()
    return smart.reindex(idx).fillna(0.0)


def compute_flow_panel(
    tickers: list[str] | None = None,
    close_panel: pd.DataFrame | None = None,
    hechos_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Panel diario point-in-time: date, ticker, FLOW_FEATURE_COLS.

    close_panel : DataFrame date × ticker de precios Close (para flow_smart).
                  Sin él, flow_smart = 0 (el resto de features no lo necesitan).
    """
    raw = load_flow_raw(tickers, hechos_dir)
    if raw.empty:
        return pd.DataFrame(columns=["date", "ticker", *FLOW_FEATURE_COLS])

    stats = _daily_flow_stats(raw)
    netshare = _casa_netshare(raw)
    smart = _smart_flow(netshare, close_panel)

    panel = stats.set_index(["date", "ticker"])
    panel["flow_smart"] = smart.reindex(panel.index).fillna(0.0)
    panel = panel.sort_index()

    panel["flow_persistence"] = (
        panel.groupby(level="ticker")["flow_imbalance"]
        .transform(lambda s: s.rolling(5, min_periods=2).mean())
    )

    out = panel.reset_index()[["date", "ticker", *FLOW_FEATURE_COLS]]
    for c in FLOW_FEATURE_COLS:
        out[c] = out[c].astype(float).fillna(0.0)
    return out


def add_flow_features(
    feat_df: pd.DataFrame,
    ticker: str,
    flow_panel: pd.DataFrame | None,
) -> pd.DataFrame:
    """Merge del panel de flujo a un frame diario de un ticker (index fecha)."""
    df = feat_df.copy()
    if flow_panel is None or flow_panel.empty:
        for c in FLOW_FEATURE_COLS:
            df[c] = 0.0
        return df

    sub = flow_panel[flow_panel["ticker"] == ticker]
    if sub.empty:
        for c in FLOW_FEATURE_COLS:
            df[c] = 0.0
        return df

    sub = sub.set_index(pd.to_datetime(sub["date"]))[FLOW_FEATURE_COLS]
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    for c in FLOW_FEATURE_COLS:
        df[c] = sub[c].reindex(idx).to_numpy()
        df[c] = df[c].astype(float).fillna(0.0)
    return df
