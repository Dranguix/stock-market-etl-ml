"""
Feature engineering para señales de mercado BMV.
Todos los indicadores son causal (sin look-ahead bias).

Uso:
    from src.ml.features import compute_features, FEATURE_COLS

    df = compute_features(ohlcv_df)   # input: Open, High, Low, Close, Volume
    X  = df[FEATURE_COLS].dropna()
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml.intraday_features import (
    INTRADAY_FEATURE_COLS,
    add_intraday_features,
)
from src.ml.news import NEWS_FEATURE_COLS
# Organizados en 5 grupos funcionales.  Todos son causales (sin look-ahead).
FEATURE_COLS = [
    # ── A: Osciladores mean-reversion ────────────────────────────────────────
    "rsi_14",       # RSI estándar 14 períodos
    "rsi_7",        # RSI rápido 7 períodos (más sensible)
    "stoch_k",      # Stocástico %K(14,3)  — posición en rango reciente
    "williams_r",   # Williams %R(14)      — oscilador invertido, -100=oversold
    "cci_20",       # Commodity Channel Index 20p
    "mfi_14",       # Money Flow Index  — RSI ponderado por volumen
    "bb_pct",       # Bollinger Band %  — 0=banda inferior, 100=banda superior

    # ── B: Estructura de precio ───────────────────────────────────────────────
    "price_vs_sma20",   # distancia % al SMA20 (clave: negativa = bajo MA)
    "price_vs_sma50",   # distancia % al SMA50
    "sma20_vs_sma50",   # golden/death cross distancia
    "pos_52w",          # posición en rango máx-mín 52 semanas (0-100%)
    "dist_20d_low",     # % sobre mínimo de los últimos 20 días
    "consec_down",      # racha de cierres consecutivos bajistas

    # ── C: Momentum / tendencia ───────────────────────────────────────────────
    "ret_5d",       # retorno 5 días (lookback)
    "ret_10d",      # retorno 10 días
    "ret_20d",      # retorno 20 días (momentum mensual)
    "sma20_slope",  # pendiente normalizada del SMA20 en 5 días
    "macd_hist",    # histograma MACD(12,26,9) normalizado por precio (Kumbure 2022)
    "macd_signal",  # línea de señal MACD normalizada por precio

    # ── D: Volumen ────────────────────────────────────────────────────────────
    "vol_ratio_20",     # volumen hoy vs media 20d
    "vol_down_up",      # ratio vol en días bajistas vs alcistas (últimos 5d)
    "obv_slope",        # pendiente del OBV 10d (acumulación/distribución)

    # ── E: Volatilidad / régimen ──────────────────────────────────────────────
    "atr_pct",      # ATR(14) / precio  — volatilidad relativa
    "bb_width",     # ancho bandas Bollinger / precio  — compresión/expansión
    "hv_ratio",     # vol histórica 5d / 20d  — expansión de volatilidad
    "hl_range_pct", # (High-Low)/Close del día
    "trend_5d",     # pendiente lineal de precio en 5 días (normalizada)

    # ── F: Macro / contexto de mercado ────────────────────────────────────────
    "usdmxn_ret5",    # retorno USD/MXN en 5d  — positivo = peso debilitado
    "usdmxn_vs_ma20", # USD/MXN vs su SMA20  — peso fuerte o débil
    "mkt_ret10",      # retorno NAFTRAC (IPC) en 10d  — ¿mercado sube o baja?
    "mkt_trend",      # NAFTRAC vs SMA50  — bull o bear market
    "rel_str5",       # retorno ticker 5d menos NAFTRAC 5d  — señal idiosincrática
    "wti_ret5",       # retorno WTI petróleo en 5d
    "sp500_ret5",     # retorno S&P 500 en 5d  — contagio global
    # ── F2: Tasas de mercado de dinero mexicanas ──────────────────────────────
    "tiie_spread",    # TIIE91 - TIIE28 (bp) — pendiente corta interbank
    "cete_slope",     # CETE364 - CETE28 (bp) — pendiente curva CETES
    "tiie28_level",   # nivel TIIE28 (% anual) — costo de capital en MX
    "tiie28_ret5",    # cambio TIIE28 en 5d (bp) — ¿sube o baja la tasa?
    "tasa_obj_delta", # cambio Tasa_Objetivo 63d (bp) — ciclo de política monetaria
    # ── F3: Dividendo ex-date ─────────────────────────────────────────────────
    "dias_a_ex_date", # días al ex-date más próximo (negativo = pasó)
    "ex_date_window", # 1 si en ventana [-3, +10] del ex-date
    "div_yield_fwd",  # dividendo anualizado / precio (%)
    # ── G: Cross-seccionales (ranking vs universo) ───────────────────────────
    "cs_ret5_rank",   # percentil retorno 5d vs universo  (momentum relativo)
    "cs_ret20_rank",  # percentil retorno 20d vs universo
    "cs_rsi_rank",    # percentil RSI vs universo  (¿más sobrevendido que pares?)
    "cs_vol_rank",    # percentil vol relativo vs universo
    "cs_mom_rank",    # momentum compuesto cross-seccional
    "cs_mean_rev",    # z-score retorno 5d  (mean-reversion cross-seccional)
]

# Features de momentum largo (>20d) — calculadas en compute_features, úsalas opcionalmente
LONG_MOMENTUM_COLS = ["ret_60d", "ret_90d"]

# Features que requieren datos macro externos (se añaden con add_macro_features)
MACRO_FEATURE_COLS = [
    "usdmxn_ret5", "usdmxn_vs_ma20",
    "mkt_ret10", "mkt_trend", "rel_str5",
    "wti_ret5", "sp500_ret5",
    "tiie_spread", "cete_slope", "tiie28_level", "tiie28_ret5", "tasa_obj_delta",
    "dias_a_ex_date", "ex_date_window", "div_yield_fwd",
]

# Features puramente técnicas (calculables con solo OHLCV del ticker)
TECH_FEATURE_COLS = [f for f in FEATURE_COLS if f not in MACRO_FEATURE_COLS]


def add_news_features(
    df: pd.DataFrame,
    news_df: pd.DataFrame,
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    """
    Merge de features de noticias por ticker.

    Parameters
    ----------
    df       : DataFrame con columna `ticker_col` (OHLCV + features técnicos)
    news_df  : DataFrame con ticker index o columna ticker y las 4 columnas
               NEWS_FEATURE_COLS (salida de batch_news_features / cache)
    ticker_col : nombre de la columna ticker en df

    Returns
    -------
    DataFrame con NEWS_FEATURE_COLS añadidas. Tickers sin noticias → 0.0.

    Uso en ensemble
    ---------------
    El EnsembleModel (WoE + LR + LightGBM) puede incluir news features
    pasando get_all_feature_cols(include_news=True) como feature_cols.
    news_sent_score captura tono agregado; news_count_7d volumen informativo;
    news_sent_delta cambio vs ventana anterior; news_sent_vol incertidumbre.
    """
    out = df.copy()
    news = news_df.copy()

    if news is None or news.empty:
        for col in NEWS_FEATURE_COLS:
            out[col] = 0.0
        return out

    if "ticker" not in news.columns:
        news = news.reset_index()
        if news.columns[0] != "ticker":
            news = news.rename(columns={news.columns[0]: "ticker"})

    available = [c for c in NEWS_FEATURE_COLS if c in news.columns]
    news_sub = news.drop_duplicates(subset=["ticker"]).set_index("ticker")

    for col in NEWS_FEATURE_COLS:
        if col in available:
            out[col] = out[ticker_col].map(news_sub[col]).fillna(0.0)
        else:
            out[col] = 0.0

    return out


def add_intraday_features_to_panel(
    ticker: str,
    feat_df: pd.DataFrame,
    cs_medians: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Añade INTRADAY_FEATURE_COLS a un panel diario ya con features técnicos."""
    return add_intraday_features(feat_df, ticker, cs_medians=cs_medians)


def add_all_optional_features(
    ticker_df: pd.DataFrame,
    ticker: str,
    *,
    include_intraday: bool = False,
    intraday_cs_medians: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Punto único para features opcionales (intradía, etc.) sobre OHLCV + técnicos.

    Parameters
    ----------
    include_intraday       : activa microestructura 5m agregada a diario
    intraday_cs_medians    : medianas cross-seccional para imputar NaN restantes
    """
    df = ticker_df
    if include_intraday:
        df = add_intraday_features_to_panel(
            ticker, df, cs_medians=intraday_cs_medians,
        )
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:  # noqa: C901
    """
    Calcula todos los features técnicos sobre un DataFrame OHLCV.

    Parameters
    ----------
    df : pd.DataFrame con columnas Open, High, Low, Close, Volume
         Index debe ser DatetimeIndex ordenado ascendente.

    Returns
    -------
    pd.DataFrame con las columnas originales más todos los features.
    Sin look-ahead: cada row usa solo datos hasta esa fecha inclusive.
    """
    df = df.copy()
    c = df["Close"]
    h = df["High"]
    l = df["Low"]
    v = df["Volume"]

    # ── Helper: pendiente lineal normalizada ─────────────────────────────────
    def linear_slope(series: pd.Series, window: int) -> pd.Series:
        slopes = np.full(len(series), np.nan)
        x = np.arange(window, dtype=float)
        xc = x - x.mean()
        denom = (xc ** 2).sum()
        arr = series.to_numpy(dtype=float)
        for i in range(window - 1, len(arr)):
            y = arr[i - window + 1 : i + 1]
            if np.any(np.isnan(y)):
                continue
            slope = (xc * (y - y.mean())).sum() / denom
            slopes[i] = slope / (abs(y.mean()) + 1e-8) * 100
        return pd.Series(slopes, index=series.index)

    # ── Retornos ─────────────────────────────────────────────────────────────
    df["ret_5d"]  = c.pct_change(5)  * 100
    df["ret_10d"] = c.pct_change(10) * 100
    df["ret_20d"] = c.pct_change(20) * 100
    df["ret_60d"] = c.pct_change(60) * 100   # momentum trimestral
    df["ret_90d"] = c.pct_change(90) * 100   # momentum cuatrimestral

    # ── Medias móviles ────────────────────────────────────────────────────────
    sma20 = c.rolling(20).mean()
    sma50 = c.rolling(50).mean()
    df["price_vs_sma20"] = (c - sma20) / (sma20 + 1e-8) * 100
    df["price_vs_sma50"] = (c - sma50) / (sma50 + 1e-8) * 100
    df["sma20_vs_sma50"] = (sma20 - sma50) / (sma50 + 1e-8) * 100
    df["sma20_slope"]    = linear_slope(sma20, 5)

    # ── Grupo A: Osciladores ──────────────────────────────────────────────────
    # RSI genérico
    def _rsi(series: pd.Series, period: int) -> pd.Series:
        d = series.diff()
        g = d.clip(lower=0).rolling(period).mean()
        ls = (-d.clip(upper=0)).rolling(period).mean()
        rs = g / ls.replace(0, np.nan)
        return 100 - 100 / (1 + rs)

    df["rsi_14"] = _rsi(c, 14)
    df["rsi_7"]  = _rsi(c, 7)

    # Stochastic %K(14,3)
    low14  = l.rolling(14).min()
    high14 = h.rolling(14).max()
    raw_k  = (c - low14) / (high14 - low14 + 1e-8) * 100
    df["stoch_k"] = raw_k.rolling(3).mean()          # %K suavizado

    # Williams %R(14)  — igual que Stoch pero invertido y en [-100, 0]
    df["williams_r"] = (high14 - c) / (high14 - low14 + 1e-8) * -100

    # CCI(20)
    tp        = (h + l + c) / 3
    tp_ma     = tp.rolling(20).mean()
    tp_mad    = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df["cci_20"] = (tp - tp_ma) / (0.015 * tp_mad + 1e-8)

    # Money Flow Index (14)  — RSI ponderado por volumen
    raw_mf  = tp * v
    delta_tp = tp.diff()
    pos_mf  = raw_mf.where(delta_tp > 0, 0).rolling(14).sum()
    neg_mf  = raw_mf.where(delta_tp < 0, 0).rolling(14).sum()
    mfr     = pos_mf / neg_mf.replace(0, np.nan)
    df["mfi_14"] = 100 - 100 / (1 + mfr)

    # Bollinger Band %  (0 = banda inferior, 100 = banda superior)
    bb_std  = c.rolling(20).std()
    bb_low  = sma20 - 2 * bb_std
    bb_high = sma20 + 2 * bb_std
    df["bb_pct"]   = (c - bb_low)  / (bb_high - bb_low + 1e-8) * 100
    df["bb_width"] = (bb_high - bb_low) / (sma20 + 1e-8) * 100

    # ── MACD(12,26,9) — normalizado por precio para comparabilidad cross-ticker
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_sig = macd_line.ewm(span=9, adjust=False).mean()
    df["macd_signal"] = macd_sig / (c + 1e-8) * 100
    df["macd_hist"]   = (macd_line - macd_sig) / (c + 1e-8) * 100

    # ── Grupo B: Estructura de precio ─────────────────────────────────────────
    high_52 = h.rolling(252, min_periods=50).max()
    low_52  = l.rolling(252, min_periods=50).min()
    df["pos_52w"] = (c - low_52) / (high_52 - low_52 + 1e-8) * 100

    # Distancia al mínimo de 20 días
    low20 = l.rolling(20).min()
    df["dist_20d_low"] = (c - low20) / (low20 + 1e-8) * 100

    # Racha de cierres consecutivos bajistas
    down     = (c.diff() < 0).astype(int)
    streaks  = np.zeros(len(c))
    for i in range(1, len(c)):
        streaks[i] = (streaks[i-1] + 1) * down.iloc[i]
    df["consec_down"] = streaks

    # ── Grupo C: Tendencia ────────────────────────────────────────────────────
    df["trend_5d"]  = linear_slope(c, 5)

    # ── Grupo D: Volumen ──────────────────────────────────────────────────────
    vol_ma20 = v.rolling(20).mean()
    df["vol_ratio_20"] = v / vol_ma20.replace(0, np.nan)

    # Ratio vol en días bajistas vs alcistas (últimos 5 días)
    is_down  = (c.diff() < 0).astype(float)
    vol_down = (v * is_down).rolling(5).sum()
    vol_up   = (v * (1 - is_down)).rolling(5).sum()
    df["vol_down_up"] = vol_down / vol_up.replace(0, np.nan)

    # OBV slope
    obv = (np.sign(c.diff()) * v).cumsum()
    df["obv_slope"] = linear_slope(obv, 10) / vol_ma20.replace(0, np.nan)

    # ── Grupo E: Volatilidad ──────────────────────────────────────────────────
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    df["atr_pct"]    = atr14 / (c + 1e-8) * 100
    df["_atr14"]     = atr14                    # auxiliar para el backtester
    df["hl_range_pct"] = (h - l) / (c + 1e-8) * 100

    hv5  = c.pct_change().rolling(5).std()
    hv20 = c.pct_change().rolling(20).std()
    df["hv_ratio"] = hv5 / hv20.replace(0, np.nan)

    return df


def compute_target(df: pd.DataFrame, horizon: int = 5, use_open_entry: bool = True) -> pd.Series:
    """
    Calcula el retorno futuro para usar como target de entrenamiento.

    Parameters
    ----------
    df      : DataFrame con columnas Close y Open
    horizon : días de holding
    use_open_entry : True = entrada al Open del día siguiente (realista)
                     False = entrada al Close del día de señal (optimista)

    Returns
    -------
    pd.Series con retorno % del horizonte.
    Alineada con el día de la SEÑAL (no el de entrada).
    """
    c = df["Close"]
    o = df["Open"]

    if use_open_entry:
        # Entry = Open[t+1], Exit = Close[t+1+horizon]
        entry  = o.shift(-1)
        exit_  = c.shift(-(horizon + 1))
    else:
        entry  = c
        exit_  = c.shift(-horizon)

    return (exit_ - entry) / (entry + 1e-8) * 100


def compute_signal_score(row: pd.Series) -> tuple[float, list[str]]:
    """
    Score heurístico del screener (idéntico al screener_lunes.py).
    Para backtest como baseline antes de introducir ML.

    Returns
    -------
    (score, señales_activas)
    """
    score = 0.0
    signals: list[str] = []

    rsi = row.get("rsi_14", np.nan)
    if pd.notna(rsi):
        if rsi < 35:
            score += 25; signals.append("RSI_oversold")
        elif rsi < 45:
            score += 15; signals.append("RSI_low")
        elif rsi > 65:
            score -= 10; signals.append("RSI_high")

    pvs20 = row.get("price_vs_sma20", np.nan)
    if pd.notna(pvs20):
        if pvs20 > 0:
            score += 20; signals.append("above_sma20")
        else:
            score -= 10; signals.append("below_sma20")

    cross = row.get("sma20_vs_sma50", np.nan)
    if pd.notna(cross):
        if cross > 0:
            score += 15; signals.append("golden_cross")
        else:
            score -= 5

    mom5 = row.get("ret_5d", np.nan)
    if pd.notna(mom5):
        if mom5 > 2:
            score += 15; signals.append("mom5_pos")
        elif mom5 < -3:
            score -= 10; signals.append("mom5_neg")

    vol = row.get("vol_ratio_20", np.nan)
    if pd.notna(vol) and vol > 1.5:
        score += 10; signals.append("high_volume")

    pos52 = row.get("pos_52w", np.nan)
    # Alias backward compat: bb_pos → bb_pct
    if pd.notna(pos52):
        if pos52 > 90:
            score -= 5
        elif pos52 < 20:
            score += 10; signals.append("near_52w_low")

    return score, signals


def score_to_category(score: float) -> str:
    if score >= 40:
        return "COMPRA"
    elif score >= 20:
        return "VIGILAR"
    elif score < 0:
        return "EVITAR"
    return "NEUTRO"


def add_macro_features(
    ticker_df: pd.DataFrame,
    macro: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Añade los 7 features macro al DataFrame de un ticker.

    Parameters
    ----------
    ticker_df : DataFrame con features técnicos ya calculados (compute_features)
    macro     : dict con DataFrames de datos macro. Claves esperadas:
                  "USDMXN"  — USD/MXN (MXN=X)
                  "NAFTRAC" — IPC ETF (NAFTRAC.MX)
                  "WTI"     — WTI petróleo (CL=F)
                  "SP500"   — S&P 500 (^GSPC)

    Returns
    -------
    DataFrame con las columnas macro añadidas.
    """
    df = ticker_df.copy()
    df.index = pd.to_datetime(df.index)

    ticker_close = df["Close"] if "Close" in df.columns else None

    # ── USD/MXN ───────────────────────────────────────────────────────────────
    if "USDMXN" in macro and macro["USDMXN"] is not None:
        usd = macro["USDMXN"]["Close"].reindex(df.index, method="ffill")
        usd_sma20 = usd.rolling(20).mean()
        df["usdmxn_ret5"]    = usd.pct_change(5) * 100
        df["usdmxn_vs_ma20"] = (usd - usd_sma20) / (usd_sma20 + 1e-8) * 100
    else:
        df["usdmxn_ret5"]    = np.nan
        df["usdmxn_vs_ma20"] = np.nan

    # ── NAFTRAC / IPC ─────────────────────────────────────────────────────────
    if "NAFTRAC" in macro and macro["NAFTRAC"] is not None:
        mkt = macro["NAFTRAC"]["Close"].reindex(df.index, method="ffill")
        mkt_sma50 = mkt.rolling(50).mean()
        df["mkt_ret10"] = mkt.pct_change(10) * 100
        df["mkt_trend"] = (mkt - mkt_sma50) / (mkt_sma50 + 1e-8) * 100
        if ticker_close is not None:
            df["rel_str5"] = (
                ticker_close.pct_change(5) * 100
                - mkt.pct_change(5) * 100
            )
        else:
            df["rel_str5"] = np.nan
    else:
        df["mkt_ret10"] = np.nan
        df["mkt_trend"] = np.nan
        df["rel_str5"]  = np.nan

    # ── WTI ───────────────────────────────────────────────────────────────────
    if "WTI" in macro and macro["WTI"] is not None:
        wti = macro["WTI"]["Close"].reindex(df.index, method="ffill")
        df["wti_ret5"] = wti.pct_change(5) * 100
    else:
        df["wti_ret5"] = np.nan

    # ── S&P 500 ───────────────────────────────────────────────────────────────
    if "SP500" in macro and macro["SP500"] is not None:
        sp = macro["SP500"]["Close"].reindex(df.index, method="ffill")
        df["sp500_ret5"] = sp.pct_change(5) * 100
    else:
        df["sp500_ret5"] = np.nan

    return df


# ─── Features cross-seccionales ──────────────────────────────────────────────

# Nombres de las nuevas columnas cross-seccionales
CROSS_SECTION_COLS = [
    "cs_ret5_rank",    # percentil de retorno 5d vs universo  (0-1)
    "cs_ret20_rank",   # percentil de retorno 20d vs universo
    "cs_rsi_rank",     # percentil de RSI 14 vs universo
    "cs_vol_rank",     # percentil de volumen relativo vs universo
    "cs_mom_rank",     # ranking compuesto de momentum
    "cs_mean_rev",     # zscore de retorno 5d dentro del universo (mean reversion)
]


# ─── Features fundamentales (DataBursátil + yfinance) ────────────────────────

_FUND_QUARTERLY_COLS = [
    "fund_gross_margin",
    "fund_oper_margin",
    "fund_net_margin",
    "fund_revenue_yoy",
    "fund_earnings_yoy",
]

_YFINANCE_FUND_COLS = [
    "pe_ratio",
    "market_cap_log",
    "dividend_yield",
    "profit_margin",
]

FUNDAMENTAL_FEATURE_COLS = _FUND_QUARTERLY_COLS + _YFINANCE_FUND_COLS


def _fetch_yfinance_fundamentals(symbol: str) -> dict[str, float]:
    """Ratios fundamentales estáticos vía yfinance.info."""
    out = {c: np.nan for c in _YFINANCE_FUND_COLS}
    try:
        import yfinance as yf
        info = yf.Ticker(symbol).info or {}
        pe = info.get("trailingPE") or info.get("forwardPE")
        mcap = info.get("marketCap")
        div_y = info.get("dividendYield") or info.get("yield")
        margin = info.get("profitMargins")
        if pe is not None and np.isfinite(pe) and pe > 0:
            out["pe_ratio"] = float(pe)
        if mcap is not None and mcap > 0:
            out["market_cap_log"] = float(np.log(mcap))
        if div_y is not None and np.isfinite(div_y):
            out["dividend_yield"] = float(div_y * 100 if div_y < 1 else div_y)
        if margin is not None and np.isfinite(margin):
            out["profit_margin"] = float(margin * 100 if abs(margin) < 1 else margin)
    except Exception:
        pass
    return out


def get_all_feature_cols(
    include_news: bool = False,
    include_fundamentals: bool = False,
    include_intraday: bool = False,
) -> list[str]:
    """
    Lista canónica de columnas de features para entrenamiento/inferencia.

    Parameters
    ----------
    include_news         : añade NEWS_FEATURE_COLS (sentiment NLP)
    include_fundamentals : añade FUNDAMENTAL_FEATURE_COLS (DataBursátil)
    include_intraday     : añade INTRADAY_FEATURE_COLS (microestructura 5m)

    Returns
    -------
    Lista ordenada sin duplicados. FEATURE_COLS base no se modifica;
    SuperModel agent puede activar news con include_news=True.
    """
    cols = list(FEATURE_COLS)
    if include_news:
        cols.extend(c for c in NEWS_FEATURE_COLS if c not in cols)
    if include_fundamentals:
        cols.extend(c for c in FUNDAMENTAL_FEATURE_COLS if c not in cols)
    if include_intraday:
        cols.extend(c for c in INTRADAY_FEATURE_COLS if c not in cols)
    return cols


def add_fundamental_features(
    ticker_df: pd.DataFrame,
    fund_df: pd.DataFrame | None = None,
    *,
    ticker: str | None = None,
    symbol: str | None = None,
) -> pd.DataFrame:
    """
    Añade features fundamentales a un DataFrame diario.

    Fuentes:
      - ``fund_df``: trimestrales DataBursátil vía merge_asof (sin look-ahead).
      - ``symbol`` / ``ticker``: P/E, market cap log, dividend yield, profit margin
        vía yfinance (snapshot constante en el panel).

    NaN handling: trimestrales sin publicación → NaN; yfinance sin dato → NaN
    por columna. El caller (SuperModel) puede imputar mediana antes de entrenar.

    Parameters
    ----------
    ticker_df : DataFrame diario con DatetimeIndex.
    fund_df   : DataFrame trimestral opcional (legacy scripts/fundamentals_backtest).
    ticker    : nombre corto (informativo).
    symbol    : símbolo yfinance (ej. ``WALMEX.MX``, ``AAPL``).

    Returns
    -------
    DataFrame con FUNDAMENTAL_FEATURE_COLS añadidas.
    """
    df = ticker_df.copy()
    df.index = pd.to_datetime(df.index)

    quarterly_cols = [
        c for c in _FUND_QUARTERLY_COLS
        if fund_df is not None and len(fund_df) > 0 and c in fund_df.columns
    ]
    if fund_df is not None and len(fund_df) > 0 and quarterly_cols:
        fund = fund_df.copy()
        if "available_date" in fund.columns:
            fund["available_date"] = pd.to_datetime(fund["available_date"])
            fund = fund.sort_values("available_date")
            left = pd.DataFrame({"_date": df.index}).sort_values("_date")
            merged = pd.merge_asof(
                left, fund[["available_date"] + quarterly_cols],
                left_on="_date", right_on="available_date",
                direction="backward",
            )
            merged.index = merged["_date"].values
            for col in quarterly_cols:
                df[col] = merged[col].reindex(df.index).values
        else:
            for col in quarterly_cols:
                df[col] = fund[col].iloc[-1]
    for col in _FUND_QUARTERLY_COLS:
        if col not in df.columns:
            df[col] = np.nan

    yf_sym = symbol or ticker
    yf_vals = _fetch_yfinance_fundamentals(yf_sym) if yf_sym else {}
    for col in _YFINANCE_FUND_COLS:
        df[col] = yf_vals.get(col, np.nan)

    return df


def add_cross_section_features(
    frames: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """
    Agrega features cross-seccionales a cada DataFrame en `frames`.

    Compara cada ticker con el resto del universo en la misma fecha:
    - Percentil de retorno 5d y 20d (momentum cross-seccional)
    - Percentil de RSI (¿más sobrecomprado/sobrevendido que sus pares?)
    - Percentil de volumen relativo (¿actividad inusual?)
    - Z-score de retorno 5d (mean-reversion cross-seccional)

    Sin look-ahead: los rankings se calculan en t usando datos hasta t.

    Args:
        frames: dict ticker → DataFrame con features ya calculadas (compute_features).
                Cada DataFrame debe tener DatetimeIndex y las columnas
                ret_5d, ret_20d, rsi_14, vol_ratio_20.

    Returns:
        Mismo dict con columnas CROSS_SECTION_COLS añadidas a cada DataFrame.
    """
    # Construir panel: fecha × ticker para cada feature base
    needed = ["ret_5d", "ret_20d", "rsi_14", "vol_ratio_20"]

    panels: dict[str, pd.DataFrame] = {}
    for feat in needed:
        cols = {}
        for ticker, df in frames.items():
            if feat in df.columns:
                cols[ticker] = df[feat]
        if cols:
            panels[feat] = pd.DataFrame(cols)

    if not panels:
        return frames

    # Calcular rankings percentil por fila (cross-seccional) ─────────────────
    def pct_rank(panel: pd.DataFrame) -> pd.DataFrame:
        """Rango percentil [0,1] dentro del universo por fecha."""
        return panel.rank(axis=1, pct=True, na_option="keep")

    def zscore_cs(panel: pd.DataFrame) -> pd.DataFrame:
        """Z-score dentro del universo por fecha."""
        mu  = panel.mean(axis=1)
        std = panel.std(axis=1).replace(0, np.nan)
        return panel.sub(mu, axis=0).div(std, axis=0)

    rank_ret5  = pct_rank(panels["ret_5d"])  if "ret_5d"      in panels else None
    rank_ret20 = pct_rank(panels["ret_20d"]) if "ret_20d"     in panels else None
    rank_rsi   = pct_rank(panels["rsi_14"])  if "rsi_14"      in panels else None
    rank_vol   = pct_rank(panels["vol_ratio_20"]) if "vol_ratio_20" in panels else None
    zscore_r5  = zscore_cs(panels["ret_5d"]) if "ret_5d"      in panels else None

    # Momentum compuesto: promedio de rank_ret5 y rank_ret20
    if rank_ret5 is not None and rank_ret20 is not None:
        rank_mom = (rank_ret5 + rank_ret20) / 2
    elif rank_ret5 is not None:
        rank_mom = rank_ret5
    else:
        rank_mom = None

    # Asignar de vuelta a cada ticker ─────────────────────────────────────────
    result = {}
    for ticker, df in frames.items():
        df = df.copy()

        def _get(panel, col):
            if panel is not None and ticker in panel.columns:
                return panel[ticker].reindex(df.index)
            return pd.Series(np.nan, index=df.index)

        df["cs_ret5_rank"]  = _get(rank_ret5,  ticker)
        df["cs_ret20_rank"] = _get(rank_ret20, ticker)
        df["cs_rsi_rank"]   = _get(rank_rsi,   ticker)
        df["cs_vol_rank"]   = _get(rank_vol,   ticker)
        df["cs_mom_rank"]   = _get(rank_mom,   ticker)
        df["cs_mean_rev"]   = _get(zscore_r5,  ticker)

        result[ticker] = df

    return result
