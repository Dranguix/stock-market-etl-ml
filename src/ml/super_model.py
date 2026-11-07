"""
SuperModel — Meta-orquestador que fusiona todos los modelos del sistema.

Combina las señales de varios modelos base en UNA decisión de convicción por
ticker, modulada por régimen de mercado y sentimiento de noticias, y produce un
ranking maestro cross-universo (los 87 tickers de los 5 universos en una sola
lista).

Capas:
  L1 — Modelos base (predictores direccionales):
        · xs_lgbm   : LGBMRegressor sobre features rank-normalizadas (cross-seccional)
        · ensemble  : EnsembleModel (WoE + LR + LGBM) → P(le gana a la mediana)
        · heuristic : compute_signal_score (screener técnico) normalizado
        · lstm_seq  : LSTMSeqPredictor → P(outperform) sobre ventanas secuenciales
  L2 — Fusión: peso de cada modelo ∝ su edge histórico (AUC − 0.5) por universo,
        O stacking LR meta-learner si ``use_stacking=True`` y supera AUC-weight.
  L3 — Moduladores: régimen (RegimeDetector HMM) × sentimiento de noticias.
  L4 — Ranking masivo cross-universo + insights (lo consume gemini_validator).

Fusión L2 — cuándo usar cada modo:
  - **Stacking LR (``use_stacking=True``, default)**: meta-LogisticRegression sobre preds
    [xs_lgbm, ensemble, heuristic, lstm_seq]; preferido post-experimentos P0 (KB-060).
  - **AUC-weight (``use_stacking=False``)**: fallback interpretable; cada modelo aporta
    según su edge medido en holdout temporal.

Diseño: cuantitativo primero. Gemini (capa 4, en gemini_validator.py) solo
valida y narra lo que este modelo decide; nunca genera la recomendación.

Uso:
    from src.ml.super_model import SuperModel
    sm = SuperModel()
    result = sm.run()          # entrena, predice, fusiona, rankea
    # result["master_ranking"] → lista de dicts ordenada por convicción

    # Con stacking meta-learner (experimental):
    sm = SuperModel(use_stacking=True)
"""
from __future__ import annotations

import logging
import warnings
from datetime import date

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

from src.data import store
from src.data.universes import UNIVERSES
from src.ml.features import (
    compute_features, add_macro_features, add_cross_section_features,
    add_fundamental_features, add_news_features,
    compute_target, compute_signal_score,
    FEATURE_COLS, LONG_MOMENTUM_COLS, FUNDAMENTAL_FEATURE_COLS, NEWS_FEATURE_COLS,
)

try:
    from src.ml.intraday_features import INTRADAY_FEATURE_COLS, add_intraday_features
    _INTRADAY_AVAILABLE = True
except ImportError:
    try:
        from src.ml.intraday_features import INTRADAY_COLS as INTRADAY_FEATURE_COLS, add_intraday_features
        _INTRADAY_AVAILABLE = True
    except ImportError:
        INTRADAY_FEATURE_COLS = []  # type: ignore[misc, assignment]
        add_intraday_features = None  # type: ignore[misc, assignment]
        _INTRADAY_AVAILABLE = False

try:
    from src.ml.lstm_seq import LSTMSeqPredictor, AUC_THRESHOLD as LSTM_AUC_THRESHOLD
    _LSTM_AVAILABLE = True
except ImportError:
    LSTMSeqPredictor = None  # type: ignore[misc, assignment]
    LSTM_AUC_THRESHOLD = 0.50
    _LSTM_AVAILABLE = False

from src.ml.super_metrics import (
    CORE_MODELS,
    compute_fusion_metrics,
    compute_universe_model_metrics,
    enrich_ranking_item,
    build_model_registry,
)
from src.ml.kb_enhance import (
    apply_kb_modulators,
    get_wti_ret5_pct,
    load_horizon_consensus,
    select_priority_features,
    select_priority_rn_cols,
)

try:
    from src.ml.flow_features import FLOW_FEATURE_COLS, add_flow_features, compute_flow_panel
    _FLOW_AVAILABLE = True
except ImportError:
    FLOW_FEATURE_COLS = []  # type: ignore[misc, assignment]
    _FLOW_AVAILABLE = False

try:
    from src.ml.graph_ising import GIM_FEATURE_COLS, add_gim_features, compute_gim_panel
    _GIM_AVAILABLE = True
except ImportError:
    GIM_FEATURE_COLS = []  # type: ignore[misc, assignment]
    _GIM_AVAILABLE = False

try:
    from src.ml.rates_features import (
        RATES_FEATURE_COLS, add_rates_features,
        build_rates_panel, compute_rates_features, fetch_tasas_snapshot,
    )
    _RATES_AVAILABLE = True
except ImportError:
    RATES_FEATURE_COLS = []  # type: ignore[misc, assignment]
    _RATES_AVAILABLE = False

try:
    from src.ml.dividend_features import (
        DIVIDEND_FEATURE_COLS, add_dividend_features,
        build_dividend_panel, fetch_all_dividends,
    )
    _DIVIDENDS_AVAILABLE = True
except ImportError:
    DIVIDEND_FEATURE_COLS = []  # type: ignore[misc, assignment]
    _DIVIDENDS_AVAILABLE = False

# Modelos base para fusión L2 / stacking (KB-012, KB-060)
STACK_BASE_MODELS = CORE_MODELS
EXTENDED_FEATURE_COLS = (
    FEATURE_COLS + LONG_MOMENTUM_COLS + FUNDAMENTAL_FEATURE_COLS + list(INTRADAY_FEATURE_COLS)
    + list(FLOW_FEATURE_COLS) + list(GIM_FEATURE_COLS)
    + list(RATES_FEATURE_COLS) + list(DIVIDEND_FEATURE_COLS)
)

# Sufijo rank-normalizado (percentil cross-seccional) — sobre features extendidas
RN_COLS = [f"rn_{c}" for c in EXTENDED_FEATURE_COLS]

# Parámetros LGBM mejorados (menos regularización → más capacidad con pocos tickers)
_LGBM_BASE = dict(
    n_estimators=500, num_leaves=31, learning_rate=0.02, min_child_samples=20,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, reg_lambda=3.0,
    n_jobs=2, random_state=42, verbosity=-1,
)

# Símbolos macro para add_macro_features
_MACRO_SYMS = {"USDMXN": "MXN=X", "NAFTRAC": "NAFTRAC.MX", "WTI": "CL=F", "SP500": "^GSPC"}


def _minmax(x: np.ndarray) -> np.ndarray:
    """Normaliza a [0,1]. Si todo es igual, devuelve 0.5."""
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.full_like(x, 0.5)
    return (x - lo) / (hi - lo)


class SuperModel:
    """Meta-orquestador de los modelos del sistema."""

    def __init__(
        self,
        top_gemini: int = 10,
        verbose: bool = True,
        use_stacking: bool = True,
        include_intraday: bool = True,
        skip_lstm: bool = False,
        include_flow: bool = True,
        include_graph_ising: bool = True,
    ):
        """
        Parameters
        ----------
        top_gemini   : cuántos tickers del ranking enviar a Gemini validator.
        verbose      : imprimir progreso en stdout.
        use_stacking : si True, entrena meta-LR sobre preds base y la usa cuando
                       supere AUC-weight en holdout temporal (KB-060).
        include_intraday : añade microestructura 5m→diario cuando hay parquet cache.
        skip_lstm    : omitir lstm_seq (modo --fast, ~20 min más rápido).
        include_flow : order flow de brokers (data/microstructure/hechos/).
        include_graph_ising : features Graph-Ising (presión emergente sobre
                       grafo de correlación con evidencia WoE).
        """
        self.top_gemini = top_gemini
        self.verbose = verbose
        self.use_stacking = use_stacking
        self.include_intraday = include_intraday and _INTRADAY_AVAILABLE
        self.skip_lstm = skip_lstm
        self.include_flow = include_flow and _FLOW_AVAILABLE
        self.include_graph_ising = include_graph_ising and _GIM_AVAILABLE
        self._macro: dict[str, pd.DataFrame] = {}
        self._regime_cache: dict[str, dict] = {}
        self._news: pd.DataFrame | None = None
        self._news_daily: pd.DataFrame | None = None
        self._horizon_consensus: dict[str, int] = {}
        self._wti_ret5_pct: float | None = None
        self._universe_metrics: dict[str, dict] = {}

    # ── Carga de datos ──────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _load_macro(self) -> None:
        for key, sym in _MACRO_SYMS.items():
            df = store.load(sym)
            if df is not None:
                self._macro[key] = df

    def _load_news(self) -> None:
        """Cache estático (hoy) + panel temporal point-in-time (KB-004)."""
        from src.ml.bet_policy import news_features_enabled
        from src.ml.news_history import history_stats, load_daily_features

        stats = history_stats()
        self._news_daily = load_daily_features()

        # Panel DataBursátil desde snapshots acumulados (preferido BMV)
        try:
            from src.ml.databursatil_news import list_snapshot_dates, build_daily_panel_from_snapshots
            from src.data.universes import UNIVERSES

            if len(list_snapshot_dates()) >= 1:
                bmv_cfg = UNIVERSES.get("🇲🇽 Acciones BMV", {})
                bmv = list(bmv_cfg.get("tickers", {}).keys())
                db_panel = build_daily_panel_from_snapshots(bmv, verbose=False)
                if db_panel is not None and not db_panel.empty:
                    if self._news_daily is None or self._news_daily.empty:
                        self._news_daily = db_panel
                    else:
                        self._news_daily = pd.concat(
                            [self._news_daily, db_panel], ignore_index=True,
                        ).drop_duplicates(subset=["date", "ticker"], keep="last")
        except Exception as e:
            logger.debug(f"databursatil panel: {e}")

        try:
            from src.ml.news import load_cached_news_features

            self._news = load_cached_news_features()
        except Exception as e:
            logger.debug(f"news cache no disponible: {e}")
            self._news = None

        if not news_features_enabled(stats) and self._news is None:
            if self._news_daily is None or self._news_daily.empty:
                self._news = None
                return

    # ── Régimen de mercado ─────────────────────────────────────────────────

    def _regime_factor(self, currency: str) -> tuple[float, dict]:
        """
        Factor multiplicativo por régimen: risk-on amplifica, risk-off reduce.
        MXN usa NAFTRAC, el resto usa S&P 500.
        """
        key = "MXN" if currency == "MXN" else "USD"
        if key in self._regime_cache:
            c = self._regime_cache[key]
            return c["factor"], c["info"]

        factor, info = 1.0, {"estado": "desconocido", "proba": {}}
        try:
            from src.ml.regime import RegimeDetector
            naftrac = self._macro.get("NAFTRAC")
            sp500   = self._macro.get("SP500")
            driver  = naftrac if key == "MXN" else sp500
            if driver is not None and len(driver) > 300:
                det = RegimeDetector()
                det.fit(driver, sp500)
                proba = det.state_proba(driver, sp500)
                estado = max(proba, key=proba.get)
                factor = {"bull": 1.10, "lateral": 1.0, "bear": 0.80}.get(estado, 1.0)
                info = {"estado": estado, "proba": {k: round(v, 3) for k, v in proba.items()}}
        except Exception as e:
            logger.debug(f"regime {key}: {e}")

        self._regime_cache[key] = {"factor": factor, "info": info}
        return factor, info

    # ── Construcción del dataset por universo ───────────────────────────────

    def _extended_cols(self, available: set[str]) -> list[str]:
        """Features extendidas: técnicas + momentum + fundamentales + intraday + news."""
        base = list(EXTENDED_FEATURE_COLS)
        if self._news is not None:
            news_cols = list(NEWS_FEATURE_COLS)
            base.extend(c for c in news_cols if c not in base)
        return [c for c in base if c in available]

    @staticmethod
    def _impute_intraday_cross_section(
        frames: dict[str, pd.DataFrame],
        cols: list[str],
    ) -> None:
        """Rellena NaN intradía con mediana cross-seccional por fecha."""
        avail_cols = [c for c in cols if any(c in df.columns for df in frames.values())]
        if not avail_cols:
            return

        date_medians: dict[str, pd.Series] = {}
        for col in avail_cols:
            stacked = pd.DataFrame(
                {name: df[col] for name, df in frames.items() if col in df.columns}
            )
            date_medians[col] = stacked.median(axis=1)

        for name, df in frames.items():
            for col in avail_cols:
                if col not in df.columns:
                    continue
                med = date_medians[col].reindex(df.index)
                fallback = date_medians[col].median()
                df[col] = df[col].fillna(med).fillna(fallback if pd.notna(fallback) else 0.0)

    def _build_universe_frames(self, tickers: dict[str, str], horizon: int):
        """Carga precios del store, computa features+macro+fundamentales+intraday+news."""
        frames: dict[str, pd.DataFrame] = {}
        for name, sym in tickers.items():
            df = store.load(sym)
            if df is None or len(df) < 150:
                continue
            try:
                feat = compute_features(df)
                if self._macro:
                    feat = add_macro_features(feat, self._macro)
                feat = add_fundamental_features(feat, symbol=sym)
                if self.include_intraday and add_intraday_features is not None:
                    feat = add_intraday_features(feat, name)
                feat["ticker"] = name
                feat["fwd_ret"] = compute_target(df, horizon=horizon)
                if self._news_daily is not None and name in self._news_daily["ticker"].values:
                    from src.ml.news_history import merge_news_history

                    feat = merge_news_history(feat, self._news_daily)
                elif self._news is not None:
                    feat = add_news_features(feat, self._news, ticker_col="ticker")
                else:
                    for col in NEWS_FEATURE_COLS:
                        feat[col] = 0.0
                frames[name] = feat
            except Exception as e:
                logger.debug(f"{name}: {e}")
        if frames:
            frames = add_cross_section_features(frames)
            if self.include_intraday and INTRADAY_FEATURE_COLS:
                self._impute_intraday_cross_section(frames, list(INTRADAY_FEATURE_COLS))
            frames = self._add_flow_and_gim(frames)
        return frames

    def _add_flow_and_gim(self, frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Order flow de brokers + Graph-Ising + tasas + dividendos (todos point-in-time)."""
        if self.include_flow:
            try:
                closes = pd.DataFrame({t: df["Close"] for t, df in frames.items()})
                closes.index = pd.to_datetime(closes.index)
                if getattr(closes.index, "tz", None) is not None:
                    closes.index = closes.index.tz_localize(None)
                flow_panel = compute_flow_panel(list(frames.keys()), close_panel=closes)
                if not flow_panel.empty:
                    n_hit = flow_panel["ticker"].nunique()
                    self._log(f"    order flow: {n_hit} tickers con hechos")
                frames = {
                    t: add_flow_features(df, t, flow_panel) for t, df in frames.items()
                }
            except Exception as e:
                logger.debug(f"flow features: {e}")

        if self.include_graph_ising and len(frames) >= 5:
            try:
                gim_panel = compute_gim_panel(frames)
                if not gim_panel.empty:
                    self._log(
                        f"    graph-ising: {gim_panel['date'].nunique()} días, "
                        f"{gim_panel['ticker'].nunique()} nodos"
                    )
                frames = {
                    t: add_gim_features(df, t, gim_panel) for t, df in frames.items()
                }
            except Exception as e:
                logger.debug(f"graph-ising features: {e}")

        # ── Tasas de mercado de dinero mexicanas ─────────────────────────────
        if _RATES_AVAILABLE:
            try:
                raw_rates = build_rates_panel()
                rates_panel = compute_rates_features(raw_rates) if not raw_rates.empty else None
                if rates_panel is not None and not rates_panel.empty:
                    self._log(f"    tasas: panel {len(rates_panel)}d "
                              f"({rates_panel.index[0].date()} → {rates_panel.index[-1].date()})")
                    frames = {
                        t: add_rates_features(df, rates_panel) for t, df in frames.items()
                    }
                else:
                    self._log("    tasas: panel vacío (guardando snapshot de hoy…)")
                    fetch_tasas_snapshot(sleep=0)
                    frames = {t: add_rates_features(df, None) for t, df in frames.items()}
            except Exception as e:
                logger.debug(f"rates features: {e}")

        # ── Dividendos ex-date ────────────────────────────────────────────────
        if _DIVIDENDS_AVAILABLE:
            try:
                import os
                token = os.environ.get("DATABURSATIL_API_TOKEN", "")
                tickers_list = list(frames.keys())
                dividends = fetch_all_dividends(tickers_list, token=token, sleep=0.4)
                if dividends:
                    # Construir index de fechas desde el primer frame disponible
                    first_df = next(iter(frames.values()))
                    date_idx = pd.DatetimeIndex(first_df.index)
                    closes = pd.DataFrame(
                        {t: df["Close"] for t, df in frames.items()
                         if "Close" in df.columns}
                    )
                    div_panel = build_dividend_panel(dividends, date_idx, close_panel=closes)
                    n_div = len([t for t in tickers_list if t in dividends])
                    self._log(f"    dividendos: {n_div} tickers con historial")
                    frames = {
                        t: add_dividend_features(df, t, div_panel) for t, df in frames.items()
                    }
            except Exception as e:
                logger.debug(f"dividend features: {e}")

        return frames

    # ── Modelos base por universo ───────────────────────────────────────────

    def _run_universe(self, uname: str, cfg: dict) -> list[dict]:
        tickers  = cfg["tickers"]
        horizon  = cfg["horizonte"]
        currency = cfg["moneda"]
        min_n    = cfg["min_tickers"]

        self._log(f"\n  [{uname}] cargando {len(tickers)} tickers...")
        frames = self._build_universe_frames(tickers, horizon)
        if len(frames) < min_n:
            self._log(f"  [{uname}] insuficientes tickers ({len(frames)}) — skip")
            return []

        feat_cols = self._extended_cols(set(next(iter(frames.values())).columns))
        from src.ml.bet_policy import news_features_enabled

        include_news = news_features_enabled() or any(
            c in feat_cols for c in NEWS_FEATURE_COLS
        )
        priority_feats = select_priority_features(feat_cols, include_news=include_news)
        ens_feats = priority_feats if len(priority_feats) >= 8 else feat_cols

        # Imputar features con cobertura parcial para no perder filas históricas.
        # Se aplica a: fundamentales (trimestrales), tasas (acumulación diaria),
        # dividendos, flow y GIM (solo últimos ~12m de datos).
        _SPARSE_COLS = (
            FUNDAMENTAL_FEATURE_COLS
            + list(RATES_FEATURE_COLS)
            + list(DIVIDEND_FEATURE_COLS)
            + list(FLOW_FEATURE_COLS)
            + list(GIM_FEATURE_COLS)
        )
        for name in list(frames.keys()):
            df = frames[name]
            for col in _SPARSE_COLS:
                if col in df.columns and df[col].isna().any():
                    med = df[col].median()
                    frames[name][col] = df[col].fillna(med if pd.notna(med) else 0.0)

        # Panel completo para entrenamiento
        rows = []
        for df in frames.values():
            d = df.dropna(subset=feat_cols + ["fwd_ret"]).copy()
            if len(d) >= 100:
                rows.append(d)
        if not rows:
            return []

        full = pd.concat(rows).sort_index()

        # Rank-normalización cross-seccional por fecha
        g = full.groupby(full.index)
        for c in feat_cols:
            full[f"rn_{c}"] = g[c].rank(pct=True)
        full["pct_fwd"] = g["fwd_ret"].rank(pct=True)
        full["y_rel"] = (g["fwd_ret"].rank(pct=True) > 0.5).astype(int)
        full["y_abs"] = (full["fwd_ret"] > 0).astype(int)

        rn_avail = [f"rn_{c}" for c in feat_cols]
        xs_rn = select_priority_rn_cols(rn_avail)
        full = full.dropna(subset=rn_avail + ["pct_fwd"])
        if len(full) < 200:
            return []

        cut = full.index[int(len(full) * 0.80)]
        train, valid = full[full.index < cut], full[full.index >= cut]
        if len(valid) < 30 or train["y_rel"].nunique() < 2:
            train, valid = full, full

        signals_by_model: dict[str, dict[str, float]] = {}
        auc: dict[str, float] = {}
        valid_preds: dict[str, np.ndarray] = {}
        meta_lr: LogisticRegression | None = None
        use_stack_fusion = False
        stack_auc = fusion_auc = 0.5

        # ── Modelo 1: xs_lgbm ─────────────────────────────────────────────────
        try:
            import lightgbm as lgb
            reg = lgb.LGBMRegressor(**_LGBM_BASE)
            reg.fit(train[xs_rn].values, train["pct_fwd"].values)
            p_val = reg.predict(valid[xs_rn].values)
            auc["xs_lgbm"] = self._safe_auc(valid["y_rel"].values, p_val)
            valid_preds["xs_lgbm"] = p_val
            today = self._today_rn(frames, feat_cols)
            if today is not None and not today.empty:
                preds = reg.predict(today[xs_rn].fillna(0.5).values)
                signals_by_model["xs_lgbm"] = dict(zip(today["ticker"], _minmax(preds)))
        except Exception as e:
            logger.debug(f"xs_lgbm {uname}: {e}")

        # ── Modelo 2: EnsembleModel ─────────────────────────────────────────
        try:
            from src.ml.ensemble import EnsembleModel
            ens = EnsembleModel(top_features=min(12, len(ens_feats)), calibrate=True)
            ens.fit(train, feature_cols=ens_feats, target_col="y_rel")
            p_val = ens.predict_proba(valid, ens_feats)
            auc["ensemble"] = self._safe_auc(valid["y_rel"].values, p_val)
            valid_preds["ensemble"] = p_val
            today = self._today_rows(frames, ens_feats)
            if today is not None and not today.empty:
                preds = ens.predict_proba(today, ens_feats)
                signals_by_model["ensemble"] = dict(zip(today["ticker"], preds))
        except Exception as e:
            logger.debug(f"ensemble {uname}: {e}")

        # ── Modelo 3: heurística ────────────────────────────────────────────
        try:
            hscores: dict[str, float] = {}
            h_val_raw = []
            for _, row in valid.iterrows():
                s, _ = compute_signal_score(row)
                h_val_raw.append(s)
            if h_val_raw:
                valid_preds["heuristic"] = _minmax(np.array(h_val_raw))
                auc["heuristic"] = self._safe_auc(
                    valid["y_rel"].values, valid_preds["heuristic"]
                )
            for name, df in frames.items():
                clean = df.dropna(subset=feat_cols)
                if clean.empty:
                    continue
                s, _ = compute_signal_score(clean.iloc[-1])
                hscores[name] = s
            if hscores:
                names = list(hscores.keys())
                norm = _minmax(np.array([hscores[n] for n in names]))
                signals_by_model["heuristic"] = dict(zip(names, norm))
                if "heuristic" not in auc:
                    auc["heuristic"] = 0.52
        except Exception as e:
            logger.debug(f"heuristic {uname}: {e}")

        # ── Modelo 4: lstm_seq (KB-012) ─────────────────────────────────────
        lstm_predictor = None
        if not self.skip_lstm and _LSTM_AVAILABLE and LSTMSeqPredictor is not None:
            try:
                lstm_predictor = LSTMSeqPredictor(sequence_length=30, epochs=12)
                lstm_predictor.fit(train, feature_cols=feat_cols, target_col="y_rel")
                if lstm_predictor.is_fitted:
                    p_val_lstm = lstm_predictor.predict_proba(valid, feat_cols)
                    mask = np.isfinite(p_val_lstm)
                    if mask.sum() >= 30:
                        auc_lstm = self._safe_auc(
                            valid["y_rel"].values[mask], p_val_lstm[mask]
                        )
                        auc["lstm_seq"] = auc_lstm
                        valid_preds["lstm_seq"] = p_val_lstm
                        today_lstm = lstm_predictor.predict_proba_today(frames, feat_cols)
                        if today_lstm:
                            signals_by_model["lstm_seq"] = today_lstm
                        if auc_lstm < LSTM_AUC_THRESHOLD:
                            logger.info(
                                f"[{uname}] lstm_seq excluido de fusión: "
                                f"AUC={auc_lstm:.3f} < {LSTM_AUC_THRESHOLD}"
                            )
                            signals_by_model.pop("lstm_seq", None)
                            valid_preds.pop("lstm_seq", None)
                            auc.pop("lstm_seq", None)
                    else:
                        logger.info(f"[{uname}] lstm_seq: pocas preds válidas en holdout")
                else:
                    logger.info(f"[{uname}] lstm_seq no fitted — excluido de fusión")
            except Exception as e:
                logger.debug(f"lstm_seq {uname}: {e}")

        # ── Modelo extra: woe_abs (producción) ──────────────────────────────
        try:
            from src.ml.ensemble import EnsembleModel
            base_feat_cols = [c for c in FEATURE_COLS if c in feat_cols]
            if base_feat_cols and train["y_abs"].nunique() > 1:
                ens_abs = EnsembleModel(top_features=min(12, len(base_feat_cols)), calibrate=True)
                ens_abs.fit(train, feature_cols=base_feat_cols, target_col="y_abs")
                p_val_abs = ens_abs.predict_proba(valid, base_feat_cols)
                auc["woe_abs"] = self._safe_auc(valid["y_abs"].values, p_val_abs)
                today = self._today_rows(frames, base_feat_cols)
                if today is not None and not today.empty:
                    preds = ens_abs.predict_proba(today, base_feat_cols)
                    signals_by_model["woe_abs"] = dict(zip(today["ticker"], preds))
        except Exception as e:
            logger.debug(f"woe_abs {uname}: {e}")

        if not signals_by_model:
            return []

        # ── Stacking meta-LR vs AUC-weight (KB-060) ──────────────────────────
        stack_models = [
            m for m in STACK_BASE_MODELS
            if m in valid_preds and m in signals_by_model
        ]
        if self.use_stacking and len(stack_models) >= 2:
            X_meta = np.column_stack([valid_preds[m] for m in stack_models])
            mask = np.all(np.isfinite(X_meta), axis=1)
            if mask.sum() >= 30 and len(np.unique(valid["y_rel"].values[mask])) >= 2:
                meta_lr = LogisticRegression(max_iter=500, random_state=42)
                meta_lr.fit(X_meta[mask], valid["y_rel"].values[mask])
                stack_auc = self._safe_auc(
                    valid["y_rel"].values[mask],
                    meta_lr.predict_proba(X_meta[mask])[:, 1],
                )
                fusion_scores = self._auc_weight_blend(
                    {m: valid_preds[m][mask] for m in stack_models}, auc, stack_models
                )
                fusion_auc = self._safe_auc(valid["y_rel"].values[mask], fusion_scores)
                use_stack_fusion = stack_auc >= fusion_auc
                self._log(
                    f"  [{uname}] stacking AUC={stack_auc:.3f} vs "
                    f"AUC-weight={fusion_auc:.3f} → "
                    f"{'stacking' if use_stack_fusion else 'AUC-weight'}"
                )

        regime_factor, regime_info = self._regime_factor(currency)

        # ── Métricas OOS + incertidumbre por universo ───────────────────────
        y_valid = valid["y_rel"].values
        model_metrics = compute_universe_model_metrics(
            valid_preds, y_valid, auc, models=list(valid_preds.keys())
        )
        stack_scores_arr: np.ndarray | None = None
        if meta_lr is not None and len(stack_models) >= 2:
            X_all = np.column_stack([valid_preds[m] for m in stack_models])
            mask_all = np.all(np.isfinite(X_all), axis=1)
            if mask_all.sum() >= 30:
                stack_scores_arr = meta_lr.predict_proba(X_all[mask_all])[:, 1]
                y_valid = y_valid[mask_all]
        fusion_meta = compute_fusion_metrics(
            stack_auc=stack_auc,
            fusion_auc=fusion_auc,
            use_stacking=use_stack_fusion,
            y_rel=y_valid,
            stack_scores=stack_scores_arr,
        )
        slug = cfg["slug"]
        self._universe_metrics[slug] = {
            "universe": uname,
            "slug": slug,
            "n_tickers": len(frames),
            "models": model_metrics,
            "fusion": fusion_meta,
            "regime_proba": regime_info.get("proba", {}),
            "horizon_d": horizon,
        }

        # ── Fusión por ticker ────────────────────────────────────────────────
        all_tickers = set().union(*[set(d.keys()) for d in signals_by_model.values()])

        results = []
        for tk in all_tickers:
            sigs = {m: signals_by_model[m].get(tk) for m in signals_by_model
                    if tk in signals_by_model[m]}
            sigs = {m: float(v) for m, v in sigs.items() if v is not None and np.isfinite(v)}
            if not sigs:
                continue

            base_sigs = {m: sigs[m] for m in STACK_BASE_MODELS if m in sigs}
            extra_sigs = {m: sigs[m] for m in sigs if m not in STACK_BASE_MODELS}

            if use_stack_fusion and meta_lr is not None and len(base_sigs) >= 2:
                ordered = [m for m in stack_models if m in base_sigs]
                X_t = np.array([[base_sigs[m] for m in ordered]])
                conviction = float(meta_lr.predict_proba(X_t)[0, 1])
                weights = {m: 1.0 / len(ordered) for m in ordered}
                for m, v in extra_sigs.items():
                    w = max(auc.get(m, 0.5) - 0.5, 0.0)
                    if w > 0:
                        conviction = 0.85 * conviction + 0.15 * v
                wsum = len(ordered)
            else:
                weights = {m: max(auc.get(m, 0.5) - 0.5, 0.0) for m in sigs}
                wsum = sum(weights.values())
                if wsum <= 0:
                    weights = {m: 1.0 for m in sigs}
                    wsum = len(sigs)
                conviction = sum(sigs[m] * weights[m] / wsum for m in sigs)

            stack_sigs = {**base_sigs, **extra_sigs}
            disagreement = float(np.std(list(stack_sigs.values()))) if len(stack_sigs) > 1 else 0.0
            conviction_adj = conviction * regime_factor

            news_sent = self._news_sent(tk)
            news_delta = self._news_delta(tk)
            news_alerta = news_sent is not None and news_sent < -0.3
            if news_alerta:
                conviction_adj *= 0.5

            n_hz = self._horizon_consensus.get(tk, 0)
            conviction_kb, kb_flags, kb_meta = apply_kb_modulators(
                ticker=tk,
                universe_slug=cfg["slug"],
                conviction=conviction_adj,
                disagreement=disagreement,
                signals=stack_sigs,
                news_sent=news_sent,
                news_sent_delta=news_delta,
                regime=regime_info["estado"],
                n_horizontes=n_hz,
                wti_ret5_pct=self._wti_ret5_pct,
            )

            price = self._last_price(frames.get(tk))
            results.append({
                "ticker": tk,
                "universe": uname,
                "universe_slug": cfg["slug"],
                "moneda": currency,
                "precio": round(price, 2) if price else None,
                "conviction_raw": round(conviction, 4),
                "conviction": round(float(np.clip(conviction_kb, 0, 1.5)), 4),
                "disagreement": round(disagreement, 4),
                "signals": {m: round(v, 4) for m, v in sigs.items()},
                "model_weights": {m: round(weights.get(m, 0) / max(wsum, 1), 3) for m in sigs},
                "fusion_mode": "stacking" if use_stack_fusion else "auc_weight",
                "regime": regime_info["estado"],
                "regime_factor": regime_factor,
                "news_sent": round(news_sent, 3) if news_sent is not None else None,
                "news_alerta": news_alerta,
                "n_horizontes": n_hz,
                "kb_flags": kb_flags,
                "kb_meta": kb_meta,
                "horizonte_d": horizon,
            })

        for r in results:
            r["model_auc"] = {m: round(auc.get(m, 0.5), 4) for m in r["signals"]}
            if self.use_stacking:
                r["stack_auc"] = round(stack_auc, 4)
                r["fusion_auc"] = round(fusion_auc, 4)

        self._universe_metrics[slug]["n_scored"] = len(results)

        auc_log = {k: round(v, 3) for k, v in auc.items()}
        lstm_tag = " +lstm" if "lstm_seq" in auc else ""
        self._log(
            f"  [{uname}] {len(results)} tickers · régimen {regime_info['estado']} "
            f"(×{regime_factor}) · AUC {auc_log}{lstm_tag}"
        )
        return results

    # ── Helpers de predicción "hoy" ─────────────────────────────────────────

    def _today_rows(self, frames, feat_cols):
        """Última fila de cada ticker (features absolutas) para Ensemble."""
        rows = []
        for name, df in frames.items():
            clean = df.dropna(subset=feat_cols)
            if not clean.empty:
                r = clean.iloc[[-1]].copy()
                r["ticker"] = name
                rows.append(r)
        return pd.concat(rows) if rows else None

    def _today_rn(self, frames, feat_cols):
        """Última fila rank-normalizada cross-seccional (para xs_lgbm)."""
        last = {}
        for name, df in frames.items():
            clean = df.dropna(subset=feat_cols)
            if not clean.empty:
                last[name] = clean.iloc[-1]
        if not last:
            return None
        rows = []
        for name, row in last.items():
            rn = {"ticker": name}
            for c in feat_cols:
                vals = [r[c] for r in last.values() if c in r.index and not pd.isna(r[c])]
                rn[f"rn_{c}"] = (sum(1 for v in vals if v <= row.get(c, 0)) / len(vals)
                                if vals else 0.5)
            rows.append(rn)
        return pd.DataFrame(rows)

    @staticmethod
    def _auc_weight_blend(
        preds: dict[str, np.ndarray],
        auc: dict[str, float],
        models: list[str],
    ) -> np.ndarray:
        """Fusión AUC-weight fila a fila (para comparar vs stacking en holdout)."""
        weights = {m: max(auc.get(m, 0.5) - 0.5, 0.0) for m in models}
        wsum = sum(weights.values()) or len(models)
        if wsum <= 0:
            weights = {m: 1.0 for m in models}
            wsum = len(models)
        out = np.zeros(len(next(iter(preds.values()))))
        for m in models:
            out += preds[m] * weights[m] / wsum
        return out

    @staticmethod
    def _safe_auc(y, p) -> float:
        try:
            if len(np.unique(y)) < 2:
                return 0.5
            return float(roc_auc_score(y, p))
        except Exception:
            return 0.5

    @staticmethod
    def _last_price(df):
        if df is None or df.empty or "Close" not in df.columns:
            return None
        return float(df["Close"].iloc[-1])

    def _news_sent(self, ticker: str):
        if self._news is None or ticker not in self._news.index:
            return None
        try:
            return float(self._news.loc[ticker, "news_sent_score"])
        except Exception:
            return None

    def _news_delta(self, ticker: str) -> float | None:
        if self._news is None or ticker not in self._news.index:
            return None
        try:
            val = self._news.loc[ticker, "news_sent_delta"]
            return float(val) if pd.notna(val) else None
        except Exception:
            return None

    # ── Insights cross-universo ─────────────────────────────────────────────

    def _build_insights(self, ranking: list[dict]) -> list[dict]:
        insights = []
        if not ranking:
            return insights

        df = pd.DataFrame(ranking)

        # 1. Convicción alta + acuerdo
        strong = df[(df["conviction"] > 0.6) & (df["disagreement"] < 0.15)]
        for _, r in strong.head(3).iterrows():
            insights.append({
                "tipo": "Consenso fuerte",
                "ticker": r["ticker"],
                "texto": f"{r['ticker']} ({r['universe_slug']}): convicción {r['conviction']:.2f}, "
                         f"modelos de acuerdo (desac. {r['disagreement']:.2f}). Régimen {r['regime']}.",
            })

        # 2. Divergencias (alta convicción pero alto desacuerdo)
        diverg = df[(df["conviction"] > 0.5) & (df["disagreement"] > 0.25)]
        for _, r in diverg.head(2).iterrows():
            insights.append({
                "tipo": "Divergencia",
                "ticker": r["ticker"],
                "texto": f"{r['ticker']}: modelos en desacuerdo (desac. {r['disagreement']:.2f}) — "
                         f"señales {r['signals']}. Revisar antes de operar.",
            })

        # 3. Vetos por noticias
        vetoed = df[df["news_alerta"]]
        for _, r in vetoed.head(2).iterrows():
            insights.append({
                "tipo": "Veto por noticias",
                "ticker": r["ticker"],
                "texto": f"{r['ticker']}: score técnico alto pero news_sent {r['news_sent']:.2f} "
                         f"redujo la convicción a la mitad.",
            })

        # 4. Concentración sectorial / universo en top-10
        top10 = df.nlargest(10, "conviction")
        uni_counts = top10["universe_slug"].value_counts()
        if uni_counts.iloc[0] >= 6:
            insights.append({
                "tipo": "Concentración",
                "ticker": "",
                "texto": f"{uni_counts.iloc[0]}/10 del top provienen de '{uni_counts.index[0]}'. "
                         f"Considerar diversificar entre clases de activo.",
            })

        # 5. Régimen dominante
        regimes = df.groupby("moneda")["regime"].first().to_dict()
        insights.append({
            "tipo": "Régimen",
            "ticker": "",
            "texto": "Régimen de mercado: " + " · ".join(f"{k}={v}" for k, v in regimes.items()),
        })

        return insights

    # ── Pipeline completo ────────────────────────────────────────────────────

    def run(self) -> dict:
        self._log("\n" + "═" * 62)
        self._log("  SUPER MODELO — Meta-orquestador cross-universo")
        self._log("═" * 62)

        self._load_macro()
        self._load_news()
        self._horizon_consensus = load_horizon_consensus()
        self._wti_ret5_pct = get_wti_ret5_pct(self._macro)

        all_results: list[dict] = []
        for uname, cfg in UNIVERSES.items():
            all_results.extend(self._run_universe(uname, cfg))

        # Ranking maestro global por convicción
        all_results.sort(key=lambda r: r["conviction"], reverse=True)
        for i, r in enumerate(all_results):
            r["rank_global"] = i + 1
            slug = r.get("universe_slug", "")
            enrich_ranking_item(r, self._universe_metrics)

        active_models: set[str] = set()
        for um in self._universe_metrics.values():
            active_models.update(um.get("models", {}).keys())

        insights = self._build_insights(all_results)

        self._log(f"\n  Ranking maestro: {len(all_results)} tickers")
        self._log("  Top-10 por convicción:")
        for r in all_results[:10]:
            self._log(f"    #{r['rank_global']:<2} {r['ticker']:<10} "
                      f"conv={r['conviction']:.3f}  desac={r['disagreement']:.3f}  "
                      f"[{r['universe_slug']}] {r['regime']}")

        return {
            "fecha": date.today().isoformat(),
            "n_tickers": len(all_results),
            "master_ranking": all_results,
            "insights": insights,
            "top_gemini": self.top_gemini,
            "model_registry": build_model_registry(active_models),
            "universe_metrics": self._universe_metrics,
        }
