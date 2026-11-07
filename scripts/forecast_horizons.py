"""
Pronóstico a 30/60/90 días hábiles (~21/42/63 sesiones) por "intento" de
modelo, con banda de incertidumbre derivada de volatilidad realizada.

Por qué esto no es "correr cada modelo tal cual": LSTM, TCN, WoE y el
régimen HMM NO pronostican precio — son clasificadores o detectores de
estado (P(up), P(bull)). Para poder dibujar "real vs pronosticado" en la
misma gráfica, cada intento expone un `signal_fn(df_features) -> float`
(point-in-time, solo con datos hasta el corte) que se CALIBRA a un retorno
esperado del horizonte vía una regresión lineal simple contra
`compute_target()`, ajustada SOLO con datos anteriores al corte (purga
ingenua: se excluyen las filas cuyo target ya mira más allá del corte).

Intentos incluidos hoy (los que sí producen una señal continua o un estado
discreto con retorno condicional calculable):
  - lightgbm_ranking : LGBMRegressor directo sobre features (igual
                       intención que generate_all_signals.py, pero
                       single-ticker — no cross-seccional — por alcance).
  - momentum_12_1     : feature ret_20d, calibrado por regresión.
  - woe_score         : WoE combinado de 3 features, calibrado por regresión.
  - regimen_hmm       : estado bull/lateral/bear del IPC, con retorno medio
                       condicional por estado.

NO incluidos (requieren TensorFlow + entrenamiento secuencial pesado, o un
panel cross-seccional completo): LSTM, TCN, Ising. Quedan como siguiente
paso si se necesitan — mismo patrón de signal_fn + calibración.

Uso:
    python scripts/forecast_horizons.py
    python scripts/forecast_horizons.py --tickers WALMEX,SPY
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.store import get as store_get, ALL_SYMBOLS
from src.ml.features import compute_features, compute_target, FEATURE_COLS

OUT_DIR = ROOT / "data" / "eval" / "forecast_demo"

HORIZONS = {"30d": 21, "60d": 42, "90d": 63}   # sesiones ~1/2/3 meses
# Cualquier ticker con parquet en el store (data/market/) — antes limitado a
# WALMEX/SPY a mano; store.get() ya resuelve nombre->símbolo vía ALL_SYMBOLS,
# así que basta exponer el catálogo completo (~95 tickers hoy).
TICKERS = {name: name for name in sorted(ALL_SYMBOLS)}
MIN_TRAIN_ROWS = 300
VOL_WINDOW = 60       # días para estimar volatilidad diaria reciente
Z_SCORE = 1.645        # banda ~90% (random walk con deriva, escalada por sqrt(t))

# Paleta fija (ver skill dataviz): identidad por color, no por rango.
# Rojo = lo real (lo que de verdad pasó); azul = lo pronosticado por cada
# intento. Los 4 modelos comparten el azul, se distinguen por trazo.
COLOR_REAL = "#C0392B"
COLOR_FORECAST = "#1f6feb"
COLOR_BAND = "#1f6feb"
MODEL_STYLES = {
    "lightgbm_ranking": "-",
    "momentum_12_1": "--",
    "woe_score": "-.",
    "regimen_hmm": ":",
}


# ───────────────────────── Intentos (signal_fn) ──────────────────────────────

def _purged_train(feat: pd.DataFrame, cutoff_idx: int, horizon: int) -> pd.DataFrame:
    """Filas de entrenamiento cuyo target ya estaba resuelto antes del corte."""
    train_end = max(0, cutoff_idx - horizon - 1)
    return feat.iloc[:train_end]


def _calibrate(signal: pd.Series, fwd_ret: pd.Series) -> tuple[float, float]:
    """Regresión lineal simple signal -> retorno esperado del horizonte (%)."""
    df = pd.DataFrame({"s": signal, "y": fwd_ret}).dropna()
    if len(df) < 50 or df["s"].std() == 0:
        return 0.0, float(df["y"].mean()) if len(df) else 0.0
    b = float(np.cov(df["s"], df["y"])[0, 1] / np.var(df["s"]))
    a = float(df["y"].mean() - b * df["s"].mean())
    return b, a


def signal_momentum_12_1(feat: pd.DataFrame) -> pd.Series:
    return feat["ret_20d"]


def signal_woe(feat: pd.DataFrame, cutoff_idx: int, horizon: int) -> pd.Series:
    from src.ml.woe import compute_all_woe, woe_transform

    train = _purged_train(feat, cutoff_idx, horizon)
    cols = ["rsi_14", "ret_20d", "vol_ratio_20"]
    cols = [c for c in cols if c in feat.columns]
    target = compute_target(feat, horizon=horizon)
    train_target = target.loc[train.index]
    woe_map = compute_all_woe(train.assign(y=train_target), cols, target_col="y")
    transformed = woe_transform(feat, woe_map)
    woe_cols = [f"{c}_woe" for c in cols if f"{c}_woe" in transformed.columns]
    if not woe_cols:
        return pd.Series(0.0, index=feat.index)
    return transformed[woe_cols].sum(axis=1)


def signal_regimen(ticker_df: pd.DataFrame, cutoff_idx: int) -> tuple[pd.Series, str]:
    """Retorna (serie de estado por fecha, estado en el corte)."""
    from src.ml.regime import RegimeDetector

    naftrac = store_get("NAFTRAC")
    sp500 = store_get("SP500")
    if naftrac is None:
        raise RuntimeError("Falta NAFTRAC.MX en data/market/ para el régimen HMM")

    cutoff_date = ticker_df.index[cutoff_idx]
    detector = RegimeDetector().fit(naftrac, sp500, train_cutoff=str(cutoff_date.date()))
    states = detector.predict_series(naftrac, sp500)
    states = states.reindex(ticker_df.index, method="ffill")
    return states, states.iloc[cutoff_idx]


def forecast_lightgbm(feat: pd.DataFrame, cutoff_idx: int, horizon: int) -> float:
    import lightgbm as lgb

    target = compute_target(feat, horizon=horizon)
    train = _purged_train(feat, cutoff_idx, horizon)
    cols = [c for c in FEATURE_COLS if c in feat.columns]
    train_x = train[cols].copy()
    train_y = target.loc[train.index]
    mask = train_x.notna().all(axis=1) & train_y.notna()
    train_x, train_y = train_x[mask], train_y[mask]
    if len(train_x) < MIN_TRAIN_ROWS:
        return float("nan")
    model = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, verbose=-1)
    model.fit(train_x.values, train_y.values)
    today_x = feat[cols].iloc[[cutoff_idx]].fillna(train_x.median())
    return float(model.predict(today_x.values)[0])


def forecast_momentum(feat: pd.DataFrame, cutoff_idx: int, horizon: int) -> float:
    target = compute_target(feat, horizon=horizon)
    train = _purged_train(feat, cutoff_idx, horizon)
    b, a = _calibrate(signal_momentum_12_1(train), target.loc[train.index])
    s_today = float(feat["ret_20d"].iloc[cutoff_idx])
    return a + b * s_today


def forecast_woe(feat: pd.DataFrame, cutoff_idx: int, horizon: int) -> float:
    target = compute_target(feat, horizon=horizon)
    woe_series = signal_woe(feat, cutoff_idx, horizon)
    train = _purged_train(feat, cutoff_idx, horizon)
    b, a = _calibrate(woe_series.loc[train.index], target.loc[train.index])
    s_today = float(woe_series.iloc[cutoff_idx])
    return a + b * s_today


def forecast_regimen(ticker_df: pd.DataFrame, feat: pd.DataFrame, cutoff_idx: int, horizon: int) -> float:
    target = compute_target(feat, horizon=horizon)
    states, state_today = signal_regimen(ticker_df, cutoff_idx)
    train = _purged_train(feat, cutoff_idx, horizon)
    train_states = states.loc[train.index]
    cond_mean = target.loc[train.index].groupby(train_states).mean()
    return float(cond_mean.get(state_today, target.loc[train.index].mean()))


MODELS = {
    "lightgbm_ranking": forecast_lightgbm,
    "momentum_12_1": forecast_momentum,
    "woe_score": forecast_woe,
    # "regimen_hmm" se maneja aparte: necesita ticker_df, no solo feat
}


# ───────────────────────── Modo rodante (día a día) ──────────────────────────
#
# El modo "estático" de arriba pronostica UNA vez en el corte y reparte ese
# único número a lo largo del horizonte — por eso la línea azul sale casi
# recta. Aquí cada modelo se AJUSTA una sola vez (barato: nada de reentrenar
# 63 veces) pero se EVALÚA día por día dentro de la ventana de validación,
# siempre con el feature de ESE día — así la línea azul sí reacciona.
#
# Cada ajuste produce 3 salidas por día: dirección (sube/baja), valor
# (precio pronosticado para el día siguiente) y ganancia esperada (%).
# Calibrado a 1 día adelante (horizon=1 en compute_target/purga), no al
# horizonte completo — es un pronóstico "mañana", repetido cada día.

def fit_lightgbm(feat: pd.DataFrame, cutoff_idx: int) -> tuple | None:
    import lightgbm as lgb

    target = compute_target(feat, horizon=1)
    train = _purged_train(feat, cutoff_idx, horizon=1)
    cols = [c for c in FEATURE_COLS if c in feat.columns]
    train_x = train[cols].copy()
    train_y = target.loc[train.index]
    mask = train_x.notna().all(axis=1) & train_y.notna()
    train_x, train_y = train_x[mask], train_y[mask]
    if len(train_x) < MIN_TRAIN_ROWS:
        return None
    model = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, verbose=-1)
    model.fit(train_x.values, train_y.values)
    return ("lgbm", model, cols, train_x.median())


def fit_momentum(feat: pd.DataFrame, cutoff_idx: int) -> tuple:
    target = compute_target(feat, horizon=1)
    train = _purged_train(feat, cutoff_idx, horizon=1)
    b, a = _calibrate(signal_momentum_12_1(train), target.loc[train.index])
    return ("linear", a, b, "ret_20d")


def fit_woe(feat: pd.DataFrame, cutoff_idx: int) -> tuple:
    target = compute_target(feat, horizon=1)
    woe_series = signal_woe(feat, cutoff_idx, horizon=1)
    train = _purged_train(feat, cutoff_idx, horizon=1)
    b, a = _calibrate(woe_series.loc[train.index], target.loc[train.index])
    return ("series_linear", a, b, woe_series)


def fit_regimen(ticker_df: pd.DataFrame, feat: pd.DataFrame, cutoff_idx: int) -> tuple:
    target = compute_target(feat, horizon=1)
    states, _ = signal_regimen(ticker_df, cutoff_idx)
    train = _purged_train(feat, cutoff_idx, horizon=1)
    cond_mean = target.loc[train.index].groupby(states.loc[train.index]).mean()
    return ("states", cond_mean, states, float(target.loc[train.index].mean()))


FIT_ROLLING = {
    "lightgbm_ranking": fit_lightgbm,
    "momentum_12_1": fit_momentum,
    "woe_score": fit_woe,
}


def apply_fitted(state: tuple, feat: pd.DataFrame, idx: int) -> float:
    """1 predicción puntual (retorno % esperado mañana) con el estado ya ajustado."""
    kind = state[0]
    if kind == "lgbm":
        _, model, cols, fill = state
        row = feat[cols].iloc[[idx]].fillna(fill)
        return float(model.predict(row.values)[0])
    if kind == "linear":
        _, a, b, col = state
        return a + b * float(feat[col].iloc[idx])
    if kind == "series_linear":
        _, a, b, series = state
        return a + b * float(series.iloc[idx])
    if kind == "states":
        _, cond_mean, states, fallback = state
        s = states.iloc[idx]
        return float(cond_mean.get(s, fallback))
    raise ValueError(f"estado desconocido: {kind}")


def run_rolling(ticker_name: str, model_name: str, horizon_label: str, horizon: int) -> dict | None:
    """Pronóstico día a día (dirección, valor, ganancia esperada) durante la
    ventana de validación de `horizon` sesiones, comparado contra lo real."""
    symbol = TICKERS[ticker_name]
    df = store_get(symbol)
    if df is None or len(df) < MIN_TRAIN_ROWS + horizon + 10:
        return None

    feat = compute_features(df)
    cutoff_idx = len(feat) - horizon - 1
    if cutoff_idx < MIN_TRAIN_ROWS:
        return None
    cutoff_date = feat.index[cutoff_idx]

    try:
        if model_name == "regimen_hmm":
            state = fit_regimen(df, feat, cutoff_idx)
        else:
            state = FIT_ROLLING[model_name](feat, cutoff_idx)
    except Exception as exc:
        print(f"  [rolling {model_name}] {ticker_name} {horizon_label}: ERROR ajuste {exc}")
        return None
    if state is None:
        return None

    rows = []
    for idx in range(cutoff_idx, cutoff_idx + horizon):
        if idx + 1 >= len(df):
            break
        price_today = float(df["Close"].iloc[idx])
        price_tomorrow_real = float(df["Close"].iloc[idx + 1])
        try:
            gain_pred_pct = apply_fitted(state, feat, idx)
        except Exception:
            continue
        if not np.isfinite(gain_pred_pct):
            continue
        price_pred = price_today * (1 + gain_pred_pct / 100.0)
        direction_pred = "sube" if gain_pred_pct >= 0 else "baja"
        gain_real_pct = (price_tomorrow_real / price_today - 1) * 100
        direction_real = "sube" if gain_real_pct >= 0 else "baja"
        rows.append({
            "date": feat.index[idx + 1], "direction_pred": direction_pred,
            "price_pred": price_pred, "gain_pred_pct": gain_pred_pct,
            "price_real": price_tomorrow_real, "gain_real_pct": gain_real_pct,
            "hit": direction_pred == direction_real,
        })

    if not rows:
        return None
    daily = pd.DataFrame(rows)
    return {
        "ticker": ticker_name, "model": model_name, "horizon": horizon_label,
        "cutoff_date": cutoff_date, "daily": daily,
        "hit_rate": float(daily["hit"].mean()),
        "mean_gain_pred": float(daily["gain_pred_pct"].mean()),
        "mean_gain_real": float(daily["gain_real_pct"].mean()),
    }


def run_sin_ancla(ticker_name: str, model_name: str, horizon_label: str, horizon: int) -> dict | None:
    """Como run_rolling, pero el precio pronosticado NO se reinicia al precio
    real cada día — se compone solo con lo que el modelo predice. Prueba
    honesta y justa porque la ventana es corta (30/60/90d), no años: así el
    modelo no se queda viendo un mundo viejo (ver mesh 'sin ancla')."""
    symbol = TICKERS[ticker_name]
    df = store_get(symbol)
    if df is None or len(df) < MIN_TRAIN_ROWS + horizon + 10:
        return None

    feat = compute_features(df)
    cutoff_idx = len(feat) - horizon - 1
    if cutoff_idx < MIN_TRAIN_ROWS:
        return None
    cutoff_date = feat.index[cutoff_idx]
    price0 = float(df["Close"].iloc[cutoff_idx])

    try:
        if model_name == "regimen_hmm":
            state = fit_regimen(df, feat, cutoff_idx)
        else:
            state = FIT_ROLLING[model_name](feat, cutoff_idx)
    except Exception as exc:
        print(f"  [sin_ancla {model_name}] {ticker_name} {horizon_label}: ERROR ajuste {exc}")
        return None
    if state is None:
        return None

    rows = []
    price_model = price0
    for idx in range(cutoff_idx, cutoff_idx + horizon):
        if idx + 1 >= len(df):
            break
        price_real = float(df["Close"].iloc[idx + 1])
        try:
            gain_pred_pct = apply_fitted(state, feat, idx)
        except Exception:
            continue
        if not np.isfinite(gain_pred_pct):
            continue
        price_model *= (1 + gain_pred_pct / 100.0)
        rows.append({"date": feat.index[idx + 1], "price_model": price_model, "price_real": price_real})

    if not rows:
        return None
    daily = pd.DataFrame(rows)
    mae_model = float((daily["price_model"] - daily["price_real"]).abs().mean())
    mae_naive = float((price0 - daily["price_real"]).abs().mean())  # buy&hold: precio no se mueve
    return {
        "ticker": ticker_name, "model": model_name, "horizon": horizon_label,
        "cutoff_date": cutoff_date, "price0": price0, "daily": daily,
        "mae_model": mae_model, "mae_naive": mae_naive,
    }


def plot_sin_ancla(results: list[dict], ticker: str, model: str) -> Path | None:
    """Una gráfica por (modelo, ticker): real vs modelo compuesto SIN ancla,
    en ventanas cortas (30/60/90d) — la prueba justa."""
    rows = [r for r in results if r["ticker"] == ticker and r["model"] == model]
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: list(HORIZONS).index(r["horizon"]))

    fig, axes = plt.subplots(1, len(rows), figsize=(6 * len(rows), 4.2))
    axes = np.atleast_1d(axes)

    for ax, r in zip(axes, rows):
        daily = r["daily"]
        ax.plot(daily["date"], daily["price_real"], color=COLOR_REAL, lw=2.3, label="Real")
        ax.plot(daily["date"], daily["price_model"], color="#8e44ad", lw=1.8,
                 ls="-.", marker="o", ms=3, label="Modelo (sin ancla, compuesto)")
        ax.axhline(r["price0"], color="#888888", lw=1.2, ls=":", label="Buy&hold (precio del corte)")
        mejora_pct = (r["mae_naive"] - r["mae_model"]) / r["mae_naive"] * 100 if r["mae_naive"] else 0.0
        ax.set_title(
            f"{r['horizon']} · corte {r['cutoff_date'].date()}\n"
            f"MAE modelo {r['mae_model']:.2f} vs buy&hold {r['mae_naive']:.2f} ({mejora_pct:+.1f}%)",
            fontsize=9.5,
        )
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)

    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"{ticker} — intento: {model} (sin ancla, ventana corta — prueba justa)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"sinancla_{model}_{ticker}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_sin_ancla_por_horizonte(results: list[dict], ticker: str, models: list[str]) -> list[Path]:
    """3 gráficas (una por horizonte 30/60/90d): real vs los 4 modelos
    (sin ancla, compuesto), todas en la misma gráfica por horizonte."""
    by_horizon: dict[str, list[dict]] = {}
    for r in results:
        if r["ticker"] == ticker and r["model"] in models:
            by_horizon.setdefault(r["horizon"], []).append(r)

    out_paths = []
    for h_label in HORIZONS:
        rows = by_horizon.get(h_label)
        if not rows:
            continue
        rows_by_model = {r["model"]: r for r in rows}
        any_row = next(iter(rows_by_model.values()))

        fig, ax = plt.subplots(figsize=(8.5, 5))
        ax.plot(any_row["daily"]["date"], any_row["daily"]["price_real"],
                 color=COLOR_REAL, lw=2.4, label="Real", zorder=5)
        ax.axhline(any_row["price0"], color="#888888", lw=1.2, ls=":",
                    label="Buy&hold (precio del corte)", zorder=1)

        for model_name in models:
            r = rows_by_model.get(model_name)
            if r is None:
                continue
            daily = r["daily"]
            ax.plot(daily["date"], daily["price_model"], color=COLOR_FORECAST, lw=1.8,
                     ls=MODEL_STYLES.get(model_name, "-"), alpha=0.9,
                     label=f"{model_name}", zorder=4)

        ax.set_title(f"{ticker} — {h_label} · corte {any_row['cutoff_date'].date()}", fontsize=12)
        ax.tick_params(axis="x", rotation=30, labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=9, frameon=False, loc="best")
        fig.suptitle("Los 4 intentos, sin ancla (solo su propia predicción compuesta)",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUT_DIR / f"sinancla_resumen_{h_label}_{ticker}.png"
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        out_paths.append(out_path)
    return out_paths


# ───────────────────────── Núcleo: forecast + banda ──────────────────────────

def build_forecast_path(price0: float, r_hat_pct: float, horizon: int, vol_daily: float) -> pd.DataFrame:
    """Camino de precio pronosticado + banda de incertidumbre (~90%, deriva
    lineal en log-retorno + banda que escala con sqrt(t) por volatilidad)."""
    t = np.arange(1, horizon + 1)
    mu_total = np.log1p(r_hat_pct / 100.0)
    mu_t = mu_total * (t / horizon)
    sigma_t = vol_daily * np.sqrt(t)
    return pd.DataFrame({
        "t": t,
        "forecast": price0 * np.exp(mu_t),
        "lower": price0 * np.exp(mu_t - Z_SCORE * sigma_t),
        "upper": price0 * np.exp(mu_t + Z_SCORE * sigma_t),
    })


def run_one(ticker_name: str, model_name: str, horizon_label: str, horizon: int) -> dict | None:
    symbol = TICKERS[ticker_name]
    df = store_get(symbol)
    if df is None or len(df) < MIN_TRAIN_ROWS + horizon + 10:
        print(f"  [{model_name}] {ticker_name} {horizon_label}: datos insuficientes")
        return None

    feat = compute_features(df)
    cutoff_idx = len(feat) - horizon - 1
    if cutoff_idx < MIN_TRAIN_ROWS:
        return None

    cutoff_date = feat.index[cutoff_idx]
    price0 = float(df["Close"].iloc[cutoff_idx])
    daily_log_ret = np.log(df["Close"] / df["Close"].shift(1))
    vol_daily = float(daily_log_ret.iloc[max(0, cutoff_idx - VOL_WINDOW):cutoff_idx].std())
    if not np.isfinite(vol_daily) or vol_daily <= 0:
        return None

    try:
        if model_name == "regimen_hmm":
            r_hat = forecast_regimen(df, feat, cutoff_idx, horizon)
        else:
            r_hat = MODELS[model_name](feat, cutoff_idx, horizon)
    except Exception as exc:
        print(f"  [{model_name}] {ticker_name} {horizon_label}: ERROR {exc}")
        return None

    if not np.isfinite(r_hat):
        return None

    path = build_forecast_path(price0, r_hat, horizon, vol_daily)
    real = df["Close"].iloc[cutoff_idx + 1: cutoff_idx + 1 + horizon].reset_index()
    real.columns = ["date", "close"]
    path["date"] = real["date"].values[: len(path)]

    return {
        "ticker": ticker_name, "model": model_name, "horizon": horizon_label,
        "cutoff_date": cutoff_date, "r_hat_pct": r_hat, "vol_daily": vol_daily,
        "path": path, "real": real,
    }


# ───────────────────────────── Gráficas ──────────────────────────────────────

def plot_model_ticker(results: list[dict], ticker: str, model: str) -> Path | None:
    """Una gráfica por (modelo, ticker): real en rojo, pronóstico en azul,
    banda de incertidumbre propia de ese modelo/horizonte."""
    rows = [r for r in results if r["ticker"] == ticker and r["model"] == model]
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: list(HORIZONS).index(r["horizon"]))

    fig, axes = plt.subplots(1, len(rows), figsize=(6 * len(rows), 4.2))
    axes = np.atleast_1d(axes)

    for ax, r in zip(axes, rows):
        path, real = r["path"], r["real"]
        ax.plot(real["date"], real["close"], color=COLOR_REAL, lw=2.3, label="Real")
        ax.plot(path["date"], path["forecast"], color=COLOR_FORECAST, lw=2,
                 ls="--", label="Pronóstico")
        ax.fill_between(path["date"], path["lower"], path["upper"],
                         color=COLOR_BAND, alpha=0.12, label="Incertidumbre ~90%")
        ax.set_title(f"{r['horizon']} · corte {r['cutoff_date'].date()}", fontsize=11)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)

    axes[0].legend(fontsize=9, frameon=False)
    fig.suptitle(f"{ticker} — intento: {model}", fontsize=13, fontweight="bold")
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{model}_{ticker}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_ticker_combined(results: list[dict], ticker: str, models: list[str]) -> Path | None:
    """Una sola gráfica por ticker: real en rojo, los N intentos en azul
    (distintos trazos), banda de incertidumbre compartida (volatilidad
    realizada del ticker — es la misma para todos los modelos, no depende
    de cuál intento se use)."""
    by_horizon: dict[str, list[dict]] = {}
    for r in results:
        if r["ticker"] == ticker and r["model"] in models:
            by_horizon.setdefault(r["horizon"], []).append(r)
    if not by_horizon:
        return None
    horizons = [h for h in HORIZONS if h in by_horizon]

    fig, axes = plt.subplots(1, len(horizons), figsize=(6.5 * len(horizons), 4.6))
    axes = np.atleast_1d(axes)

    for ax, h in zip(axes, horizons):
        rows = {r["model"]: r for r in by_horizon[h]}
        any_row = next(iter(rows.values()))
        real = any_row["real"]
        ax.plot(real["date"], real["close"], color=COLOR_REAL, lw=2.5, label="Real", zorder=5)

        # Banda: la envolvente más ancha entre los modelos disponibles (una
        # sola región, no una por modelo, para no saturar la gráfica).
        lowers = pd.concat([rows[m]["path"]["lower"] for m in rows], axis=1).min(axis=1)
        uppers = pd.concat([rows[m]["path"]["upper"] for m in rows], axis=1).max(axis=1)
        dates = any_row["path"]["date"]
        ax.fill_between(dates, lowers, uppers, color=COLOR_BAND, alpha=0.10,
                         label="Incertidumbre ~90% (volatilidad)", zorder=1)

        for model_name, r in rows.items():
            path = r["path"]
            ax.plot(path["date"], path["forecast"], color=COLOR_FORECAST, lw=1.8,
                     ls=MODEL_STYLES.get(model_name, "-"), alpha=0.9,
                     label=f"Pronóstico — {model_name}", zorder=4)

        ax.set_title(f"{h} · corte {any_row['cutoff_date'].date()}", fontsize=11)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)

    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle(f"{ticker} — real (rojo) vs pronóstico por intento (azul)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"combined_{ticker}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_rolling(results_rolling: list[dict], ticker: str, model: str) -> Path | None:
    """Real (rojo) vs pronóstico rodante día-a-día (azul) — a diferencia del
    modo estático, aquí la línea azul SÍ se mueve: es una serie de
    pronósticos 'mañana', uno por día, no un solo número repartido."""
    rows = [r for r in results_rolling if r["ticker"] == ticker and r["model"] == model]
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: list(HORIZONS).index(r["horizon"]))

    fig, axes = plt.subplots(1, len(rows), figsize=(6 * len(rows), 4.2))
    axes = np.atleast_1d(axes)

    for ax, r in zip(axes, rows):
        daily = r["daily"]
        ax.plot(daily["date"], daily["price_real"], color=COLOR_REAL, lw=2.3, label="Real")
        ax.plot(daily["date"], daily["price_pred"], color=COLOR_FORECAST, lw=1.6,
                 ls="--", marker="o", ms=3, label="Pronóstico rodante (mañana)")
        ax.set_title(
            f"{r['horizon']} · hit {r['hit_rate']*100:.0f}%  "
            f"· gan. prom. pred {r['mean_gain_pred']:+.2f}% vs real {r['mean_gain_real']:+.2f}%",
            fontsize=9.5,
        )
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)

    axes[0].legend(fontsize=9, frameon=False)
    fig.suptitle(f"{ticker} — intento: {model} (rodante, 1 día adelante)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"rolling_{model}_{ticker}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# ───────────────────── Modo mesh (serie completa, rodante) ───────────────────
#
# A diferencia de `run_rolling` (que solo rueda dentro de la ventana de
# validación del horizonte, 21/42/63 días), aquí el ajuste se hace UNA vez
# en un corte fijo (FULL_TRAIN_FRAC de la historia) y el pronóstico "mañana"
# se rueda día a día sobre TODO lo que queda de la serie — para ver, en una
# sola gráfica, la serie completa real y qué tanto el pronóstico rodante la
# sigue a lo largo de meses/años, no solo en los últimos 1-3 meses.

FULL_TRAIN_FRAC = 0.7  # % de la historia usado para ajustar; el resto se rueda


def run_rolling_full(ticker_name: str, model_name: str) -> dict | None:
    symbol = TICKERS[ticker_name]
    df = store_get(symbol)
    if df is None or len(df) < MIN_TRAIN_ROWS + 60:
        return None

    feat = compute_features(df)
    cutoff_idx = max(MIN_TRAIN_ROWS, int(len(feat) * FULL_TRAIN_FRAC))
    if cutoff_idx >= len(feat) - 10:
        return None
    cutoff_date = feat.index[cutoff_idx]

    try:
        if model_name == "regimen_hmm":
            state = fit_regimen(df, feat, cutoff_idx)
        else:
            state = FIT_ROLLING[model_name](feat, cutoff_idx)
    except Exception as exc:
        print(f"  [mesh {model_name}] {ticker_name}: ERROR ajuste {exc}")
        return None
    if state is None:
        return None

    rows = []
    for idx in range(cutoff_idx, len(df) - 1):
        price_today = float(df["Close"].iloc[idx])
        price_tomorrow_real = float(df["Close"].iloc[idx + 1])
        try:
            gain_pred_pct = apply_fitted(state, feat, idx)
        except Exception:
            continue
        if not np.isfinite(gain_pred_pct):
            continue
        price_pred = price_today * (1 + gain_pred_pct / 100.0)
        direction_pred = "sube" if gain_pred_pct >= 0 else "baja"
        gain_real_pct = (price_tomorrow_real / price_today - 1) * 100
        direction_real = "sube" if gain_real_pct >= 0 else "baja"
        # Línea base ingenua: "mañana = precio de hoy, sin modelo" (persistencia).
        # Su dirección siempre es "sube" (gain=0 >= 0) — lo que importa no es
        # su dirección sino qué tan cerca cae price_pred_naive del precio real,
        # comparado contra qué tan cerca cae el pronóstico del modelo.
        price_pred_naive = price_today
        rows.append({
            "date": feat.index[idx + 1], "price_pred": price_pred,
            "price_pred_naive": price_pred_naive,
            "gain_pred_pct": gain_pred_pct, "price_real": price_tomorrow_real,
            "gain_real_pct": gain_real_pct, "hit": direction_pred == direction_real,
        })

    if not rows:
        return None
    daily = pd.DataFrame(rows)
    full_real = df["Close"].reset_index()
    full_real.columns = ["date", "close"]
    mae_model = float((daily["price_pred"] - daily["price_real"]).abs().mean())
    mae_naive = float((daily["price_pred_naive"] - daily["price_real"]).abs().mean())
    return {
        "ticker": ticker_name, "model": model_name, "cutoff_date": cutoff_date,
        "full_real": full_real, "daily": daily,
        "hit_rate": float(daily["hit"].mean()),
        "mae_model": mae_model, "mae_naive": mae_naive,
    }


def plot_mesh(results_full: list[dict], model: str, tickers: list[str]) -> Path | None:
    """Mesh (rejilla) de una gráfica por ticker para `model`: serie completa
    real (rojo) + pronóstico rodante 'mañana' (azul) desde el corte de ajuste
    hasta el final — línea vertical marca dónde empieza a rodar."""
    rows = [r for r in results_full if r["model"] == model and r["ticker"] in tickers]
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: tickers.index(r["ticker"]))

    fig, axes = plt.subplots(len(rows), 1, figsize=(11, 4.2 * len(rows)))
    axes = np.atleast_1d(axes)

    for ax, r in zip(axes, rows):
        full_real, daily = r["full_real"], r["daily"]
        ax.plot(full_real["date"], full_real["close"], color=COLOR_REAL, lw=1.6, label="Real (serie completa)")
        ax.plot(daily["date"], daily["price_pred_naive"], color="#888888", lw=1.3,
                 ls=":", alpha=0.9, label="Ingenuo: mañana = precio de hoy")
        ax.plot(daily["date"], daily["price_pred"], color=COLOR_FORECAST, lw=1.3,
                 ls="--", alpha=0.85, label="Pronóstico rodante del modelo")
        ax.axvline(r["cutoff_date"], color="#555555", lw=1, ls=":")
        ax.text(r["cutoff_date"], ax.get_ylim()[1], " ajuste → rodando", fontsize=8,
                 color="#555555", va="top")
        mejora_pct = (r["mae_naive"] - r["mae_model"]) / r["mae_naive"] * 100 if r["mae_naive"] else 0.0
        ax.set_title(
            f"{r['ticker']} — hit {r['hit_rate']*100:.0f}%  ·  "
            f"MAE modelo {r['mae_model']:.3f} vs ingenuo {r['mae_naive']:.3f} "
            f"({mejora_pct:+.1f}% vs no hacer nada)",
            fontsize=10.5,
        )
        ax.tick_params(axis="x", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=9, frameon=False, loc="upper left")

    fig.suptitle(f"Intento: {model} — serie completa, pronóstico rodante día a día", fontsize=13, fontweight="bold")
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"mesh_{model}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", default="WALMEX,SPY")
    parser.add_argument("--models", default="lightgbm_ranking,momentum_12_1,woe_score,regimen_hmm")
    parser.add_argument("--modes", default="static,rolling,mesh,sin_ancla",
                         help="static = un pronóstico por horizonte; rolling = día a día "
                              "dentro de la ventana de validación; mesh = serie completa, "
                              "una rejilla de gráficas por modelo (todos los tickers); "
                              "sin_ancla = día a día SIN copiar precio real, ventana corta "
                              "(30/60/90d) — la prueba justa")
    args = parser.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",")]
    models = [m.strip() for m in args.models.split(",")]
    modes = [m.strip() for m in args.modes.split(",")]

    if "static" in modes:
        print("=== Modo estático (un pronóstico por horizonte) ===")
        results = []
        for ticker in tickers:
            for model in models:
                for h_label, h_days in HORIZONS.items():
                    print(f"[{model}] {ticker} {h_label} ({h_days}d)...")
                    r = run_one(ticker, model, h_label, h_days)
                    if r:
                        results.append(r)
                        print(f"  r_hat={r['r_hat_pct']:+.2f}%  vol_diaria={r['vol_daily']:.4f}")

        print()
        for ticker in tickers:
            for model in models:
                out = plot_model_ticker(results, ticker, model)
                if out:
                    print(f"Guardado: {out}")

    if "rolling" in modes:
        print("\n=== Modo rodante (día a día) ===")
        results_rolling = []
        summary_rows = []
        for ticker in tickers:
            for model in models:
                for h_label, h_days in HORIZONS.items():
                    print(f"[rolling {model}] {ticker} {h_label} ({h_days}d)...")
                    r = run_rolling(ticker, model, h_label, h_days)
                    if r:
                        results_rolling.append(r)
                        print(f"  hit_rate={r['hit_rate']*100:.1f}%  "
                              f"gan.pred={r['mean_gain_pred']:+.3f}%  "
                              f"gan.real={r['mean_gain_real']:+.3f}%")
                        summary_rows.append({k: v for k, v in r.items() if k != "daily"})

        print()
        for ticker in tickers:
            for model in models:
                out = plot_rolling(results_rolling, ticker, model)
                if out:
                    print(f"Guardado: {out}")

        if summary_rows:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            summary_path = OUT_DIR / "rolling_summary.csv"
            pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
            print(f"Guardado resumen: {summary_path}")

    if "mesh" in modes:
        print("\n=== Modo mesh (serie completa, rodante) ===")
        results_full = []
        for ticker in tickers:
            for model in models:
                print(f"[mesh {model}] {ticker}...")
                r = run_rolling_full(ticker, model)
                if r:
                    results_full.append(r)
                    print(f"  hit_rate={r['hit_rate']*100:.1f}%  corte={r['cutoff_date'].date()}")

        print()
        for model in models:
            out = plot_mesh(results_full, model, tickers)
            if out:
                print(f"Guardado: {out}")

    if "sin_ancla" in modes:
        print("\n=== Modo sin ancla (día a día, ventana corta, prueba justa) ===")
        results_sa = []
        for ticker in tickers:
            for model in models:
                for h_label, h_days in HORIZONS.items():
                    print(f"[sin_ancla {model}] {ticker} {h_label} ({h_days}d)...")
                    r = run_sin_ancla(ticker, model, h_label, h_days)
                    if r:
                        results_sa.append(r)
                        print(f"  MAE modelo={r['mae_model']:.2f}  MAE buy&hold={r['mae_naive']:.2f}")

        print()
        for ticker in tickers:
            for model in models:
                out = plot_sin_ancla(results_sa, ticker, model)
                if out:
                    print(f"Guardado: {out}")


if __name__ == "__main__":
    main()
