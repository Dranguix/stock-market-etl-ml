#!/usr/bin/env python3
"""
A/B walk-forward de la frontera: Graph-Ising + Order Flow vs baseline.

Brazos (mismo panel, mismos folds, mismo modelo LGBM):
  A  baseline    — técnicos + macro + cross-section (núcleo actual)
  B  +gim        — A + GIM_FEATURE_COLS (Graph-Ising sobre grafo de correlación)
  C  +flow       — A + FLOW_FEATURE_COLS (order flow de brokers, hechos BMV)
  D  +gim+flow   — todo

Protocolo:
  - Panel BMV point-in-time (36 tickers, features causales).
  - Walk-forward expanding: refit cada `refit_days` hábiles, embargo = horizon+1.
  - Métricas: AUC (y_rel) con CI bootstrap por bloques de fecha, hit rate top-2,
    Sharpe neto long-only top-2 (costos COMM por lado).
  - Ventana FLOW: sub-análisis solo en fechas con datos de hechos, donde C/D
    pueden diferenciarse de verdad.

Regla de promoción (plan): ΔAUC > +0.015 y Sharpe neto > baseline.

Ejecución:
    python scripts/eval_frontier_ab.py --horizon 20
    python scripts/eval_frontier_ab.py --horizon 5 --test-years 3
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from sklearn.metrics import roc_auc_score

from src.data.store import get as store_get, BMV_TICKERS, MACRO_TICKERS
from src.ml.features import (
    FEATURE_COLS,
    LONG_MOMENTUM_COLS,
    add_cross_section_features,
    add_macro_features,
    compute_features,
    compute_target,
)
from src.ml.flow_features import FLOW_FEATURE_COLS, add_flow_features, compute_flow_panel
from src.ml.graph_ising import GIM_FEATURE_COLS, add_gim_features, compute_gim_panel

COMM = 0.0025      # costo por lado
TOP_N = 2          # posiciones long (igual que MAX_SIGNALS en bet_policy)
MIN_NAMES = 10

BASE_COLS = FEATURE_COLS + LONG_MOMENTUM_COLS

_LGBM_PARAMS = dict(
    n_estimators=400, num_leaves=31, learning_rate=0.03, min_child_samples=30,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0,
    n_jobs=4, random_state=42, verbosity=-1,
)


def build_panel(horizon: int) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """Panel BMV con baseline + flow + gim (todo point-in-time)."""
    macro = {n: store_get(n) for n in MACRO_TICKERS}
    macro = {k: v for k, v in macro.items() if v is not None}

    frames: dict[str, pd.DataFrame] = {}
    for ticker in BMV_TICKERS:
        raw = store_get(ticker)
        if raw is None or len(raw) < 300:
            continue
        feat = compute_features(raw)
        feat = add_macro_features(feat, macro)
        feat["fwd_ret"] = compute_target(raw, horizon=horizon, use_open_entry=True)
        feat["ticker"] = ticker
        frames[ticker] = feat
    frames = add_cross_section_features(frames)

    # Order flow (solo fechas con hechos descargados)
    closes = pd.DataFrame({t: df["Close"] for t, df in frames.items()})
    closes.index = pd.to_datetime(closes.index)
    flow_panel = compute_flow_panel(list(frames.keys()), close_panel=closes)
    flow_start = (
        pd.to_datetime(flow_panel["date"]).min() if not flow_panel.empty else None
    )
    frames = {t: add_flow_features(df, t, flow_panel) for t, df in frames.items()}

    # Graph-Ising (usa flow como evidencia si existe)
    gim_panel = compute_gim_panel(frames)
    frames = {t: add_gim_features(df, t, gim_panel) for t, df in frames.items()}

    df = pd.concat(frames.values()).sort_index()
    df = df.dropna(subset=BASE_COLS + ["fwd_ret"])

    df["uni_n"] = df.groupby(df.index)["fwd_ret"].transform("count")
    df["uni_med"] = df.groupby(df.index)["fwd_ret"].transform("median")
    df = df[df["uni_n"] >= MIN_NAMES]
    df["y_rel"] = (df["fwd_ret"] > df["uni_med"]).astype(int)
    return df, flow_start


def walk_forward_predict(
    panel: pd.DataFrame,
    cols: list[str],
    horizon: int,
    test_start: pd.Timestamp,
    refit_days: int = 63,
) -> pd.DataFrame:
    """Predicciones OOS con refits expanding-window y embargo anti-lookahead."""
    import lightgbm as lgb

    dates = panel.index.unique().sort_values()
    test_dates = dates[dates >= test_start]
    if len(test_dates) == 0:
        return pd.DataFrame()

    out = []
    for i in range(0, len(test_dates), refit_days):
        fold_dates = test_dates[i : i + refit_days]
        fold_start = fold_dates[0]
        embargo = fold_start - pd.tseries.offsets.BDay(horizon + 1)

        train = panel[panel.index <= embargo]
        test = panel.loc[panel.index.isin(fold_dates)]
        if len(train) < 1000 or test.empty or train["y_rel"].nunique() < 2:
            continue

        clf = lgb.LGBMClassifier(**_LGBM_PARAMS)
        clf.fit(train[cols].values, train["y_rel"].values)
        p = clf.predict_proba(test[cols].fillna(0.0).values)[:, 1]

        fold_out = test[["ticker", "fwd_ret", "y_rel"]].copy()
        fold_out["p"] = p
        out.append(fold_out)

    return pd.concat(out) if out else pd.DataFrame()


def block_bootstrap_auc_ci(
    preds: pd.DataFrame, n_boot: int = 500, seed: int = 42,
) -> tuple[float, float]:
    """CI 90% del AUC re-muestreando bloques por fecha (respeta clustering)."""
    rng = np.random.default_rng(seed)
    dates = preds.index.unique().to_numpy()
    groups = {d: g for d, g in preds.groupby(preds.index)}
    aucs = []
    for _ in range(n_boot):
        sample_dates = rng.choice(dates, size=len(dates), replace=True)
        sample = pd.concat([groups[d] for d in sample_dates])
        if sample["y_rel"].nunique() < 2:
            continue
        aucs.append(roc_auc_score(sample["y_rel"], sample["p"]))
    if not aucs:
        return (0.5, 0.5)
    return (float(np.percentile(aucs, 5)), float(np.percentile(aucs, 95)))


def topk_portfolio(preds: pd.DataFrame, horizon: int) -> dict:
    """Long-only top-N semanal-ish (cada `horizon` días), neto de costos."""
    dates = preds.index.unique().sort_values()
    step = max(1, horizon)
    rets, hits = [], []
    i = 0
    while i < len(dates):
        day = dates[i]
        grp = preds.loc[preds.index == day]
        if len(grp) >= MIN_NAMES:
            top = grp.sort_values("p", ascending=False).head(TOP_N)
            r = top["fwd_ret"].mean() / 100.0 - COMM * 2
            rets.append(r)
            hits.append(float((top["y_rel"] == 1).mean()))
        i += step

    if not rets:
        return {"sharpe_net": 0.0, "hit_top2": 0.0, "n_periods": 0, "avg_ret_pct": 0.0}
    pr = np.array(rets)
    ppy = 252 / horizon
    sharpe = float(pr.mean() / (pr.std() + 1e-9) * np.sqrt(ppy))
    return {
        "sharpe_net": round(sharpe, 3),
        "hit_top2": round(float(np.mean(hits)), 4),
        "n_periods": len(rets),
        "avg_ret_pct": round(float(pr.mean() * 100), 4),
    }


def eval_arm(
    panel: pd.DataFrame,
    cols: list[str],
    horizon: int,
    test_start: pd.Timestamp,
    flow_start: pd.Timestamp | None,
) -> dict:
    preds = walk_forward_predict(panel, cols, horizon, test_start)
    if preds.empty or preds["y_rel"].nunique() < 2:
        return {"auc": 0.5, "error": "sin predicciones"}

    auc = float(roc_auc_score(preds["y_rel"], preds["p"]))
    ci = block_bootstrap_auc_ci(preds)
    port = topk_portfolio(preds, horizon)

    result = {
        "auc": round(auc, 4),
        "auc_ci90": [round(ci[0], 4), round(ci[1], 4)],
        "n_preds": len(preds),
        **port,
    }

    # Sub-ventana con order flow real disponible
    if flow_start is not None:
        sub = preds[preds.index >= flow_start]
        if len(sub) > 200 and sub["y_rel"].nunique() > 1:
            result["flow_window"] = {
                "auc": round(float(roc_auc_score(sub["y_rel"], sub["p"])), 4),
                **topk_portfolio(sub, horizon),
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B frontera Graph-Ising + Order Flow")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--test-years", type=float, default=3.0)
    parser.add_argument("--out", default="data/eval/frontier_ab.json")
    args = parser.parse_args()

    print(f"\n══ A/B Frontera — horizonte {args.horizon}d ══\n")
    print("  Construyendo panel (features + flow + graph-ising)…")
    panel, flow_start = build_panel(args.horizon)
    print(f"  Panel: {len(panel):,} filas, {panel['ticker'].nunique()} tickers, "
          f"{panel.index.min().date()} → {panel.index.max().date()}")
    if flow_start is not None:
        print(f"  Order flow desde: {flow_start.date()}")

    test_start = panel.index.max() - pd.DateOffset(years=args.test_years)

    arms = {
        "A_baseline": BASE_COLS,
        "B_gim": BASE_COLS + GIM_FEATURE_COLS,
        "C_flow": BASE_COLS + FLOW_FEATURE_COLS,
        "D_gim_flow": BASE_COLS + GIM_FEATURE_COLS + FLOW_FEATURE_COLS,
    }

    results: dict[str, dict] = {}
    for name, cols in arms.items():
        print(f"\n  ── {name} ({len(cols)} features) ──")
        r = eval_arm(panel, cols, args.horizon, test_start, flow_start)
        results[name] = r
        fw = r.get("flow_window")
        print(f"     AUC={r['auc']:.4f} CI90={r.get('auc_ci90')}  "
              f"Sharpe_net={r.get('sharpe_net')}  hit_top2={r.get('hit_top2')}")
        if fw:
            print(f"     [ventana flow] AUC={fw['auc']:.4f}  "
                  f"Sharpe_net={fw['sharpe_net']}  hit_top2={fw['hit_top2']}")

    base_auc = results["A_baseline"].get("auc", 0.5)
    base_sharpe = results["A_baseline"].get("sharpe_net", 0.0)
    verdicts = {}
    for name in ("B_gim", "C_flow", "D_gim_flow"):
        r = results.get(name, {})
        d_auc = round(r.get("auc", 0.5) - base_auc, 4)
        d_sharpe = round(r.get("sharpe_net", 0.0) - base_sharpe, 3)
        promote = d_auc > 0.015 and d_sharpe > 0
        verdicts[name] = {"delta_auc": d_auc, "delta_sharpe": d_sharpe, "promote": promote}

    output = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "horizon": args.horizon,
        "test_start": str(test_start.date()),
        "flow_data_start": str(flow_start.date()) if flow_start is not None else None,
        "promotion_rule": "delta_auc > 0.015 y delta_sharpe > 0 vs A_baseline",
        "arms": results,
        "verdicts": verdicts,
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2))

    print("\n  ── Veredictos vs baseline ──")
    for name, v in verdicts.items():
        flag = "PROMOVER" if v["promote"] else "no promover"
        print(f"     {name}: ΔAUC={v['delta_auc']:+.4f}  ΔSharpe={v['delta_sharpe']:+.3f} → {flag}")
    print(f"\n  Guardado: {out}\n")


if __name__ == "__main__":
    main()
