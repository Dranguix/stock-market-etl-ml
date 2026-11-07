"""Backtest del libro de 50 USD con ejecución honesta.

Señal al cierre de la última sesión del mes, ejecución en la apertura siguiente.
Precios MX convertidos a USD con MXN=X (la divisa entra al resultado), BMV en
unidades enteras, mínimo ~100 MXN por orden, costo 0.6% ida y vuelta (0.8% en
estrés) más el sobrecargo de conversión en lo que se compra en pesos.

Políticas (mismo motor, misma ejecución):
  mecanico   la regla de src/agent/core_satellite.py. Es el gemelo sin LLM.
  veto       la misma regla, con Haiku vetando nombres del satélite por noticias.
             Su aporte es veto - mecanico. Sin titulares point-in-time el veto no
             hace nada, así que en historia vieja los dos salen iguales.
  llm        Haiku reparte los 50 como antes (reporte 360 de bmv_report), sin
             stops intramensuales. Solo para comparar contra el gemelo.

Cada corrida se anota en data/eval/trial_log.jsonl y el Deflated Sharpe usa
ese N más --prior-trials (las variantes que ya se probaron antes de la bitácora).

    python scripts/backtest_core_satellite.py --months 24
    python scripts/backtest_core_satellite.py --months 24 --stress
    python scripts/backtest_core_satellite.py --months 24 --policy veto --replicas 5
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import core_satellite as cs  # noqa: E402
from src.ml.trial_stats import deflated_sharpe, expected_max_sharpe, log_trial, n_trials  # noqa: E402

BUDGET = 50.0


class Market:
    """Opens y closes alineados al calendario de SPY, en moneda local y en USD."""

    def __init__(self, frames: dict[str, pd.DataFrame], days: pd.DatetimeIndex):
        self.days = days
        fx = frames[cs.FX_SYMBOL]
        self.fx_close = fx["Close"].reindex(days).ffill()
        self.fx_open = fx["Open"].reindex(days).fillna(self.fx_close)
        self.close: dict[str, pd.Series] = {}
        self.open: dict[str, pd.Series] = {}
        self.mx: dict[str, bool] = {}
        for name, df in frames.items():
            if name == cs.FX_SYMBOL:
                continue
            close = df["Close"].reindex(df.index.union(days)).ffill().reindex(days)
            opened = df["Open"].reindex(days)
            traded = df.index.intersection(days)
            # Si la emisora no operó ese día (asueto BMV), se llena al último cierre.
            opened = opened.where(opened.index.isin(traded)).fillna(close)
            self.close[name], self.open[name] = close, opened
            self.mx[name] = cs.is_mx(cs.symbol_of(name))

    def usd(self, name: str, day: pd.Timestamp, at: str = "close") -> float | None:
        series = (self.close if at == "close" else self.open).get(name)
        if series is None:
            return None
        px = series.get(day)
        if px is None or not np.isfinite(px) or px <= 0:
            return None
        if not self.mx[name]:
            return float(px)
        fx = (self.fx_close if at == "close" else self.fx_open).get(day)
        return float(px / fx)

    def local(self, name: str, day: pd.Timestamp, at: str = "open") -> float:
        return float((self.close if at == "close" else self.open)[name].get(day))


def simulate(policy: str, months: int, cost_rt: float, fx_cost_rt: float, veto_kwargs: dict, verbose: bool = True) -> dict:
    frames = cs.load_frames()
    inputs = cs.Inputs.load()
    calendar = frames["SPY"].index
    end = calendar[-1]
    days = calendar[calendar >= end - pd.DateOffset(months=months)]
    mkt = Market(frames, days)

    veto_fn = None
    if policy == "veto":
        from src.agent.veto import llm_veto

        veto_fn = lambda cands, asof: llm_veto(cands, asof, **veto_kwargs)  # noqa: E731

    cash = BUDGET
    shares: dict[str, float] = {}
    peak: dict[str, float] = {}
    held_sat: list[str] = []
    pending: dict | None = None
    decisions, daily = [], []
    costs_total = turnover_total = 0.0
    n_trades = 0
    pnl = {"nucleo_local": 0.0, "nucleo_fx": 0.0, "satelite_local": 0.0, "satelite_fx": 0.0}
    prev_close_usd: dict[str, float] = {}

    def equity(day, at="close") -> float:
        return cash + sum(sh * (mkt.usd(t, day, at) or 0.0) for t, sh in shares.items())

    def sleeve(t: str) -> str:
        return "nucleo" if t in cs.CORE else "satelite"

    def execute(day, order: dict) -> None:
        nonlocal cash, costs_total, turnover_total, n_trades
        eq = equity(day, "open")
        fx = float(mkt.fx_open.get(day))
        min_usd = cs.min_order_usd(fx)
        if order["kind"] == "sell":
            target = {t: 0.0 for t in order["names"]}
            target = {**{t: shares[t] for t in shares if t not in target}, **target}
            target_sh = target
        else:
            current_w = {t: sh * (mkt.usd(t, day, "open") or 0.0) / eq for t, sh in shares.items()}
            weights = cs.apply_drift_band(order["weights"], current_w) if order.get("band") else order["weights"]
            target_sh = {}
            for t in set(weights) | set(shares):
                w = weights.get(t, 0.0)
                px_usd = mkt.usd(t, day, "open")
                if px_usd is None:
                    target_sh[t] = shares.get(t, 0.0)
                    continue
                usd = w * eq
                if mkt.mx[t]:
                    target_sh[t] = math.floor(usd * fx / mkt.local(t, day, "open"))
                else:
                    target_sh[t] = usd / px_usd
        # Ventas primero, luego compras con el efectivo que haya.
        for side in ("sell", "buy"):
            for t, tgt in target_sh.items():
                cur = shares.get(t, 0.0)
                delta = tgt - cur
                if (side == "sell" and delta >= 0) or (side == "buy" and delta <= 0):
                    continue
                px_usd = mkt.usd(t, day, "open")
                if px_usd is None:
                    continue
                traded = abs(delta) * px_usd
                closing = side == "sell" and tgt == 0
                if traded < min_usd and not closing:
                    continue
                rate = cost_rt / 2 + (fx_cost_rt / 2 if mkt.mx[t] else 0.0)
                if side == "buy":
                    affordable = cash / (px_usd * (1 + rate))
                    if mkt.mx[t]:
                        affordable = math.floor(affordable)
                    delta = min(delta, affordable)
                    traded = delta * px_usd
                    if delta <= 0 or traded < min_usd:
                        continue
                    cash -= traded * (1 + rate)
                    if cur == 0:
                        peak[t] = px_usd
                else:
                    cash += traded * (1 - rate)
                shares[t] = cur + delta
                if shares[t] <= 1e-9:
                    shares.pop(t)
                    peak.pop(t, None)
                costs_total += traded * rate
                turnover_total += traded
                n_trades += 1

    def attribute(t: str, sh: float, p0: float, p1: float, fx0: float, fx1: float) -> None:
        total = sh * (p1 - p0)
        fx_part = sh * p0 * fx0 * (1 / fx1 - 1 / fx0) if mkt.mx[t] else 0.0
        pnl[f"{sleeve(t)}_fx"] += fx_part
        pnl[f"{sleeve(t)}_local"] += total - fx_part

    last_fx = float(mkt.fx_close.iloc[0])
    for i, day in enumerate(days):
        fx_o, fx_c = float(mkt.fx_open.get(day)), float(mkt.fx_close.get(day))
        mark_fx = last_fx
        if pending is not None:
            # Hueco del cierre anterior a la apertura, luego se ejecuta a la apertura.
            for t, sh in shares.items():
                p1 = mkt.usd(t, day, "open")
                if p1 is not None and t in prev_close_usd:
                    attribute(t, sh, prev_close_usd[t], p1, last_fx, fx_o)
                    prev_close_usd[t] = p1
            execute(day, pending)
            pending = None
            for t in shares:
                if t not in prev_close_usd:
                    prev_close_usd[t] = mkt.usd(t, day, "open")
            mark_fx = fx_o
        for t, sh in shares.items():
            p1 = mkt.usd(t, day, "close")
            if p1 is not None and t in prev_close_usd:
                attribute(t, sh, prev_close_usd[t], p1, mark_fx, fx_c)
                prev_close_usd[t] = p1
        for t in list(prev_close_usd):
            if t not in shares:
                prev_close_usd.pop(t)
        last_fx = fx_c

        eq = equity(day)
        mx_w = sum(sh * (mkt.usd(t, day) or 0) for t, sh in shares.items() if mkt.mx[t]) / eq if eq else 0.0
        daily.append({"day": day, "equity": eq, "mx_weight": mx_w, "fx": fx_c})

        # Stop de catástrofe: al cierre, se vende en la apertura siguiente.
        hits = []
        for t in list(held_sat):
            px_c = mkt.usd(t, day)
            if t not in shares or px_c is None:
                continue
            peak[t] = max(peak.get(t, px_c), px_c)
            if px_c <= peak[t] * (1 - cs.CATASTROPHE_DD):
                hits.append(t)
        if hits:
            held_sat = [t for t in held_sat if t not in hits]
            pending = {"kind": "sell", "names": hits}

        if i == len(days) - 1:
            break
        next_day = days[i + 1]
        if i == 0 or cs.is_rebalance_day(day.date(), next_day.date()):
            if policy == "llm":
                weights, motivo = _llm_weights(day)
                decisions.append({"asof": str(day.date()), "weights": weights, "motivo": motivo})
                pending = {"kind": "target", "weights": weights, "band": False}
            else:
                d = cs.decide(
                    frames, day, inputs, equity_usd=eq, held_satellite=held_sat,
                    reselect=True if i == 0 else None, veto=veto_fn,
                )
                held_sat = list(d.satellite)
                decisions.append({k: v for k, v in d.as_dict().items() if k != "trend"})
                pending = {"kind": "target", "weights": d.weights, "band": i > 0}
            if verbose:
                print(f"  {day.date()}  equity {eq:6.2f}  {decisions[-1]['motivo'][:110]}")

    curve = pd.DataFrame(daily).set_index("day")
    return _summary(policy, curve, mkt, decisions, costs_total, turnover_total, n_trades, pnl)


def _llm_weights(day: pd.Timestamp) -> tuple[dict[str, float], str]:
    from scripts.backtest_bmv_month import _call_claude
    from src.agent.bmv_report import build_bmv_report

    report = build_bmv_report(asof=day, index_options=(("SPY", "SPY"), ("NAFTRAC", "NAFTRAC.MX")))
    proposal = _call_claude(report)
    weights = {}
    for a in proposal.get("allocations") or []:
        t = str(a.get("ticker", "")).upper()
        usd = float(a.get("usd") or 0)
        if usd > 0:
            weights[t] = usd / BUDGET
    total = sum(weights.values())
    if total > 1:
        weights = {t: w / total for t, w in weights.items()}
    return weights, (proposal.get("motivo") or "")[:200]


def _stats(r: pd.Series) -> dict:
    r = r.dropna()
    eq = (1 + r).cumprod()
    years = len(r) / 252
    return {
        "retorno_pct": round(float(eq.iloc[-1] - 1) * 100, 2),
        "cagr_pct": round(float(eq.iloc[-1] ** (1 / years) - 1) * 100, 2) if years > 0 else None,
        "vol_anual_pct": round(float(r.std() * np.sqrt(252)) * 100, 2),
        "sharpe": round(float(r.mean() / r.std() * np.sqrt(252)), 3) if r.std() > 0 else None,
        "max_dd_pct": round(float((eq / eq.cummax() - 1).min()) * 100, 2),
    }


def _summary(policy, curve, mkt, decisions, costs, turnover, n_trades, pnl) -> dict:
    days = curve.index
    r_book = curve["equity"].pct_change()
    r_spy = mkt.close["SPY"].pct_change()
    naf_usd = mkt.close["NAFTRAC"] / mkt.fx_close
    r_naf = naf_usd.pct_change()
    w = float(curve["mx_weight"].mean())
    # Benchmark mezclado: w en NAFTRAC + (1-w) en SPY, con el peso México promedio del libro.
    r_blend = w * r_naf + (1 - w) * r_spy
    fx_ret = curve["fx"].pct_change()
    to_mxn = lambda r: (1 + r) * (1 + fx_ret) - 1  # noqa: E731
    active = (r_book - r_blend).dropna()
    ir = float(active.mean() / active.std() * np.sqrt(252)) if active.std() > 0 else None
    return {
        "policy": policy,
        "start": str(days[0].date()), "end": str(days[-1].date()),
        "final_equity_usd": round(float(curve["equity"].iloc[-1]), 2),
        "peso_mexico_promedio": round(w, 3),
        "usd": {
            "libro": _stats(r_book), "SPY": _stats(r_spy), "NAFTRAC": _stats(r_naf),
            f"mezcla_{w:.2f}NAFTRAC": _stats(r_blend),
        },
        "mxn": {
            "libro": _stats(to_mxn(r_book)), "SPY": _stats(to_mxn(r_spy)),
            "NAFTRAC": _stats(mkt.close["NAFTRAC"].pct_change()),
            f"mezcla_{w:.2f}NAFTRAC": _stats(to_mxn(r_blend)),
        },
        "vs_mezcla": {"active_pct_anual": round(float(active.mean() * 252) * 100, 2), "ir": None if ir is None else round(ir, 3)},
        "atribucion_usd": {k: round(v, 2) for k, v in pnl.items()} | {"costos": round(-costs, 2)},
        "costos_usd": round(costs, 2), "turnover_usd": round(turnover, 2), "operaciones": n_trades,
        "daily_returns": r_book.dropna().round(6).tolist(),
        "decisions": decisions,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["mecanico", "veto", "llm"], default="mecanico")
    ap.add_argument("--months", type=int, default=24)
    ap.add_argument("--cost-rt", type=float, default=0.006, help="comisión+IVA+spread ida y vuelta")
    ap.add_argument("--fx-cost-rt", type=float, default=0.004, help="sobrecargo de conversión USD/MXN ida y vuelta")
    ap.add_argument("--stress", action="store_true", help="0.8%% ida y vuelta")
    ap.add_argument("--replicas", type=int, default=5, help="réplicas del veto por fecha")
    ap.add_argument("--prior-trials", type=int, default=15, help="variantes probadas antes de la bitácora")
    ap.add_argument("--no-log", action="store_true", help="no anotar en la bitácora (solo para tests)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cost_rt = 0.008 if args.stress else args.cost_rt

    print(f"[{args.policy}] {args.months} meses, costo {cost_rt*100:.1f}% ida y vuelta + FX {args.fx_cost_rt*100:.1f}%")
    res = simulate(args.policy, args.months, cost_rt, args.fx_cost_rt, {"replicas": args.replicas})

    # Los costos no son otra estrategia: no entran al hash ni suben N.
    cfg = {
        "policy": args.policy, "months": args.months,
        "regla": cs.config() if args.policy != "llm" else None,
        "replicas": args.replicas if args.policy == "veto" else None,
    }
    family = f"libro50_{args.months}m"
    if args.no_log:
        n = n_trials(family) + args.prior_trials
    else:
        n = log_trial(
            family, cfg,
            {"cost_rt": cost_rt, "fx_cost_rt": args.fx_cost_rt, **{k: res[k] for k in ("usd", "vs_mezcla", "final_equity_usd")}},
        ) + args.prior_trials
    years = len(res["daily_returns"]) / 252
    res["dsr"] = deflated_sharpe(np.array(res["daily_returns"]), max(n, 1))
    res["n_trials"] = n
    res["sharpe_azar_esperado"] = round(expected_max_sharpe(n, years), 3)

    u, m = res["usd"], res["mxn"]
    print(f"\n{res['start']} -> {res['end']}  equity ${res['final_equity_usd']:.2f} (desde $50)")
    for label, block in (("USD", u), ("MXN", m)):
        print(f"  {label}: " + "  ".join(f"{k} {v['retorno_pct']:+.1f}% (SR {v['sharpe']}, DD {v['max_dd_pct']:.1f}%)" for k, v in block.items()))
    print(f"  vs mezcla: activo {res['vs_mezcla']['active_pct_anual']:+.2f}%/año, IR {res['vs_mezcla']['ir']}")
    print(f"  atribución USD: {res['atribucion_usd']}")
    print(f"  operaciones {res['operaciones']}, turnover ${res['turnover_usd']:.0f}, costos ${res['costos_usd']:.2f}")
    print(
        f"  N={n} pruebas sobre esta historia -> el azar solo da Sharpe ~{res['sharpe_azar_esperado']:.2f}; "
        f"DSR {res['dsr']['dsr']} (>0.95 sería evidencia)"
    )

    out = ROOT / (args.out or f"data/eval/core_satellite_{args.policy}_{args.months}m{'_stress' if args.stress else ''}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str))
    print(f"guardado en {out}")


if __name__ == "__main__":
    main()
