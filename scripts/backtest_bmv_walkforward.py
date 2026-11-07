"""Walk-forward de varios meses, cadencia mensual (como ganó en la prueba de 1 mes),
con stop-loss sí ejecutado día a día entre decisiones y costo real de transacción.
Soporta universo "bmv" (35 emisoras MX) o "full" (BMV + USA + ETFs + cripto + FIBRAs).

    python scripts/backtest_bmv_walkforward.py --months 12
    python scripts/backtest_bmv_walkforward.py --months 24 --universe full --index SPY,NAFTRAC
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.backtest_bmv_month import _call_claude  # noqa: E402
from src.agent.bmv_report import _all_symbols, _bars, _bmv_symbols, atr_stop, build_bmv_report  # noqa: E402

COST = 0.0055

INDEX_SYMBOLS = {"NAFTRAC": "NAFTRAC.MX", "SPY": "SPY", "VOO": "VOO", "VTI": "VTI"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=12)
    ap.add_argument("--hold-days", type=int, default=21)
    ap.add_argument("--stop-mult", type=float, default=1.5, help="ancho del stop en x ATR(14)")
    ap.add_argument("--cooldown", action="store_true", help="no reofrecer un nombre el mes tras tocar su stop")
    ap.add_argument("--universe", choices=["bmv", "full"], default="bmv")
    ap.add_argument("--index", default="NAFTRAC", help="nombres de índice separados por coma, ej SPY,NAFTRAC")
    ap.add_argument("--out", default="data/eval/bmv_walkforward.json")
    args = ap.parse_args()

    index_options = tuple((n, INDEX_SYMBOLS[n]) for n in args.index.split(","))
    index_names = {n for n, _ in index_options}

    symbols = _bmv_symbols() if args.universe == "bmv" else _all_symbols()
    frames = {name: _bars(sym) for name, sym in symbols.items()}
    frames = {k: v for k, v in frames.items() if v is not None and len(v) > 300}
    for idx_name, idx_sym in index_options:
        idx_df = _bars(idx_sym)
        if idx_df is not None:
            frames[idx_name] = idx_df
    base = max(frames.values(), key=len)
    all_days = sorted(base.index)

    anchors = all_days[-(args.months * args.hold_days + 1) :: args.hold_days]
    anchors = anchors[-(args.months + 1) :]
    print(
        f"Walk-forward mensual [universo={args.universe}, {len(symbols)} nombres, "
        f"índice={args.index}, stop={args.stop_mult}x ATR14, cooldown={args.cooldown}]: "
        f"{len(anchors) - 1} meses, {anchors[0].date()} -> {anchors[-1].date()}"
    )

    def price_on(name: str, day: pd.Timestamp) -> float | None:
        df = frames.get(name)
        if df is None:
            return None
        hist = df.loc[:day]
        return float(hist["Close"].iloc[-1]) if not hist.empty else None

    def low_on(name: str, day: pd.Timestamp) -> float | None:
        df = frames.get(name)
        if df is None or "Low" not in df.columns or day not in df.index:
            return None
        v = df.loc[day, "Low"]
        return float(v) if pd.notna(v) else None

    equity = 50.0
    months_log = []
    stopped_out_count = 0
    cooldown_names: set[str] = set()

    for m in range(len(anchors) - 1):
        decide_day = anchors[m]
        month_days = [d for d in all_days if decide_day < d <= anchors[m + 1]]
        if not month_days:
            continue

        report = build_bmv_report(
            asof=decide_day,
            exclude=cooldown_names if args.cooldown else None,
            universe=args.universe,
            index_options=index_options,
        )
        try:
            proposal = _call_claude(report)
        except Exception as exc:
            print(f"  {decide_day.date()}  ERROR {exc}")
            continue

        allocations = proposal.get("allocations") or []
        scale = equity / 50.0
        positions: dict[str, dict] = {}
        cash = equity
        turnover = 0.0
        for a in allocations:
            t = str(a.get("ticker", "")).upper()
            usd = float(a.get("usd") or 0) * scale
            px = price_on(t, decide_day)
            if px is None or px <= 0 or usd < 1 or t not in frames:
                continue
            hist = frames[t].loc[:decide_day]
            stop_px = atr_stop(hist, px, mult=args.stop_mult)
            positions[t] = {"shares": usd / px, "entry_usd": usd, "stop": stop_px, "stopped": False}
            cash -= usd
            turnover += usd
        cost_paid = turnover * COST
        cash -= cost_paid

        stopped_this_month = []
        for day in month_days:
            for t, pos in positions.items():
                if pos["stopped"] or t in index_names:
                    continue  # el índice ya está diversificado, no se le aplica stop de corto plazo
                low = low_on(t, day)
                if low is not None and low <= pos["stop"]:
                    proceeds = pos["shares"] * pos["stop"]
                    exit_cost = proceeds * COST
                    cash += proceeds - exit_cost
                    pos["stopped"] = True
                    pos["exit_day"] = str(day.date())
                    stopped_this_month.append(t)
                    stopped_out_count += 1

        final_day = month_days[-1]
        month_equity = cash + sum(
            pos["shares"] * (price_on(t, final_day) or 0)
            for t, pos in positions.items() if not pos["stopped"]
        )
        ret_pct = (month_equity / equity - 1) * 100 if equity else 0.0
        names = ", ".join(
            f"{t}{'(stop)' if pos['stopped'] else ''}" for t, pos in positions.items()
        ) or "cash"
        print(
            f"  {decide_day.date()}  equity {equity:7.2f} -> {month_equity:7.2f}  "
            f"({ret_pct:+5.1f}%)  [{names}]"
        )
        months_log.append({
            "decide_day": str(decide_day.date()),
            "equity_before": round(equity, 2),
            "equity_after": round(month_equity, 2),
            "retorno_pct": round(ret_pct, 2),
            "allocations": {t: round(p["entry_usd"], 2) for t, p in positions.items()},
            "stopped_out": stopped_this_month,
            "motivo": (proposal.get("motivo") or "")[:200],
        })
        equity = month_equity
        cooldown_names = set(stopped_this_month)

    # Benchmark: el primer índice de la lista, buy & hold en la misma ventana.
    bench_name, bench_sym = index_options[0]
    bench_df = _bars(bench_sym)
    bench_ret = None
    if bench_df is not None:
        bench_hist = bench_df.loc[anchors[0] : anchors[-1]]
        if len(bench_hist) > 1:
            bench_ret = (float(bench_hist["Close"].iloc[-1]) / float(bench_hist["Close"].iloc[0]) - 1) * 100

    print(f"\nFINAL {anchors[-1].date()}  equity ${equity:.2f}  (desde $50)")
    print(f"  retorno total: {(equity / 50.0 - 1) * 100:+.1f}%  en {len(months_log)} meses")
    print(f"  veces que se tocó un stop: {stopped_out_count}")
    if bench_ret is not None:
        print(f"  {bench_name} buy&hold mismo periodo: {bench_ret:+.1f}%")

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "universe": args.universe, "index": args.index,
        "start": str(anchors[0].date()), "end": str(anchors[-1].date()),
        "final_equity": round(equity, 2),
        "retorno_total_pct": round((equity / 50.0 - 1) * 100, 2),
        "benchmark": bench_name,
        "benchmark_ret_pct": round(bench_ret, 2) if bench_ret is not None else None,
        "stopped_out_count": stopped_out_count,
        "months": months_log,
    }, ensure_ascii=False, indent=2))
    print(f"guardado en {out_path}")


if __name__ == "__main__":
    main()
