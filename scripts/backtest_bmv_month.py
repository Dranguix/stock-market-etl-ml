"""Corre el pipeline completo (reporte 360 + llamada real a Claude) sobre ~1 mes
de historia real, con control de frecuencia de rebalanceo, costo de transacción
y nota anti-churn.

    python scripts/backtest_bmv_month.py --days 21 --freq daily
    python scripts/backtest_bmv_month.py --days 21 --freq weekly --cost 0.0055
    python scripts/backtest_bmv_month.py --days 21 --freq monthly --cost 0.0055
    python scripts/backtest_bmv_month.py --days 21 --freq daily --cost 0.0055 --anti-churn
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.agent.bmv_report import _bars, _bmv_symbols, build_bmv_report  # noqa: E402

API_URL = "https://api.anthropic.com/v1/messages"


def _call_claude(report: str) -> dict:
    key = os.environ["ANTHROPIC_API_KEY"]
    model = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-5-5")
    resp = httpx.post(
        API_URL,
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 3000,
            "messages": [{"role": "user", "content": report}],
        },
        timeout=90,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"Anthropic {resp.status_code}: {resp.text[:300]}")
    body = resp.json()
    text = "".join(b.get("text", "") for b in body.get("content") or [] if b.get("type") == "text")
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise RuntimeError(f"sin JSON en la respuesta: {text[:200]}")
    return json.loads(text[start : end + 1])


FREQ_STEP = {"daily": 1, "weekly": 5, "monthly": 21}


def simulate(days_n: int, freq: str, cost_rate: float, anti_churn: bool, verbose: bool = True) -> dict:
    symbols = _bmv_symbols()
    frames = {name: _bars(sym) for name, sym in symbols.items()}
    frames = {k: v for k, v in frames.items() if v is not None and len(v) > 260}
    naftrac_df = _bars("NAFTRAC.MX")
    if naftrac_df is not None:
        frames["NAFTRAC"] = naftrac_df

    base = max(frames.values(), key=len)
    all_days = sorted(base.index)
    days = all_days[-(days_n + 1):]
    step = FREQ_STEP[freq]
    anchors = set(range(0, len(days) - 1, step))

    def price_on(name: str, day: pd.Timestamp) -> float | None:
        df = frames.get(name)
        if df is None:
            return None
        hist = df.loc[:day]
        if hist.empty:
            return None
        return float(hist["Close"].iloc[-1])

    cash = 50.0
    shares: dict[str, float] = {}
    held_since: dict[str, int] = {}  # ticker -> índice del día en que se abrió
    log = []
    trade_usd_total = 0.0

    label = f"[{freq}{' +anti-churn' if anti_churn else ''}{f' cost={cost_rate:.4f}' if cost_rate else ''}]"
    if verbose:
        print(f"{label} Simulando {len(days) - 1} sesiones: {days[0].date()} -> {days[-1].date()}")

    for i, day in enumerate(days[:-1]):
        next_day = days[i + 1]
        equity_today = cash + sum((price_on(t, day) or 0) * sh for t, sh in shares.items())

        if i not in anchors:
            # Día sin rebalanceo: solo deja correr el mercado.
            next_equity = cash + sum((price_on(t, next_day) or 0) * sh for t, sh in shares.items())
            log.append({
                "day": str(day.date()), "rebalanceo": False,
                "equity_before": round(equity_today, 2), "equity_next": round(next_equity, 2),
            })
            continue

        held_days = {t: i - held_since.get(t, i) for t in shares}
        report = build_bmv_report(
            asof=day,
            held=held_days if (anti_churn and held_days) else None,
            cost_note=anti_churn,
        )
        try:
            proposal = _call_claude(report)
        except Exception as exc:
            if verbose:
                print(f"  {day.date()}  ERROR {exc}")
            log.append({"day": str(day.date()), "error": str(exc), "equity_before": round(equity_today, 2)})
            continue

        allocations = proposal.get("allocations") or []
        scale = equity_today / 50.0
        targets = {a["ticker"]: max(0.0, float(a.get("usd") or 0) * scale) for a in allocations}

        turnover = 0.0
        for t in list(shares):
            if t not in targets:
                px = price_on(t, day)
                sold = shares[t] * (px or 0)
                turnover += sold
                if px:
                    cash += sold
                shares.pop(t)
                held_since.pop(t, None)

        for t, usd in targets.items():
            px = price_on(t, day)
            if px is None or px <= 0:
                continue
            current_usd = shares.get(t, 0.0) * px
            delta = usd - current_usd
            if abs(delta) < 0.5:
                continue
            turnover += abs(delta)
            cash -= delta
            shares[t] = usd / px
            if t not in held_since or current_usd == 0:
                held_since[t] = i

        cost_paid = turnover * cost_rate
        cash -= cost_paid
        trade_usd_total += turnover

        next_equity = cash + sum((price_on(t, next_day) or 0) * sh for t, sh in shares.items())
        pos_txt = ", ".join(
            f"{t} ${(price_on(t, day) or 0) * sh:.1f}" for t, sh in shares.items()
        ) or "cash"
        if verbose:
            print(
                f"  {day.date()}  equity {equity_today:6.2f} -> {next_equity:6.2f}  "
                f"cash {cash:6.2f}  costo {cost_paid:4.2f}  [{pos_txt}]"
            )
        log.append({
            "day": str(day.date()), "rebalanceo": True,
            "equity_before": round(equity_today, 2), "equity_next": round(next_equity, 2),
            "cash": round(cash, 2), "turnover_usd": round(turnover, 2), "costo_usd": round(cost_paid, 4),
            "allocations": targets, "motivo": (proposal.get("motivo") or "")[:200],
        })

    final_day = days[-1]
    final_equity = cash + sum((price_on(t, final_day) or 0) * sh for t, sh in shares.items())
    result = {
        "label": label, "freq": freq, "cost_rate": cost_rate, "anti_churn": anti_churn,
        "start": str(days[0].date()), "end": str(final_day.date()),
        "final_equity": round(final_equity, 2),
        "retorno_pct": round((final_equity / 50.0 - 1) * 100, 2),
        "turnover_usd_total": round(trade_usd_total, 2),
        "costo_usd_total": round(trade_usd_total * cost_rate, 2),
        "log": log,
    }
    if verbose:
        print(
            f"{label} FINAL {final_day.date()}  equity ${final_equity:.2f}  "
            f"retorno {result['retorno_pct']:+.1f}%  turnover ${trade_usd_total:.0f}  "
            f"costo ${result['costo_usd_total']:.2f}\n"
        )
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=21)
    ap.add_argument("--freq", choices=["daily", "weekly", "monthly"], default="daily")
    ap.add_argument("--cost", type=float, default=0.0, help="costo ida y vuelta, ej 0.0055")
    ap.add_argument("--anti-churn", action="store_true")
    ap.add_argument("--out", default="data/eval/bmv_month_backtest.json")
    args = ap.parse_args()

    result = simulate(args.days, args.freq, args.cost, args.anti_churn)
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"guardado en {out_path}")


if __name__ == "__main__":
    main()
