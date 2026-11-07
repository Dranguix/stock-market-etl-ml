"""Corre las 3 variantes pedidas sobre el mismo mes y las compara:
  1) rebalanceo semanal (menos turnover por diseño) + costo real
  2) rebalanceo mensual + costo real
  3) rebalanceo diario + nota anti-churn (penaliza rotar sin evidencia nueva) + costo real
y de paso recalcula el diario original con costo real para comparar parejo.

    python scripts/run_backtest_suite.py --days 21
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.backtest_bmv_month import simulate  # noqa: E402

COST = 0.0055


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=21)
    args = ap.parse_args()

    variants = [
        ("daily_cost", dict(freq="daily", cost_rate=COST, anti_churn=False)),
        ("daily_cost_antichurn", dict(freq="daily", cost_rate=COST, anti_churn=True)),
        ("weekly_cost", dict(freq="weekly", cost_rate=COST, anti_churn=False)),
        ("monthly_cost", dict(freq="monthly", cost_rate=COST, anti_churn=False)),
    ]

    results = {}
    for name, kwargs in variants:
        print(f"\n=== {name} ===")
        result = simulate(args.days, **kwargs)
        results[name] = result
        out_path = ROOT / "data" / "eval" / f"bmv_month_{name}.json"
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    print("\n\n=== RESUMEN ===")
    print(f"{'variante':24s} {'equity':>8s} {'retorno':>9s} {'turnover':>10s} {'costo':>8s}")
    for name, r in results.items():
        print(
            f"{name:24s} ${r['final_equity']:7.2f} {r['retorno_pct']:+8.1f}% "
            f"${r['turnover_usd_total']:9.2f} ${r['costo_usd_total']:7.2f}"
        )

    summary_path = ROOT / "data" / "eval" / "bmv_month_suite_summary.json"
    summary_path.write_text(json.dumps(
        {k: {kk: vv for kk, vv in v.items() if kk != "log"} for k, v in results.items()},
        ensure_ascii=False, indent=2,
    ))
    print(f"\nguardado resumen en {summary_path}")


if __name__ == "__main__":
    main()
