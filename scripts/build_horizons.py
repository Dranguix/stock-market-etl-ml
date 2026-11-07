#!/usr/bin/env python3
"""Arma los tres horizontes con el último cierre y los guarda en data/signals/horizons.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.horizons import libro

OUT = ROOT / "data" / "signals" / "horizons.json"


def main() -> None:
    px = pd.read_parquet(ROOT / "data" / "market" / "AAPL.parquet", columns=["Close"])
    asof = pd.to_datetime(px.index).tz_localize(None).max()
    payload = {
        "todo": libro(asof, "todo"),
        "bmv": libro(asof, "bmv"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    for sleeve, block in payload.items():
        print(f"\n{sleeve}  cierre {block['asof']}")
        for kind, picks in block["libro"].items():
            names = ", ".join(f"{p['ticker']} ${p['usd']:.0f}" for p in picks) or "—"
            print(f"  {kind:8} {names}")


if __name__ == "__main__":
    main()
