#!/usr/bin/env python3
"""
Backfill de tasas históricas de mercado de dinero mexicano.

Usa la API pública de Banxico (SIE) para rellenar el historial antes de
que tengamos snapshots de DataBursátil, y luego guarda el snapshot de hoy.

Token Banxico: gratis en https://www.banxico.org.mx/SieAPIRest/
  (también funciona sin token con rate-limit más bajo)

Uso:
    python scripts/backfill_rates.py                     # hoy + Banxico 2024-01-01
    python scripts/backfill_rates.py --start 2023-01-01  # desde otra fecha
    python scripts/backfill_rates.py --bmx-token TU_TOK  # con token Banxico
    python scripts/backfill_rates.py --today-only        # solo snapshot de hoy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from src.ml.rates_features import (
    RATES_DIR,
    backfill_from_banxico,
    build_rates_panel,
    compute_rates_features,
    fetch_tasas_snapshot,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill tasas MX (Banxico + DataBursátil)")
    parser.add_argument("--start", default="2024-01-01", help="Fecha inicio Banxico backfill")
    parser.add_argument("--bmx-token", default=None, help="Token API Banxico (opcional)")
    parser.add_argument("--today-only", action="store_true", help="Solo guardar snapshot de hoy")
    parser.add_argument("--no-banxico", action="store_true", help="Saltar Banxico, solo hoy")
    args = parser.parse_args()

    print(f"\n══ Backfill Tasas MX ══\n  Directorio: {RATES_DIR}\n")

    # ── 1. Snapshot de hoy vía DataBursátil ──────────────────────────────────
    print("  [1/3] Snapshot de hoy (DataBursátil /v2/tasas)…")
    snap = fetch_tasas_snapshot(sleep=0)
    if snap:
        print(f"  OK: {len(snap)} tasas guardadas")
        for k, v in snap.items():
            print(f"       {k:<15}: {v:.4f}%")
    else:
        print("  WARN: no se pudo guardar snapshot de hoy")

    if args.today_only or args.no_banxico:
        _print_summary()
        return

    # ── 2. Bootstrap histórico vía Banxico ────────────────────────────────────
    print(f"\n  [2/3] Backfill Banxico desde {args.start}…")
    n = backfill_from_banxico(
        start=args.start,
        bmx_token=args.bmx_token,
        sleep=0.5,
    )
    if n > 0:
        print(f"  OK: {n} días guardados desde Banxico")
    else:
        print("  WARN: Banxico sin datos (¿token requerido?)")
        print("        Obtén tu token gratis en: https://www.banxico.org.mx/SieAPIRest/")

    # ── 3. Resumen del panel ──────────────────────────────────────────────────
    _print_summary()


def _print_summary() -> None:
    print("\n  [3/3] Resumen del panel…")
    panel = build_rates_panel()
    if panel.empty:
        print("  Panel vacío — solo hay snapshot de hoy (features nivel ya disponibles)")
        return

    feats = compute_rates_features(panel)
    print(f"\n  Panel: {len(panel)} días ({panel.index[0].date()} → {panel.index[-1].date()})")
    print("\n  Últimas features:")
    print(feats.tail(5).to_string())
    print()


if __name__ == "__main__":
    main()
