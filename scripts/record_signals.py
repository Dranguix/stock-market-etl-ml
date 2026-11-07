#!/usr/bin/env python3
"""
Registro inmutable de señales diarias — Track Record Builder.

Corre el SuperModel, guarda las señales con timestamp en:
  data/signals/history/YYYY-MM-DD.json

Cada archivo es append-only: si ya existe el día de hoy, solo actualiza
con las apuestas nuevas (no sobreescribe el timestamp original).

El historial sirve como track record verificable para:
  - Audit de performance vs predicciones pasadas
  - Cálculo de hit rate / Sharpe real en vivo (n_closed > 0 eventualmente)
  - Base para el reporte semanal vendible

Uso:
    python scripts/record_signals.py               # con Gemini, modo rápido (--fast)
    python scripts/record_signals.py --no-gemini   # solo cuantitativo
    python scripts/record_signals.py --full        # con LSTM (más lento)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

HISTORY_DIR = ROOT / "data" / "signals" / "history"
INDEX_FILE  = ROOT / "data" / "signals" / "history" / "index.json"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _load_index() -> dict:
    if INDEX_FILE.exists():
        try:
            return json.loads(INDEX_FILE.read_text())
        except Exception:
            pass
    return {"entries": []}


def _save_index(idx: dict) -> None:
    INDEX_FILE.write_text(json.dumps(idx, indent=2, ensure_ascii=False))


def run_pipeline(use_gemini: bool, skip_lstm: bool) -> dict:
    from src.ml.signal_pipeline import run_super_pipeline
    return run_super_pipeline(
        top_gemini_global=5,
        top_gemini_bmv=5,
        use_gemini=use_gemini,
        use_stacking=True,
        verbose=False,
        agent_role="trader",
        skip_lstm=skip_lstm,
    )


def build_record(result: dict, ts: datetime) -> dict:
    """Extrae lo esencial para el historial (compacto, auditable)."""
    bets = result.get("actionable_bets_bmv") or result.get("actionable_bets") or []
    ranking = result.get("master_ranking") or []

    # Top-10 del ranking con datos suficientes para auditar
    top10 = []
    for item in ranking[:10]:
        g = item.get("gemini") or {}
        top10.append({
            "rank": item.get("rank_global"),
            "ticker": item.get("ticker"),
            "conviction": round(float(item.get("conviction", 0)), 4),
            "veredicto": g.get("veredicto") or g.get("accion") or "—",
            "apostaria": g.get("apostaria") or g.get("accion_final") or "—",
        })

    record = {
        "date": ts.strftime("%Y-%m-%d"),
        "timestamp_utc": ts.isoformat(),
        "model_version": "supermodel_v3_gim",        # bump al añadir features mayores
        "universe": "bmv",
        "regime": (result.get("insights_bundle") or {}).get("regime_info", {}).get("estado", "?"),
        "bets": [
            {
                "ticker": b.get("ticker"),
                "conviction": round(float(b.get("conviction", 0)), 4),
                "direction": "LONG",
                "horizon_days": 20,
            }
            for b in bets
        ],
        "n_bets": len(bets),
        "top10_ranking": top10,
        "hash": "",   # se llena abajo
    }

    # Hash SHA-256 del contenido esencial (sin el hash mismo) — inmutabilidad
    canonical = json.dumps(
        {k: v for k, v in record.items() if k != "hash"},
        sort_keys=True, ensure_ascii=False,
    )
    record["hash"] = _sha256(canonical)
    return record


def save_record(record: dict) -> Path:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    date_str = record["date"]
    out = HISTORY_DIR / f"{date_str}.json"

    if out.exists():
        # Si ya existe, preservamos el timestamp original y solo actualizamos bets
        existing = json.loads(out.read_text())
        record["timestamp_utc"] = existing.get("timestamp_utc", record["timestamp_utc"])
        record["hash"] = existing.get("hash", record["hash"])
        print(f"  Ya existe {date_str}.json — actualizando bets (timestamp preservado)")

    out.write_text(json.dumps(record, indent=2, ensure_ascii=False))

    # Actualizar índice
    idx = _load_index()
    entry = {
        "date": date_str,
        "n_bets": record["n_bets"],
        "bets": [b["ticker"] for b in record["bets"]],
        "regime": record["regime"],
        "hash": record["hash"],
    }
    # Reemplazar si ya existe ese date
    idx["entries"] = [e for e in idx["entries"] if e["date"] != date_str]
    idx["entries"].append(entry)
    idx["entries"].sort(key=lambda e: e["date"])
    idx["last_updated"] = record["timestamp_utc"]
    _save_index(idx)

    return out


def print_summary(record: dict) -> None:
    print(f"\n  Fecha:    {record['date']}")
    print(f"  Régimen:  {record['regime']}")
    print(f"  Apuestas: {record['n_bets']}")
    for b in record["bets"]:
        print(f"    → {b['ticker']:<12} conv={b['conviction']:.3f} LONG h={b['horizon_days']}d")
    if not record["bets"]:
        print("    (ninguna apuesta activa hoy)")
    print(f"  Hash:     {record['hash']}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Registro diario de señales (track record)")
    parser.add_argument("--no-gemini", action="store_true", help="Omitir Gemini (más rápido)")
    parser.add_argument("--full", action="store_true", help="Incluir LSTM")
    parser.add_argument("--dry-run", action="store_true", help="No guardar, solo mostrar")
    args = parser.parse_args()

    print("\n══ Record de Señales ══\n")

    ts = datetime.now(tz=timezone.utc)
    print(f"  Corriendo SuperModel ({ts.strftime('%Y-%m-%d %H:%M UTC')})…")

    result = run_pipeline(
        use_gemini=not args.no_gemini,
        skip_lstm=not args.full,
    )

    record = build_record(result, ts)
    print_summary(record)

    if args.dry_run:
        print("  [dry-run] No guardado.")
        return

    out = save_record(record)
    print(f"  Guardado: {out}")
    print(f"  Índice:   {INDEX_FILE}\n")


if __name__ == "__main__":
    main()
