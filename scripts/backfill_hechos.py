#!/usr/bin/env python3
"""
Backfill de hechos (operaciones con casa compra/venta) desde DataBursátil.

GET /v2/descargas?archivo=hechos&fecha=YYYY-MM-DD → 7z con CSV de todas las
operaciones del día en BMV: EmisoraSerie, UltHecho, VolHecho, CasaCompra,
CasaVende, Hora, Folio, Importe, ...

Guardamos el flujo AGREGADO por (ticker, casa) y día — compacto y suficiente
para features de order flow (imbalance, concentración, smart money):

  data/microstructure/hechos/YYYY-MM-DD.parquet
    date, ticker, casa, vol_buy, vol_sell, imp_buy, imp_sell, n_buy, n_sell

Uso:
  python scripts/backfill_hechos.py --months 12
  python scripts/backfill_hechos.py --start 2025-07-01 --end 2026-06-30
  python scripts/backfill_hechos.py --fecha 2026-06-26   # un solo día
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import py7zr
import requests

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

HECHOS_DIR = ROOT / "data" / "microstructure" / "hechos"
MANIFEST_PATH = HECHOS_DIR / "manifest.json"
BASE_URL = "https://api.databursatil.com/v2/descargas"
SEVENZ_MAGIC = b"7z\xbc\xaf"

CSV_COLS = [
    "EmisoraSerie", "UltHecho", "VolHecho", "CasaCompra", "CasaVende",
    "Hora", "Folio", "Importe", "TipoOperacion", "Concertacion",
    "Liquidacion", "TipoValor", "PicoLote",
]


def normalize_ticker(emisora_serie: str) -> str:
    """'WALMEX *' → WALMEX; 'GFNORTE O' → GFNORTEO; 'FEMSA UBD' → FEMSAUBD."""
    parts = str(emisora_serie).split()
    if not parts:
        return ""
    emisora = parts[0]
    serie = parts[1] if len(parts) > 1 else ""
    if serie in ("*", ""):
        return emisora
    return emisora + serie


def parse_hechos_csv(raw: bytes) -> pd.DataFrame:
    """Parsea el CSV (línea 1 = fecha, línea 2 = header, sep ';')."""
    df = pd.read_csv(
        io.BytesIO(raw),
        sep=";",
        skiprows=1,
        dtype=str,
        engine="c",
        on_bad_lines="skip",
    )
    df.columns = [c.strip() for c in df.columns]
    needed = ["EmisoraSerie", "VolHecho", "CasaCompra", "CasaVende", "Importe"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"columnas faltantes en CSV: {missing}")

    out = pd.DataFrame({
        "ticker": df["EmisoraSerie"].map(normalize_ticker),
        "vol": pd.to_numeric(df["VolHecho"], errors="coerce").fillna(0).astype("int64"),
        "importe": pd.to_numeric(df["Importe"], errors="coerce").fillna(0.0),
        "casa_compra": df["CasaCompra"].str.strip(),
        "casa_vende": df["CasaVende"].str.strip(),
    })
    return out[out["ticker"] != ""]


def aggregate_flows(trades: pd.DataFrame, fecha: str) -> pd.DataFrame:
    """Agrega a (ticker, casa) con lados compra/venta separados."""
    buys = (
        trades.groupby(["ticker", "casa_compra"])
        .agg(vol_buy=("vol", "sum"), imp_buy=("importe", "sum"), n_buy=("vol", "size"))
        .reset_index()
        .rename(columns={"casa_compra": "casa"})
    )
    sells = (
        trades.groupby(["ticker", "casa_vende"])
        .agg(vol_sell=("vol", "sum"), imp_sell=("importe", "sum"), n_sell=("vol", "size"))
        .reset_index()
        .rename(columns={"casa_vende": "casa"})
    )
    agg = buys.merge(sells, on=["ticker", "casa"], how="outer")
    for col in ("vol_buy", "vol_sell", "n_buy", "n_sell"):
        agg[col] = agg[col].fillna(0).astype("int64")
    for col in ("imp_buy", "imp_sell"):
        agg[col] = agg[col].fillna(0.0)
    agg.insert(0, "date", fecha)
    return agg


class NoCreditsError(RuntimeError):
    """La API respondió 429 créditos insuficientes — abortar el backfill."""


def download_day(fecha: str, token: str, timeout: int = 180) -> bytes | None:
    """Descarga el 7z del día. None si no hay archivo (feriado/futuro)."""
    r = requests.get(
        BASE_URL,
        params={"token": token, "archivo": "hechos", "fecha": fecha},
        timeout=timeout,
    )
    if r.status_code == 429:
        raise NoCreditsError(r.text[:120])
    if r.status_code != 200 or not r.content.startswith(SEVENZ_MAGIC):
        return None
    return r.content


def extract_csv(archive: bytes) -> bytes | None:
    with tempfile.TemporaryDirectory() as tmp:
        with py7zr.SevenZipFile(io.BytesIO(archive)) as z:
            z.extractall(tmp)
        for path in Path(tmp).rglob("*"):
            if path.is_file() and path.suffix.lower() == ".csv":
                return path.read_bytes()
    return None


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text())
        except Exception:
            pass
    return {}


def _save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def process_day(fecha: str, token: str) -> tuple[str, int]:
    """Devuelve (status, n_rows). status: ok | no_file | error."""
    out_path = HECHOS_DIR / f"{fecha}.parquet"
    if out_path.exists():
        return "cached", 0

    archive = download_day(fecha, token)
    if archive is None:
        return "no_file", 0

    csv_raw = extract_csv(archive)
    if csv_raw is None:
        return "no_file", 0

    trades = parse_hechos_csv(csv_raw)
    if trades.empty:
        return "no_file", 0

    agg = aggregate_flows(trades, fecha)
    agg.to_parquet(out_path, index=False)
    return "ok", len(agg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill hechos BMV (order flow)")
    parser.add_argument("--months", type=int, default=12, help="meses hacia atrás")
    parser.add_argument("--start", help="fecha inicial YYYY-MM-DD (ignora --months)")
    parser.add_argument("--end", help="fecha final YYYY-MM-DD (default: ayer)")
    parser.add_argument("--fecha", help="un solo día YYYY-MM-DD")
    parser.add_argument("--sleep", type=float, default=0.5, help="pausa entre requests")
    args = parser.parse_args()

    token = os.getenv("DATABURSATIL_API_TOKEN")
    if not token:
        print("ERROR: DATABURSATIL_API_TOKEN no configurado")
        sys.exit(1)

    HECHOS_DIR.mkdir(parents=True, exist_ok=True)

    if args.fecha:
        days = [args.fecha]
    else:
        end = pd.Timestamp(args.end) if args.end else pd.Timestamp(date.today() - timedelta(days=1))
        start = pd.Timestamp(args.start) if args.start else end - pd.DateOffset(months=args.months)
        days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, end)]

    manifest = _load_manifest()
    retryable = ("error", "no_credits")
    pending = [
        d for d in days
        if d not in manifest or manifest[d].get("status") in retryable
    ]
    print(f"══ Backfill hechos: {len(days)} días hábiles, {len(pending)} pendientes ══")

    n_ok = n_skip = n_err = 0
    for i, fecha in enumerate(pending):
        try:
            status, n_rows = process_day(fecha, token)
            if status in ("ok", "cached"):
                n_ok += 1
            else:
                n_skip += 1
            manifest[fecha] = {"status": status, "rows": n_rows}
            print(f"  [{i + 1}/{len(pending)}] {fecha}: {status} ({n_rows} filas)", flush=True)
        except NoCreditsError:
            manifest[fecha] = {"status": "no_credits"}
            print(f"  [{i + 1}/{len(pending)}] {fecha}: SIN CRÉDITOS — abortando", flush=True)
            break
        except Exception as exc:
            n_err += 1
            manifest[fecha] = {"status": "error", "error": str(exc)[:200]}
            print(f"  [{i + 1}/{len(pending)}] {fecha}: ERROR {exc}", flush=True)

        if (i + 1) % 10 == 0:
            _save_manifest(manifest)
        time.sleep(args.sleep)

    _save_manifest(manifest)
    print(f"\n✅ Backfill terminado: {n_ok} ok, {n_skip} sin archivo, {n_err} errores")
    print(f"   Parquets en {HECHOS_DIR}")


if __name__ == "__main__":
    main()
