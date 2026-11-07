"""Lectura del esquema lab (sql/03_lab_schema.sql), de Postgres o de archivos.

Las dos fuentes devuelven las mismas filas: las columnas exactas de cada
tabla, con fechas en ISO y numeric como float. Así el dominio Lab/Trading no
sabe de dónde vino el dato.

  FuentePostgres: SELECT plano sobre lab.*, con la conexión de lab_store.
  FuenteArchivos: data/lab/dummy/<tabla>.jsonl, una fila por línea. Son
    datos dummy (scripts/build_lab_dummy.py), no decisiones reales.

fuente() elige: LAB_FUENTE=postgres o LAB_FUENTE=dummy. Sin LAB_FUENTE usa
Postgres si están las variables de lab_store.connect(), y si no, el dummy.
Solo lee: nada aquí escribe en lab.
"""
from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from src.db import lab_store

ROOT = Path(__file__).resolve().parents[2]
DUMMY_DIR = ROOT / "data" / "lab" / "dummy"

TABLAS: dict[str, tuple[str, ...]] = {
    "decision": ("id", "decided_at", "session_date", "model", "cash_usd", "motivo", "raw"),
    "entry_plan": ("id", "decision_id", "ticker", "usd", "horizon", "stop_usd", "p_sube", "enter_on"),
    "outcome": ("id", "entry_plan_id", "scored_at", "realized_return", "hit_stop", "note"),
    "price_daily": (
        "ticker", "trade_date", "open", "high", "low", "close", "volume", "moneda", "source", "loaded_at",
    ),
}


def _valor(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str) and value.startswith(("{", "[")):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _fila(tabla: str, row: dict) -> dict:
    return {col: _valor(row.get(col)) for col in TABLAS[tabla]}


def _es_columna(tabla: str, col: str) -> None:
    if col not in TABLAS[tabla]:
        raise ValueError(f"lab.{tabla} no tiene la columna {col}")


class Fuente:
    nombre = "fuente"
    es_dummy = False

    def filas(self, tabla: str, donde: dict[str, Iterable] | None = None) -> list[dict]:
        raise NotImplementedError

    def decisiones(self, n: int | None = None, hasta: str | None = None) -> list[dict]:
        """lab.decision, de la más reciente a la más vieja."""
        rows = self.filas("decision")
        if hasta:
            rows = [r for r in rows if str(r["session_date"]) <= hasta]
        rows.sort(key=lambda r: (str(r["session_date"]), r["id"]), reverse=True)
        return rows[:n] if n else rows

    def planes(self, decision_ids: Iterable[int]) -> list[dict]:
        return self.filas("entry_plan", {"decision_id": list(decision_ids)})

    def resultados(self, plan_ids: Iterable[int]) -> list[dict]:
        return self.filas("outcome", {"entry_plan_id": list(plan_ids)})

    def precios(self, tickers: Iterable[str], desde: str, hasta: str) -> list[dict]:
        rows = self.filas("price_daily", {"ticker": list(tickers)})
        return [r for r in rows if desde <= str(r["trade_date"]) <= hasta]


class FuenteArchivos(Fuente):
    es_dummy = True

    def __init__(self, carpeta: Path = DUMMY_DIR):
        self.carpeta = Path(carpeta)
        try:
            ruta = self.carpeta.relative_to(ROOT)
        except ValueError:
            ruta = self.carpeta
        self.nombre = f"datos dummy de {ruta.as_posix()} (no son decisiones reales)"

    def filas(self, tabla: str, donde: dict[str, Iterable] | None = None) -> list[dict]:
        path = self.carpeta / f"{tabla}.jsonl"
        if not path.exists():
            return []
        rows = [
            _fila(tabla, json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for col, valores in (donde or {}).items():
            _es_columna(tabla, col)
            permitidos = set(valores)
            rows = [r for r in rows if r[col] in permitidos]
        return rows


class FuentePostgres(Fuente):
    nombre = "Postgres, esquema lab"

    def __init__(self, connect: Callable[[], Any] = lab_store.connect):
        self.connect = connect

    def filas(self, tabla: str, donde: dict[str, Iterable] | None = None) -> list[dict]:
        cols = ", ".join(TABLAS[tabla])
        sql = f"SELECT {cols} FROM lab.{tabla}"
        params: dict[str, list] = {}
        condiciones = []
        for col, valores in (donde or {}).items():
            _es_columna(tabla, col)
            condiciones.append(f"{col} = ANY(%({col})s)")
            params[col] = list(valores)
        if condiciones:
            sql += " WHERE " + " AND ".join(condiciones)
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                nombres = [d[0] for d in cur.description]
                return [_fila(tabla, dict(zip(nombres, row))) for row in cur.fetchall()]


def hay_postgres() -> bool:
    return bool(
        lab_store._env("LAB_DATABASE_URL")
        or (lab_store._env("LAB_PGHOST", "LAB_HOST") and lab_store._env("LAB_PGUSER", "LAB_USER"))
    )


def fuente() -> Fuente:
    elegida = (os.environ.get("LAB_FUENTE") or "").strip().lower()
    if elegida == "postgres" or (not elegida and hay_postgres()):
        return FuentePostgres()
    if elegida not in ("", "dummy"):
        raise ValueError(f"LAB_FUENTE={elegida}: usa postgres o dummy")
    # Permite leer el mismo sandbox de pruebas del Trading Agent sin alterar
    # los archivos dummy originales del equipo.
    if elegida == "dummy" and os.getenv("LAB_DUMMY_DIR"):
        return FuenteArchivos(Path(os.environ["LAB_DUMMY_DIR"]).expanduser())
    return FuenteArchivos()
