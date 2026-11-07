"""Dominio Lab/Trading: lectura del esquema lab y herramientas, con datos dummy."""
from __future__ import annotations

import asyncio
import json
import re
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

pytest.importorskip("claude_agent_sdk")

from scripts.build_lab_dummy import AS_OF, build, write  # noqa: E402
from src.agent.core_satellite import config  # noqa: E402
from src.agents_mcp.domains import lab  # noqa: E402
from src.db import lab_reader  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "sql" / "03_lab_schema.sql"
RAW_DEL_JOB = {"llm_used", "enter_on", "book", "plans", "regla", "gemelo", "vetos", "candidatos", "tendencia"}


def _columnas_del_esquema() -> dict[str, tuple[str, ...]]:
    out = {}
    for tabla, cuerpo in re.findall(r"CREATE TABLE IF NOT EXISTS lab\.(\w+) \((.*?)\n\);", SCHEMA.read_text(encoding="utf-8"), re.S):
        cols = []
        for linea in cuerpo.splitlines():
            linea = linea.strip()
            if linea and not linea.startswith("PRIMARY KEY"):
                cols.append(linea.split()[0])
        out[tabla] = tuple(cols)
    return out


@pytest.fixture
def carpeta(tmp_path) -> Path:
    write(build(), tmp_path)
    return tmp_path


@pytest.fixture
def fuente(carpeta, monkeypatch) -> lab_reader.FuenteArchivos:
    f = lab_reader.FuenteArchivos(carpeta)
    monkeypatch.setattr(lab_reader, "fuente", lambda: f)
    monkeypatch.setattr(lab, "_hoy", lambda: AS_OF)
    return f


def _correr(tool, args: dict | None = None) -> str:
    return asyncio.run(tool.handler(args or {}))["content"][0]["text"]


def test_el_lector_usa_las_columnas_del_esquema():
    esquema = _columnas_del_esquema()
    for tabla, cols in lab_reader.TABLAS.items():
        assert cols == esquema[tabla], tabla


@pytest.mark.parametrize("tabla", list(lab_reader.TABLAS))
def test_el_dummy_trae_exactamente_las_columnas_del_esquema(tabla):
    esquema = _columnas_del_esquema()[tabla]
    filas = (ROOT / "data" / "lab" / "dummy" / f"{tabla}.jsonl").read_text(encoding="utf-8").splitlines()
    assert filas
    for linea in filas:
        assert tuple(json.loads(linea)) == esquema


def test_el_dummy_guardado_es_el_que_genera_el_script(tmp_path):
    write(build(), tmp_path)
    for tabla in lab_reader.TABLAS:
        guardado = (ROOT / "data" / "lab" / "dummy" / f"{tabla}.jsonl").read_text(encoding="utf-8")
        assert (tmp_path / f"{tabla}.jsonl").read_text(encoding="utf-8") == guardado, tabla


def test_el_raw_tiene_la_forma_que_escribe_el_job():
    tablas = build()
    planes = {}
    for p in tablas["entry_plan"]:
        planes.setdefault(p["decision_id"], []).append({k: v for k, v in p.items() if k not in ("id", "decision_id")})
    for d in tablas["decision"]:
        assert set(d["raw"]) == RAW_DEL_JOB
        assert d["raw"]["regla"] == config()
        assert d["raw"]["plans"] == planes[d["id"]]
        assert d["cash_usd"] + sum(p["usd"] for p in planes[d["id"]]) == pytest.approx(50)


def test_los_resultados_cuadran_con_el_calculo_del_dominio(fuente):
    planes = {p["id"]: p for p in fuente.filas("entry_plan")}
    decisiones = {d["id"]: d for d in fuente.filas("decision")}
    for o in fuente.filas("outcome"):
        p = planes[o["entry_plan_id"]]
        d = decisiones[p["decision_id"]]
        entra = date.fromisoformat(p["enter_on"])
        sale = lab._mas_habiles(entra, lab._dias(p["horizon"]))
        stop_dd = None if p["stop_usd"] is None else d["raw"]["regla"]["catastrophe_dd"]
        r, toco, error = lab._retorno_precio(fuente, p["ticker"], date.fromisoformat(d["session_date"]), entra, sale, stop_dd)
        assert error is None
        assert toco is o["hit_stop"]
        assert r == pytest.approx(o["realized_return"], abs=1e-4)


def test_no_hay_resultado_antes_de_que_cierre_la_ventana():
    tablas = build()
    calificados = {o["entry_plan_id"] for o in tablas["outcome"]}
    for p in tablas["entry_plan"]:
        sale = lab._mas_habiles(date.fromisoformat(p["enter_on"]), 20)
        assert (p["id"] in calificados) is (sale <= AS_OF)


class _Cursor:
    description = [("id",), ("decided_at",), ("session_date",), ("model",), ("cash_usd",), ("motivo",), ("raw",)]

    def __init__(self, log):
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params):
        self.log.append((sql, params))

    def fetchall(self):
        return [(1, None, date(2026, 9, 30), "rules", Decimal("6.25"), "m", {"gemelo": {"SPY": 0.5}})]


class _Conexion:
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        return _Cursor(self.log)


def test_postgres_lee_solo_columnas_del_esquema_y_normaliza():
    log = []
    pg = lab_reader.FuentePostgres(connect=lambda: _Conexion(log))
    filas = pg.filas("decision", {"id": [1, 2]})
    sql, params = log[0]
    assert sql == (
        "SELECT id, decided_at, session_date, model, cash_usd, motivo, raw FROM lab.decision "
        "WHERE id = ANY(%(id)s)"
    )
    assert params == {"id": [1, 2]}
    assert filas[0]["session_date"] == "2026-09-30"
    assert filas[0]["cash_usd"] == 6.25 and isinstance(filas[0]["cash_usd"], float)
    assert filas[0]["raw"] == {"gemelo": {"SPY": 0.5}}


@pytest.mark.parametrize("fuente_cls", [lab_reader.FuentePostgres, lab_reader.FuenteArchivos])
def test_no_acepta_columnas_fuera_del_esquema(fuente_cls, tmp_path):
    f = fuente_cls(lambda: None) if fuente_cls is lab_reader.FuentePostgres else fuente_cls(tmp_path)
    (tmp_path / "entry_plan.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        f.filas("entry_plan", {"entry_px": [1]})


def test_fuente_elige_dummy_sin_postgres(monkeypatch):
    for var in ("LAB_FUENTE", "LAB_DATABASE_URL", "LAB_PGHOST", "LAB_HOST", "LAB_PGUSER", "LAB_USER"):
        monkeypatch.delenv(var, raising=False)
    assert isinstance(lab_reader.fuente(), lab_reader.FuenteArchivos)
    monkeypatch.setenv("LAB_DATABASE_URL", "postgresql://x")
    assert isinstance(lab_reader.fuente(), lab_reader.FuentePostgres)
    monkeypatch.setenv("LAB_FUENTE", "dummy")
    assert isinstance(lab_reader.fuente(), lab_reader.FuenteArchivos)
    monkeypatch.setenv("LAB_FUENTE", "excel")
    with pytest.raises(ValueError):
        lab_reader.fuente()


def test_el_agente_trae_las_herramientas_de_lab():
    assert [t.name for t in lab.lab_agent.tools] == [
        "schema_lab", "query_lab", "ultima_decision_lab", "resultados_lab", "veto_vs_gemelo_lab",
    ]


def test_cada_respuesta_dice_que_son_datos_dummy(fuente):
    for tool in (lab.ultima_decision_lab, lab.resultados_lab, lab.veto_vs_gemelo_lab, lab.query_lab):
        assert "datos dummy" in _correr(tool, {"pregunta": "estado"})


def test_ultima_decision_trae_plan_vetos_y_gemelo(fuente):
    texto = _correr(lab.ultima_decision_lab, {"hasta": "2026-09-30"})
    assert "Decisión 3 del 2026-09-30" in texto
    assert "KIMBERA: recorte_distribucion" in texto
    assert "FIBRAPL $6.25 (satélite" in texto
    assert "Gemelo (la regla sin veto): SPY 50.0%, NAFTRAC 25.0%, ALSEA 12.5%, KIMBERA 12.5%." in texto


def test_ultima_decision_sin_veto_dice_que_la_regla_corrio_sola(fuente):
    texto = _correr(lab.ultima_decision_lab)
    assert "Decisión 4 del 2026-10-30" in texto
    assert "apagado (la regla corrió sola)" in texto
    assert "igual al libro" in texto


def test_resultados_no_inventa_lo_pendiente(fuente):
    texto = _correr(lab.resultados_lab, {"decisiones": 2})
    bloque_4 = texto.split("Decisión 3")[0]
    assert "pendiente, se califica el 2026-11-30" in bloque_4
    assert "%" not in bloque_4
    assert "no hay resultado del libro todavía" in bloque_4


def test_resultados_marca_el_stop(fuente):
    texto = _correr(lab.resultados_lab, {"decisiones": 4})
    assert re.search(r"FUNO11 \$6\.25: -14\.\d\d% \(-0\.\d\d USD, tocó el stop\)", texto)


def test_veto_vs_gemelo_calcula_el_aporte(fuente):
    texto = _correr(lab.veto_vs_gemelo_lab)
    assert "Haiku vetó KIMBERA" in texto
    assert "FIBRAPL libro 12.5% vs gemelo 0.0%, KIMBERA libro 0.0% vs gemelo 12.5%" in texto
    aporte = float(re.search(r"Libro − gemelo: ([+-]\d+\.\d\d) USD", texto).group(1))
    fibrapl = float(re.search(r"FIBRAPL: ([+-]\d+\.\d\d)%", texto).group(1)) / 100
    kimbera = float(re.search(r"KIMBERA: ([+-]\d+\.\d\d)%", texto).group(1)) / 100
    assert aporte == pytest.approx(6.25 * (fibrapl - kimbera), abs=0.01)
    assert "no hay evidencia" in texto


def test_veto_vs_gemelo_espera_a_que_cierre_la_ventana(fuente, monkeypatch):
    monkeypatch.setattr(lab, "_hoy", lambda: date(2026, 10, 15))
    texto = _correr(lab.veto_vs_gemelo_lab)
    assert "Pendiente: la ventana cierra el 2026-10-29." in texto
    assert "Libro − gemelo" not in texto


def test_sin_decisiones_lo_dice(tmp_path, monkeypatch):
    monkeypatch.setattr(lab_reader, "fuente", lambda: lab_reader.FuenteArchivos(tmp_path))
    for tool in (lab.ultima_decision_lab, lab.resultados_lab, lab.veto_vs_gemelo_lab):
        assert "No hay decisiones en lab.decision" in _correr(tool)


def test_si_postgres_falla_responde_en_texto(monkeypatch):
    def roto():
        raise RuntimeError("sin red")

    monkeypatch.setattr(lab_reader, "fuente", lambda: lab_reader.FuentePostgres(connect=roto))
    texto = _correr(lab.ultima_decision_lab)
    assert texto.startswith("Fuente: Postgres, esquema lab.")
    assert "No pude leer lab (RuntimeError: sin red)" in texto
