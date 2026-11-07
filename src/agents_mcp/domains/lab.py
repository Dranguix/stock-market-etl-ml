"""Agente de dominio: Lab/Trading — decisiones, plan, veto y gemelo.

Esquema real, Postgres esquema `lab` (sql/03_lab_schema.sql):
  lab.decision     — una fila por fecha de rebalanceo: regla, pesos, veto,
                     gemelo (raw jsonb)
  lab.entry_plan   — una fila por posición del plan (ticker, USD, horizonte,
                     stop, precio de entrada)
  lab.outcome      — score del plan a 20 días hábiles
  lab.price_daily / lab.news_daily / lab.metric_daily — snapshots diarios

Sin Postgres, los planes caen a data/lab/entry_plans.jsonl.

Este agente NO necesita conocer otros dominios — si una pregunta mezcla
Lab con Noticias o Mercado, eso lo resuelve el orquestador llamando a los
agentes por separado y combinando respuestas (ver ejemplo end-to-end en
la presentación de arquitectura).

TODO equipo Trading Agent: implementar query_lab de verdad sobre Postgres
`lab` (o el jsonl de fallback).
"""
from __future__ import annotations

import json
from pathlib import Path

from claude_agent_sdk import tool

from src.agents_mcp.domain_agent import DomainAgent

ROOT = Path(__file__).resolve().parents[3]
BOOK_PATH = ROOT / "data" / "agent" / "book.json"
PLANS_PATH = ROOT / "data" / "lab" / "entry_plans.jsonl"
LLM_SIM_PATHS = [
    ROOT / "data" / "eval" / "forecast_demo" / "llm_sim_3m.json",
    ROOT / "data" / "eval" / "forecast_demo" / "llm_sim_2m.json",
    ROOT / "data" / "eval" / "forecast_demo" / "llm_sim_1m.json",
]

SYSTEM_PROMPT = """Eres el agente de dominio de Lab/Trading: la decisión
real del libro (regla, pesos, veto, gemelo), su plan de entrada y su
calificación 20 días después. Explica siempre si una decisión vino de la
regla sola, o si el veto de Haiku cambió algo respecto al gemelo. No
inventes un outcome que todavía no se calificó."""


@tool(
    "schema_lab",
    "Describe las tablas de decisión/plan/outcome del dominio Lab/Trading.",
    {},
)
async def schema_lab(args: dict) -> dict:
    return {
        "content": [{
            "type": "text",
            "text": (
                "Postgres esquema lab (sql/03_lab_schema.sql): "
                "lab.decision (regla, pesos, veto, gemelo en raw jsonb), "
                "lab.entry_plan (ticker, usd, horizonte='20d', stop, "
                "precio de entrada), lab.outcome (score a 20 días hábiles). "
                "Fallback sin Postgres: data/lab/entry_plans.jsonl. "
                "Job que escribe esto: scripts/gcp_daily_job.py "
                "(run_decide, run_score)."
            ),
        }]
    }


def _regla_vigente() -> str:
    """Regla núcleo-satélite vigente, leída de src/agent/core_satellite.py
    (valores reales, no copiados a mano, para que nunca queden desfasados)."""
    try:
        from src.agent.core_satellite import (
            CATASTROPHE_DD,
            CORE,
            DRIFT_BAND,
            ENTER_TOP,
            EXIT_TOP,
            SAT_MAX_NAMES,
            SAT_WEIGHT,
            TREND_DAYS,
        )
    except Exception as exc:  # import pesado (pandas/numpy/datos) puede fallar
        return f"No pude importar src/agent/core_satellite.py ({exc})."

    core_txt = ", ".join(f"{nombre} {peso*100:.0f}%" for nombre, (_, peso) in CORE.items())
    return (
        f"Núcleo {sum(p for _, p in CORE.values())*100:.0f}%: {core_txt}. "
        f"Satélite {SAT_WEIGHT*100:.0f}% en hasta {SAT_MAX_NAMES} emisoras BMV/FIBRAs "
        f"(entra en el top {ENTER_TOP*100:.0f}%, sale al caer debajo del top "
        f"{EXIT_TOP*100:.0f}%, reselección trimestral). Filtro de tendencia: SMA de "
        f"{TREND_DAYS} sesiones (~10 meses). Stop de catástrofe del satélite: "
        f"{CATASTROPHE_DD*100:.0f}% debajo del máximo desde la entrada. "
        f"Banda de deriva: {DRIFT_BAND*100:.0f} pp antes de rebalancear."
    )


def _estado_libro() -> str:
    """Estado actual del libro (data/agent/book.json), sin ejecutar nada pesado."""
    if not BOOK_PATH.exists():
        return (
            "No encontré data/agent/book.json (el libro todavía no tiene snapshot "
            "guardado, o corre solo en el job diario de GCP)."
        )
    try:
        data = json.loads(BOOK_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return f"data/agent/book.json existe pero no se pudo leer ({exc})."
    posiciones = data.get("positions", {})
    pos_txt = ", ".join(
        f"{t}: ${float(p.get('usd', 0)):.2f} USD" for t, p in posiciones.items()
    ) or "sin posiciones abiertas"
    return (
        f"Libro: efectivo ${float(data.get('cash_usd', 0)):.2f} USD de un "
        f"presupuesto de ${float(data.get('budget_usd', 0)):.2f} USD, abierto desde "
        f"{data.get('opened_at')}. Posiciones: {pos_txt}."
    )


def _plan_entrada_reciente(n: int = 5) -> str:
    """Últimas filas de data/lab/entry_plans.jsonl (fallback sin Postgres)."""
    if not PLANS_PATH.exists():
        return "No encontré data/lab/entry_plans.jsonl."
    lineas = [l for l in PLANS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lineas:
        return "data/lab/entry_plans.jsonl está vacío."
    filas = []
    for linea in lineas[-n:]:
        try:
            row = json.loads(linea)
        except json.JSONDecodeError:
            continue
        filas.append(
            f"{row.get('ticker')} ${row.get('usd')} USD, entra {row.get('enter_on')} "
            f"a {row.get('entry_px')}, stop {row.get('stop')}, horizonte "
            f"{row.get('horizon_days')}d (p_sube={row.get('p_sube')})"
        )
    return f"Últimas {len(filas)} filas del plan de entrada: " + "; ".join(filas)


def _decision_reciente_llm() -> str | None:
    """Última decisión real del agente LLM, tomada del sim más reciente
    disponible en data/eval/forecast_demo/llm_sim_*m.json."""
    for path in LLM_SIM_PATHS:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        decisiones = data.get("decisions") or []
        if not decisiones:
            continue
        ultima = decisiones[-1]
        pesos = ", ".join(f"{t}: {w*100:.0f}%" for t, w in ultima.get("weights", {}).items())
        return (
            f"Decisión real más reciente del agente LLM ({path.name}, "
            f"asof {ultima.get('asof')}): pesos [{pesos}]. "
            f"Razonamiento: {ultima.get('motivo', '').strip()}"
        )
    return None


@tool(
    "query_lab",
    "Responde una pregunta en lenguaje natural sobre decisiones/plan/outcome: "
    "la regla núcleo-satélite vigente, el estado del libro y la decisión "
    "reciente (de book.json o del sim más reciente del agente LLM).",
    {"pregunta": str},
)
async def query_lab(args: dict) -> dict:
    partes = [_regla_vigente(), _estado_libro(), _plan_entrada_reciente()]
    decision_llm = _decision_reciente_llm()
    if decision_llm:
        partes.append(decision_llm)
    return {"content": [{"type": "text", "text": "\n\n".join(partes)}]}


lab_agent = DomainAgent(
    name="lab",
    system_prompt=SYSTEM_PROMPT,
    tools=[schema_lab, query_lab],
)
