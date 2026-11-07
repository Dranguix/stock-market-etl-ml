"""Agente de dominio: Lab/Trading — decisiones, plan, veto y gemelo.

Esquema real, Postgres esquema `lab` (sql/03_lab_schema.sql):
  lab.decision     — una fila por fecha de rebalanceo: modelo, efectivo,
                     motivo y raw jsonb (regla, gemelo, vetos, candidatos,
                     tendencia, plans)
  lab.entry_plan   — una fila por posición del plan (ticker, usd, horizon,
                     stop_usd, p_sube, enter_on)
  lab.outcome      — score del plan a 20 días hábiles (realized_return,
                     hit_stop, note)
  lab.price_daily / lab.news_daily / lab.metric_daily — snapshots diarios

La lectura vive en src/db/lab_reader.py. Sin Postgres lee los datos dummy de
data/lab/dummy (mismas columnas, scripts/build_lab_dummy.py) y cada respuesta
lo dice.

Este agente NO necesita conocer otros dominios — si una pregunta mezcla
Lab con Noticias o Mercado, eso lo resuelve el orquestador llamando a los
agentes por separado y combinando respuestas (ver ejemplo end-to-end en
la presentación de arquitectura).
"""
from __future__ import annotations

import json
import re
from datetime import date, timedelta
from pathlib import Path

from claude_agent_sdk import tool

from src.agents_mcp.domain_agent import DomainAgent
from src.db import lab_reader

ROOT = Path(__file__).resolve().parents[3]
BOOK_PATH = ROOT / "data" / "agent" / "book.json"
LLM_SIM_PATHS = [
    ROOT / "data" / "eval" / "forecast_demo" / "llm_sim_3m.json",
    ROOT / "data" / "eval" / "forecast_demo" / "llm_sim_2m.json",
    ROOT / "data" / "eval" / "forecast_demo" / "llm_sim_1m.json",
]
FX_TICKER = "USDMXN"  # así lo escribe run_ingest en lab.price_daily

SYSTEM_PROMPT = """Eres el agente de dominio de Lab/Trading: la decisión
real del libro (regla, pesos, veto, gemelo), su plan de entrada y su
calificación 20 días después. Explica siempre si una decisión vino de la
regla sola, o si el veto de Haiku cambió algo respecto al gemelo. No
inventes un outcome que todavía no se calificó. Cada herramienta dice su
fuente en la primera línea: si son datos dummy, dilo en tu respuesta."""


def _texto(texto: str) -> dict:
    return {"content": [{"type": "text", "text": texto}]}


def _hoy() -> date:
    return date.today()


def _fecha(value) -> date:
    return date.fromisoformat(str(value)[:10])


def _dias(horizon) -> int:
    """Horizonte de entry_plan en días hábiles, como _parse_horizon del job."""
    text = str(horizon or "").strip().lower()
    if text.endswith("d") and text[:-1].isdigit():
        return int(text[:-1])
    if text.endswith("s") and text[:-1].isdigit():
        return 5 * int(text[:-1])
    return int(text) if text.isdigit() else 20


def _mas_habiles(day: date, n: int) -> date:
    """Días hábiles lun-vie, sin asuetos BMV, igual que _add_bdays del job."""
    while n > 0:
        day += timedelta(days=1)
        if day.weekday() < 5:
            n -= 1
    return day


def _salida(plan: dict, outcome: dict | None) -> date:
    match = re.search(r"exit_on=(\d{4}-\d{2}-\d{2})", (outcome or {}).get("note") or "")
    if match:
        return _fecha(match.group(1))
    return _mas_habiles(_fecha(plan["enter_on"]), _dias(plan["horizon"]))


def _usd(x: float | None) -> str:
    return "—" if x is None else f"${x:,.2f}"


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:+.2f}%"


def _pesos_libro(decision: dict, planes: list[dict]) -> tuple[dict[str, float], float]:
    presupuesto = sum(float(p["usd"] or 0) for p in planes) + float(decision["cash_usd"] or 0)
    if not presupuesto:
        return {}, 0.0
    return {p["ticker"]: float(p["usd"] or 0) / presupuesto for p in planes}, presupuesto


def _cargar(fuente: lab_reader.Fuente, decisiones: list[dict]) -> tuple[dict, dict]:
    planes = fuente.planes([d["id"] for d in decisiones])
    resultados = fuente.resultados([p["id"] for p in planes])
    por_decision: dict[int, list[dict]] = {}
    for p in sorted(planes, key=lambda r: r["id"]):
        por_decision.setdefault(p["decision_id"], []).append(p)
    return por_decision, {o["entry_plan_id"]: o for o in resultados}


def _texto_decision(decision: dict, planes: list[dict]) -> str:
    raw = decision.get("raw") or {}
    nucleo = set((raw.get("regla") or {}).get("core") or ())
    enter_on = raw.get("enter_on") or (planes[0]["enter_on"] if planes else "—")
    veto = "activo" if raw.get("llm_used") else "apagado (la regla corrió sola)"
    lineas = [
        f"Decisión {decision['id']} del {decision['session_date']}, entra el {enter_on}. "
        f"Modelo: {decision['model']}. Veto de Haiku: {veto}.",
    ]
    partes = []
    for p in planes:
        if p["ticker"] in nucleo:
            partes.append(f"{p['ticker']} {_usd(p['usd'])} (núcleo, sin stop)")
        else:
            partes.append(f"{p['ticker']} {_usd(p['usd'])} (satélite, stop {_usd(p['stop_usd'])})")
    partes.append(f"efectivo {_usd(decision['cash_usd'])}")
    lineas.append("Plan: " + ", ".join(partes) + ".")
    abajo = [t for t, ok in (raw.get("tendencia") or {}).items() if ok is False]
    if abajo:
        lineas.append(f"Debajo de su tendencia: {', '.join(abajo)}.")
    if raw.get("candidatos"):
        lineas.append(f"Candidatos de la reselección: {', '.join(raw['candidatos'])}.")
    vetos = raw.get("vetos") or {}
    if vetos:
        lineas.append("Vetos: " + "; ".join(f"{t}: {m}" for t, m in vetos.items()) + ".")
    else:
        lineas.append("Sin vetos.")
    pesos, _ = _pesos_libro(decision, planes)
    gemelo = raw.get("gemelo")
    if gemelo is None:
        lineas.append("Esta decisión no guardó gemelo.")
    elif _mismos_pesos(pesos, gemelo):
        lineas.append("Gemelo (la regla sin veto): igual al libro.")
    else:
        g = ", ".join(f"{t} {w * 100:.1f}%" for t, w in gemelo.items())
        lineas.append(f"Gemelo (la regla sin veto): {g}.")
    lineas.append(f"Motivo registrado: {decision['motivo']}")
    return "\n".join(lineas)


def _mismos_pesos(a: dict, b: dict) -> bool:
    return all(abs(float(a.get(t, 0)) - float(b.get(t, 0))) < 1e-6 for t in set(a) | set(b))


def _retorno_precio(
    fuente: lab_reader.Fuente, ticker: str, sesion: date, entra: date, sale: date, stop_dd: float | None,
) -> tuple[float | None, bool, str | None]:
    """Retorno en USD de cierre a cierre con stop de catástrofe, como run_score.

    El stop sale del cierre del día de la decisión, como lo pone la regla.
    Regresa (retorno, tocó_stop, motivo si no se pudo)."""
    rows = fuente.precios([ticker, FX_TICKER], sesion.isoformat(), sale.isoformat())
    fx = {r["trade_date"]: r["close"] for r in rows if r["ticker"] == FX_TICKER}
    serie = sorted((r for r in rows if r["ticker"] == ticker), key=lambda r: r["trade_date"])
    if not serie:
        return None, False, f"sin precios de {ticker} en lab.price_daily"

    def a_usd(row: dict, col: str) -> float | None:
        if row["moneda"] != "MXN":
            return row[col]
        tipo = fx.get(row["trade_date"])
        return row[col] / tipo if tipo else None

    base = [r for r in serie if r["trade_date"] <= sesion.isoformat()]
    ventana = [r for r in serie if entra.isoformat() <= r["trade_date"] <= sale.isoformat()]
    if not ventana or ventana[-1]["trade_date"] != sale.isoformat():
        return None, False, f"faltan precios de {ticker} hasta el {sale}"
    entrada, cierre = a_usd(ventana[0], "close"), a_usd(ventana[-1], "close")
    minimos = [a_usd(r, "low") for r in ventana]
    if entrada is None or cierre is None or None in minimos:
        return None, False, f"falta {FX_TICKER} para pasar {ticker} a dólares"
    stop = None
    if stop_dd is not None and base:
        cierre_decision = a_usd(base[-1], "close")
        stop = None if cierre_decision is None else cierre_decision * (1 - stop_dd)
    toco = stop is not None and min(minimos) <= stop
    return ((stop if toco else cierre) - entrada) / entrada, toco, None


def _fuente_o_error() -> tuple[lab_reader.Fuente | None, str | None]:
    try:
        return lab_reader.fuente(), None
    except Exception as exc:
        return None, f"No pude abrir la fuente de lab ({exc})."


def _leer(fn, fuente: lab_reader.Fuente) -> str:
    try:
        return f"Fuente: {fuente.nombre}.\n\n{fn(fuente)}"
    except Exception as exc:
        return f"Fuente: {fuente.nombre}.\n\nNo pude leer lab ({type(exc).__name__}: {exc})."


def ultima_decision(fuente: lab_reader.Fuente, hasta: str | None = None) -> str:
    decisiones = fuente.decisiones(n=1, hasta=hasta)
    if not decisiones:
        return "No hay decisiones en lab.decision" + (f" hasta el {hasta}." if hasta else ".")
    planes, _ = _cargar(fuente, decisiones)
    d = decisiones[0]
    return _texto_decision(d, planes.get(d["id"], []))


def resultados(fuente: lab_reader.Fuente, n: int = 6) -> str:
    decisiones = fuente.decisiones(n=n)
    if not decisiones:
        return "No hay decisiones en lab.decision."
    planes, scores = _cargar(fuente, decisiones)
    hoy = _hoy()
    bloques = []
    for d in decisiones:
        filas = planes.get(d["id"], [])
        lineas = [f"Decisión {d['id']} del {d['session_date']}:"]
        pnl, pendientes = 0.0, 0
        for p in filas:
            o = scores.get(p["id"])
            sale = _salida(p, o)
            if o is None:
                pendientes += 1
                cuando = "se califica" if sale > hoy else "debió calificarse"
                lineas.append(f"- {p['ticker']} {_usd(p['usd'])}: pendiente, {cuando} el {sale}.")
                continue
            ganancia = float(p["usd"] or 0) * float(o["realized_return"])
            pnl += ganancia
            stop = ", tocó el stop" if o["hit_stop"] else ""
            lineas.append(
                f"- {p['ticker']} {_usd(p['usd'])}: {_pct(o['realized_return'])} "
                f"({ganancia:+.2f} USD{stop}), salió el {sale}."
            )
        _, presupuesto = _pesos_libro(d, filas)
        if filas and not pendientes and presupuesto:
            lineas.append(f"Libro: {pnl:+.2f} USD, {pnl / presupuesto * 100:+.2f}% sobre {_usd(presupuesto)}.")
        elif pendientes:
            lineas.append(f"{pendientes} de {len(filas)} planes sin calificar: no hay resultado del libro todavía.")
        bloques.append("\n".join(lineas))
    return "\n\n".join(bloques)


def veto_vs_gemelo(fuente: lab_reader.Fuente, n: int = 12) -> str:
    decisiones = fuente.decisiones(n=n)
    if not decisiones:
        return "No hay decisiones en lab.decision."
    planes, _ = _cargar(fuente, decisiones)
    hoy = _hoy()
    bloques, aportes = [], []
    for d in decisiones:
        raw = d.get("raw") or {}
        filas = planes.get(d["id"], [])
        pesos, presupuesto = _pesos_libro(d, filas)
        gemelo = raw.get("gemelo")
        if gemelo is None or _mismos_pesos(pesos, gemelo):
            continue
        sesion = _fecha(d["session_date"])
        entra = _fecha(raw.get("enter_on") or filas[0]["enter_on"])
        sale = _mas_habiles(entra, _dias(filas[0]["horizon"] if filas else "20d"))
        vetos = raw.get("vetos") or {}
        distintos = sorted(t for t in set(pesos) | set(gemelo) if abs(pesos.get(t, 0) - gemelo.get(t, 0)) > 1e-6)
        lineas = [
            f"Decisión {d['id']} del {d['session_date']}: "
            + ("; ".join(f"Haiku vetó {t} ({m})" for t, m in vetos.items()) or "el libro difiere del gemelo")
            + ".",
            "Cambia: " + ", ".join(
                f"{t} libro {pesos.get(t, 0) * 100:.1f}% vs gemelo {gemelo.get(t, 0) * 100:.1f}%" for t in distintos
            ) + ".",
        ]
        if sale > hoy:
            lineas.append(f"Pendiente: la ventana cierra el {sale}.")
            bloques.append("\n".join(lineas))
            continue
        nucleo = set((raw.get("regla") or {}).get("core") or ())
        stop_dd = (raw.get("regla") or {}).get("catastrophe_dd")
        aporte, faltan = 0.0, []
        for t in distintos:
            r, toco, error = _retorno_precio(fuente, t, sesion, entra, sale, None if t in nucleo else stop_dd)
            if error:
                faltan.append(error)
                continue
            aporte += (pesos.get(t, 0) - gemelo.get(t, 0)) * presupuesto * r
            lineas.append(f"- {t}: {_pct(r)}" + (" (tocó el stop)" if toco else "") + f" del {entra} al {sale}.")
        if faltan:
            lineas.append("No se puede comparar: " + "; ".join(faltan) + ".")
        else:
            aportes.append(aporte)
            efecto = "ayudó" if aporte > 0 else "restó" if aporte < 0 else "no cambió nada"
            lineas.append(f"Libro − gemelo: {aporte:+.2f} USD ({aporte / presupuesto * 100:+.2f} pp). El veto {efecto}.")
        bloques.append("\n".join(lineas))
    if not bloques:
        return f"En las últimas {len(decisiones)} decisiones el libro fue igual al gemelo: el veto no cambió nada."
    total = (
        f"Total calificado: {sum(aportes):+.2f} USD en {len(aportes)} decisión(es) con veto. "
        "Con tan pocas decisiones no hay evidencia de que el veto agregue valor."
        if aportes else "Ninguna decisión con veto está calificada todavía."
    )
    metodo = (
        "Método: para los nombres que cambian, mismo cálculo que run_score del job: cierre de entrada "
        "a cierre de salida en USD (USDMXN de lab.price_daily) y stop de catástrofe del satélite "
        "contra el mínimo del día."
    )
    return "\n\n".join(bloques + [total, metodo])


def resumen(fuente: lab_reader.Fuente) -> str:
    decisiones = fuente.decisiones()
    if not decisiones:
        return "No hay decisiones en lab.decision."
    planes, scores = _cargar(fuente, decisiones)
    total = sum(len(v) for v in planes.values())
    con_veto = sum(1 for d in decisiones if (d.get("raw") or {}).get("vetos"))
    return (
        f"{len(decisiones)} decisiones, del {decisiones[-1]['session_date']} al "
        f"{decisiones[0]['session_date']}. {total} planes: {len(scores)} calificados y "
        f"{total - len(scores)} pendientes. {con_veto} decisión(es) con veto de Haiku.\n\n"
        f"Última decisión:\n{_texto_decision(decisiones[0], planes.get(decisiones[0]['id'], []))}"
    )


@tool(
    "schema_lab",
    "Describe las tablas de decisión/plan/outcome del dominio Lab/Trading.",
    {},
)
async def schema_lab(args: dict) -> dict:
    fuente, error = _fuente_o_error()
    origen = error or f"Fuente activa: {fuente.nombre}."
    return _texto(
        "Postgres esquema lab (sql/03_lab_schema.sql): "
        "lab.decision (id, decided_at, session_date, model, cash_usd, motivo, raw jsonb "
        "con regla, gemelo, vetos, candidatos, tendencia y plans), "
        "lab.entry_plan (id, decision_id, ticker, usd, horizon='20d', stop_usd, p_sube, "
        "enter_on), lab.outcome (id, entry_plan_id, scored_at, realized_return, hit_stop, "
        "note) y lab.price_daily (precios en la moneda de la emisora; USDMXN para el tipo "
        "de cambio). Job que escribe esto: scripts/gcp_daily_job.py (run_decide, "
        f"run_score). {origen}"
    )


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
    "la regla núcleo-satélite vigente, un resumen de lab (decisiones, planes "
    "calificados y pendientes, vetos, última decisión), el estado del libro y "
    "el sim más reciente del agente LLM.",
    {"pregunta": str},
)
async def query_lab(args: dict) -> dict:
    fuente, error = _fuente_o_error()
    partes = [_regla_vigente(), error or _leer(resumen, fuente), _estado_libro()]
    decision_llm = _decision_reciente_llm()
    if decision_llm:
        partes.append(decision_llm)
    return _texto("\n\n".join(partes))


@tool(
    "ultima_decision_lab",
    "La decisión más reciente del libro (o la última hasta una fecha): plan por "
    "nombre con USD y stop, efectivo, tendencia, candidatos, vetos de Haiku y el gemelo.",
    {
        "type": "object",
        "properties": {
            "hasta": {"type": "string", "description": "Fecha YYYY-MM-DD. Opcional: la última decisión hasta ese día."},
        },
    },
)
async def ultima_decision_lab(args: dict) -> dict:
    fuente, error = _fuente_o_error()
    hasta = str(args.get("hasta") or "").strip() or None
    return _texto(error or _leer(lambda f: ultima_decision(f, hasta), fuente))


@tool(
    "resultados_lab",
    "Cómo le fue a cada plan a 20 días hábiles (lab.outcome): retorno, si tocó el stop, "
    "ganancia en USD y el resultado del libro por decisión. Lo no calificado sale como pendiente.",
    {
        "type": "object",
        "properties": {
            "decisiones": {"type": "integer", "description": "Cuántas decisiones recientes. Opcional, 6."},
        },
    },
)
async def resultados_lab(args: dict) -> dict:
    fuente, error = _fuente_o_error()
    n = int(args.get("decisiones") or 6)
    return _texto(error or _leer(lambda f: resultados(f, n), fuente))


@tool(
    "veto_vs_gemelo_lab",
    "Libro contra gemelo: en qué decisiones el veto de Haiku cambió el libro respecto a la "
    "regla sola, cuánto aportó o restó en USD y cuáles siguen pendientes.",
    {
        "type": "object",
        "properties": {
            "decisiones": {"type": "integer", "description": "Cuántas decisiones recientes revisar. Opcional, 12."},
        },
    },
)
async def veto_vs_gemelo_lab(args: dict) -> dict:
    fuente, error = _fuente_o_error()
    n = int(args.get("decisiones") or 12)
    return _texto(error or _leer(lambda f: veto_vs_gemelo(f, n), fuente))


lab_agent = DomainAgent(
    name="lab",
    system_prompt=SYSTEM_PROMPT,
    tools=[schema_lab, query_lab, ultima_decision_lab, resultados_lab, veto_vs_gemelo_lab],
)
