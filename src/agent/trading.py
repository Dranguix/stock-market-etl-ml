"""Agente de trading (equipo Trading Agent): solo consulta a los agentes de Features/ML.

Tres candados:
  1. Claude solo ve los schemas de FEATURES_ML más proponer_asignacion.
  2. dispatch rechaza cualquier nombre que no esté en FEATURES_ML.
  3. El guard solo deja comprar lo que un agente de Features/ML declaró comprable
     en esta corrida, con señal reciente y fuera de su lista evitar.

El libro es estado propio, no una consulta: va en el mensaje del usuario.

Entrada del orquestador: consultar(pregunta). Nunca escribe el libro.
Escribir es otra acción: AgenteTrading.aplicar(asignacion).

Este agente no maneja stops: ver_salida no es de Features/ML. No hay stop_usd
en el schema, el que llegue se descarta y no hay excepción a los días mínimos.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.agent.book import BUDGET_USD, Book
from src.agent.chat import function_tool
from src.agent.features_ml import build_features_ml_tools
from src.agent.investor import AgenteInversor, _prices_for
from src.agent.policy import MAX_NAMES, MAX_TICKET, MIN_HOLD_DAYS, MIN_TICKET, TOL, PoliticaInversor
from src.agent.tools import asignacion_schema

ROOT = Path(__file__).resolve().parents[2]
BOOK_PATH = ROOT / "data" / "agent" / "trading_book.json"
_DUMMY_DIR = ROOT / "data" / "lab" / "dummy"

SUBDOMINIO = "trading"


def _next_id(path: Path) -> int:
    """Devuelve el próximo ID disponible leyendo la última línea del JSONL."""
    if not path.exists():
        return 1
    last_id = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            last_id = max(last_id, int(json.loads(line).get("id", 0)))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return last_id + 1


def _append_to_lab_dummy(validada: dict, resultado: dict) -> None:
    """Escribe una decisión y sus entry_plans en los archivos dummy (JSONL).

    Nunca lanza excepciones: se llama dentro de un try/except en aplicar().
    Solo escribe si LAB_FUENTE != 'postgres' (es decir, en modo dummy/offline).
    """
    import os
    if (os.environ.get("LAB_FUENTE") or "").strip().lower() == "postgres":
        return  # No tocar el dummy cuando hay Postgres configurado

    _DUMMY_DIR.mkdir(parents=True, exist_ok=True)
    dec_path = _DUMMY_DIR / "decision.jsonl"
    plan_path = _DUMMY_DIR / "entry_plan.jsonl"

    from datetime import datetime, timezone
    ahora = datetime.now(timezone.utc).isoformat()
    hoy = date.today().isoformat()

    # --- decision ---
    dec_id = _next_id(dec_path)
    allocations = validada.get("allocations") or []
    cash = validada.get("cash_usd")
    if cash is None:
        total_inv = sum(float(a.get("usd", 0)) for a in allocations)
        cash = max(0.0, BUDGET_USD - total_inv)
    decision_row = {
        "id": dec_id,
        "decided_at": ahora,
        "session_date": hoy,
        "model": "trading_agent_local",
        "cash_usd": float(cash),
        "motivo": str(validada.get("motivo") or ""),
        "raw": {
            "allocations": allocations,
            "book_snapshot": resultado.get("book"),
        },
    }
    with dec_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(decision_row, ensure_ascii=False) + "\n")

    # --- entry_plan (una fila por ticker con usd > 0) ---
    plan_id = _next_id(plan_path)
    for alloc in allocations:
        usd = float(alloc.get("usd", 0))
        if usd <= 0:
            continue
        plan_row = {
            "id": plan_id,
            "decision_id": dec_id,
            "ticker": str(alloc.get("ticker", "")),
            "usd": usd,
            "horizon": str(alloc.get("horizon") or ""),
            "stop_usd": alloc.get("stop_usd"),
            "p_sube": alloc.get("p_sube"),
            "enter_on": hoy,
        }
        with plan_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(plan_row, ensure_ascii=False) + "\n")
        plan_id += 1
MAX_DIAS_SENAL = 3  # días hábiles lun-vie, sin calendario de asuetos BMV
PREGUNTA_REPARTO = "Reparte el libro y termina con proponer_asignacion."


def dias_habiles(desde: str | date, hasta: date) -> int | None:
    """Días hábiles entre dos fechas. None si la fecha no se puede leer."""
    try:
        inicio = date.fromisoformat(str(desde)[:10])
    except ValueError:
        return None
    return int(np.busday_count(inicio, hasta))


class PoliticaTrading(PoliticaInversor):
    name = "trading_features_ml"
    respeta_stop = False

    def __init__(self, hoy: date | None = None) -> None:
        self.hoy = hoy or date.today()
        self.vistos: dict[str, str | None] = {}
        self.evitados: set[str] = set()
        self.agentes: list[str] = []

    def reiniciar(self) -> None:
        self.vistos = {}
        self.evitados = set()

    def system_prompt(self) -> str:
        agentes = ", ".join(self.agentes) or "ninguno"
        return f"""Eres el agente de trading de un libro de paper trading.
El presupuesto es uno solo: {BUDGET_USD:.0f} dólares. Tú repartes esa cantidad.

Solo puedes consultar a los agentes del equipo Features/ML: {agentes}.
No tienes otra fuente. Si un dato no viene de ellos, no lo tienes.

Reglas que el sistema va a rechazar si las rompes:
- Long only. El efectivo es una posición válida.
- Máximo {MAX_NAMES} nombres con dólares > 0.
- Cada nombre lleva 0, o entre {MIN_TICKET:.0f} y {MAX_TICKET:.0f} dólares.
- La suma de los dólares de los nombres más el efectivo es exactamente {BUDGET_USD:.0f}.
- No cambies el tamaño de una posición con menos de {MIN_HOLD_DAYS} días.
- Solo compras nombres del top de un ranking de esta conversación. Un ticker que
  solo pasó por otro agente no se puede comprar, y uno de la lista evitar tampoco.
- La señal no puede tener más de {MAX_DIAS_SENAL} días hábiles. Compara fecha_senal con
  la fecha de hoy: si es más vieja, no compres.
- Lo que ya está en el libro se puede mantener o reducir.

Cómo decidir:
1. Consulta a los agentes de Features/ML que necesites.
2. Si la evidencia es floja, los agentes fallan o la señal es vieja, deja el dinero en efectivo.
3. Si te piden repartir el libro, cierra con proponer_asignacion y explica en motivo
   qué agente te dio cada razón. Si es otra pregunta, contesta con texto.
"""

    def user_message(self, book: Book, pregunta: str = PREGUNTA_REPARTO) -> str:
        libro = json.dumps(book.snapshot(), ensure_ascii=False)
        return (
            f"Hoy es {self.hoy.isoformat()}, después del cierre de la BMV. "
            f"Este es el libro: {libro}. "
            f"{pregunta}"
        )

    def guard(self, proposal: dict, book: Book, prices_usd: dict[str, float] | None = None) -> dict:
        out = super().guard(proposal, book, prices_usd)
        if not out.get("ok"):
            return out
        for row in out["allocations"]:
            row["stop_usd"] = None
            error = self._candado(row, book)
            if error:
                return {"ok": False, "error": error}
        return out

    def _candado(self, row: dict, book: Book) -> str | None:
        ticker = row["ticker"]
        pos = book.positions.get(ticker)
        if pos and row["usd"] <= float(pos["usd"]) + TOL:
            return None
        if ticker in self.evitados:
            return f"{ticker} está en la lista evitar. No se puede comprar."
        if ticker not in self.vistos:
            return f"{ticker} no salió del top de un ranking de Features/ML. Solo puedes comprar esos nombres."
        fecha = self.vistos[ticker]
        edad = dias_habiles(fecha, self.hoy) if fecha else None
        if edad is None or edad < 0:
            return f"La señal de {ticker} no trae una fecha válida ({fecha}). Deja ese dinero en efectivo."
        if edad > MAX_DIAS_SENAL:
            return (
                f"La señal de {ticker} es del {fecha}: tiene {edad} días hábiles y el límite "
                f"es {MAX_DIAS_SENAL}. Deja ese dinero en efectivo."
            )
        return None


class AgenteTrading(AgenteInversor):
    name = "trading"

    def __init__(self, book: Book, model: str | None = None, hoy: date | None = None):
        super().__init__(book, PoliticaTrading(hoy), model)
        self.tools = build_features_ml_tools()
        self.policy.agentes = list(self.tools)
        self.consultados: list[str] = []

    def observe(self, pregunta: str = PREGUNTA_REPARTO, **params) -> dict:
        out = super().observe(**params)
        schemas = [a.tool_schema() for a in self.tools.values()]
        schemas.append(_asignacion_sin_stop())
        out["tools"] = [function_tool(schema) for schema in schemas]
        out["user"] = self.policy.user_message(self.book, pregunta)
        return out

    def dispatch(self, name: str, params: dict | None) -> dict:
        agent = self.tools.get(name)
        if agent is None:
            return {
                "ok": False,
                "error": f"{name} no es un agente de Features/ML. Solo puedes consultar: {', '.join(self.tools)}",
            }
        if name not in self.consultados:
            self.consultados.append(name)
        out = super().dispatch(name, params)
        if out.get("ok"):
            fecha = out.get("fecha_senal")
            for ticker in agent.comprables(out):
                self.policy.vistos[ticker.upper().strip()] = fecha
            self.policy.evitados |= {t.upper().strip() for t in agent.evitados(out)}
        return out

    def act(self, observation: dict | None = None) -> dict:
        """Una corrida. Solo propone: escribir el libro es aplicar()."""
        self.policy.reiniciar()
        self.consultados = []
        return super().act(observation, apply=False)

    def aplicar(self, asignacion: dict) -> dict:
        """Acción explícita: vuelve a pasar el guard de esta corrida y escribe el libro.

        Después de aplicar exitosamente, registra la decisión en data/lab/dummy/
        (solo en modo offline/dummy, nunca consume tokens ni conecta a Postgres).
        """
        prices = _prices_for(asignacion)
        validada = self.policy.guard(asignacion, self.book, prices)
        if not validada.get("ok"):
            return validada
        try:
            resultado = {"ok": True, **self.book.apply(validada, prices)}
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        # Registrar en lab dummy (nunca falla el flujo principal)
        try:
            _append_to_lab_dummy(validada, resultado)
        except Exception:
            pass
        return resultado


def consultar(pregunta: str, agente: AgenteTrading | None = None) -> dict:
    """Entrada del orquestador: una pregunta y un dict serializable a JSON. No escribe."""
    if not str(pregunta or "").strip():
        return _respuesta(error="falta la pregunta")
    try:
        agente = agente or AgenteTrading(Book(BOOK_PATH))
        result = agente.act(agente.observe(pregunta=pregunta))
    except Exception as exc:
        return _respuesta(agente, error=f"trading fallo: {exc}")
    if result.get("error"):
        return _respuesta(agente, error=result["error"])

    ultimo = (result.get("transcript") or [{}])[-1]
    texto = (ultimo.get("message") or {}).get("content")
    asignacion = result.get("asignacion")
    if asignacion is None and (not texto or ultimo.get("finish_reason") == "tool_calls"):
        return _respuesta(agente, texto, error=f"sin respuesta final en {result.get('turnos')} turnos")
    return _respuesta(agente, texto, asignacion)


def _respuesta(
    agente: AgenteTrading | None = None,
    respuesta: str | None = None,
    asignacion: dict | None = None,
    error: str | None = None,
) -> dict:
    return {
        "ok": error is None,
        "respuesta": respuesta,
        "asignacion": asignacion,
        "agentes_consultados": list(agente.consultados) if agente else [],
        "error": error,
    }


def describir_agentes() -> list[dict]:
    return [{
        "name": AgenteTrading.name,
        "description": "Reparte el libro de 50 dólares consultando solo a Features/ML. Preguntar no escribe el libro.",
        "input_schema": {
            "type": "object",
            "required": ["pregunta"],
            "properties": {"pregunta": {"type": "string"}},
        },
    }]


def _asignacion_sin_stop() -> dict[str, Any]:
    schema = asignacion_schema()
    schema["input_schema"]["properties"]["allocations"]["items"]["properties"].pop("stop_usd")
    return schema
