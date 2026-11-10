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
import os
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
_DUMMY_DIR = ROOT / "data" / "lab" / "dummy"  # Ejemplos compartidos: NO modificar

SUBDOMINIO = "trading"


def _next_id(path: Path) -> int:
    """Busca el mayor id de un JSONL y devuelve el siguiente."""
    if not path.exists():
        return 1
    ultimo = 0
    for linea in path.read_text(encoding="utf-8").splitlines():
        if not linea.strip():
            continue
        try:
            ultimo = max(ultimo, int(json.loads(linea).get("id", 0)))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return ultimo + 1


def _destino_lab_dummy() -> Path | None:
    """Solo permite escribir si el usuario lo habilitó expresamente.

    LAB_FUENTE=dummy, LAB_DUMMY_WRITE=1 y LAB_DUMMY_DIR deben configurarse
    juntos. Nunca se escribe en data/lab/dummy, que contiene fixtures del equipo.
    """
    if (os.getenv("LAB_FUENTE") or "").lower().strip() != "dummy":
        return None
    if os.getenv("LAB_DUMMY_WRITE") != "1":
        return None
    valor = (os.getenv("LAB_DUMMY_DIR") or "").strip()
    if not valor:
        raise ValueError("Falta LAB_DUMMY_DIR: elige una carpeta separada para las pruebas")
    carpeta = Path(valor).expanduser().resolve()
    if carpeta == _DUMMY_DIR.resolve():
        raise ValueError("No se permite escribir en data/lab/dummy: usa una carpeta de pruebas")
    return carpeta


def _append_to_lab_dummy(validada: dict, resultado: dict, carpeta: Path, hoy: date) -> dict:
    """Registra la asignación en JSONL, con columnas del esquema lab.

    Es una simulación local. No se comunica con Postgres ni con Anthropic.
    """
    carpeta.mkdir(parents=True, exist_ok=True)
    dec_path = carpeta / "decision.jsonl"
    plan_path = carpeta / "entry_plan.jsonl"
    dec_id = _next_id(dec_path)
    plan_id = _next_id(plan_path)
    allocations = validada.get("allocations") or []
    cash = validada.get("cash_usd")
    if cash is None:
        cash = BUDGET_USD - sum(float(a["usd"]) for a in allocations)
    decision_row = {
        "id": dec_id,
        "decided_at": datetime.now(timezone.utc).isoformat(),
        "session_date": hoy.isoformat(),
        "model": "trading_agent_local_dummy",
        "cash_usd": float(cash),
        "motivo": str(validada.get("motivo") or ""),
        "raw": {
            "tipo": "simulacion_local_no_produccion",
            "allocations": allocations,
            "book_snapshot": resultado.get("libro"),
        },
    }
    planes = []
    for alloc in allocations:
        usd = float(alloc.get("usd", 0))
        if usd <= 0:
            continue
        planes.append({
            "id": plan_id,
            "decision_id": dec_id,
            "ticker": str(alloc["ticker"]),
            "usd": usd,
            "horizon": str(alloc.get("horizon") or "20d"),
            "stop_usd": None,  # Trading no calcula stops
            "p_sube": None,    # No inventar probabilidades
            "enter_on": hoy.isoformat(),
        })
        plan_id += 1

    # Ambos ficheros se preparan antes de escribir. Son datos sandbox locales;
    # no es una transacción entre archivos, así que no usar para producción.
    with dec_path.open("a", encoding="utf-8") as archivo:
        archivo.write(json.dumps(decision_row, ensure_ascii=False) + "\n")
    with plan_path.open("a", encoding="utf-8") as archivo:
        for plan in planes:
            archivo.write(json.dumps(plan, ensure_ascii=False) + "\n")
    return {"decision_id": dec_id, "planes": len(planes), "carpeta": str(carpeta)}

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
- Solo compras nombres declarados en comprables de un ranking de esta conversación.
  Un ticker que solo esté en el top informativo, que solo pasó por otro agente o que
  esté en la lista evitar no se puede comprar.
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
            return (
                f"{ticker} no está en comprables de un ranking de Features/ML "
                f"(solo se autorizan apuestas LONG activas del top). No se puede comprar."
            )
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
        """Valida, aplica al libro y, solo si se autoriza, registra en sandbox dummy."""
        try:
            carpeta = _destino_lab_dummy()
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        prices = _prices_for(asignacion)
        validada = self.policy.guard(asignacion, self.book, prices)
        if not validada.get("ok"):
            return validada
        try:
            resultado = {"ok": True, **self.book.apply(validada, prices)}
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        if carpeta is None:
            resultado["lab_dummy"] = {"estado": "omitido", "motivo": "escritura dummy no habilitada"}
            return resultado
        try:
            registrado = _append_to_lab_dummy(validada, resultado, carpeta, self.policy.hoy)
            resultado["lab_dummy"] = {"estado": "registrado", **registrado}
        except (OSError, TypeError, ValueError) as exc:
            # El libro ya fue aplicado: informar explícitamente si Lab falló.
            resultado["lab_dummy"] = {"estado": "error", "detalle": str(exc)}
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
