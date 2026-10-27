"""Agente de trading (equipo Trading Agent): solo consulta a los agentes de Features/ML.

Tres candados:
  1. Claude solo ve los schemas de FEATURES_ML más proponer_asignacion.
  2. dispatch rechaza cualquier nombre que no esté en FEATURES_ML.
  3. El guard rechaza tickers que ningún agente de Features/ML devolvió en esta
     corrida, salvo los que ya están en el libro.

El libro es estado propio, no una consulta: va en el mensaje del usuario.
"""
from __future__ import annotations

import json
from datetime import date
from typing import Any

from src.agent.book import BUDGET_USD, Book
from src.agent.features_ml import build_features_ml_tools
from src.agent.investor import AgenteInversor
from src.agent.policy import MAX_NAMES, MAX_TICKET, MIN_HOLD_DAYS, MIN_TICKET, PoliticaInversor


class PoliticaTrading(PoliticaInversor):
    name = "trading_features_ml"

    def __init__(self) -> None:
        self.vistos: set[str] = set()
        self.agentes: list[str] = []

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
- No cambies el tamaño de una posición con menos de {MIN_HOLD_DAYS} días, salvo que el precio haya tocado su stop.
- Solo compras nombres que te devolvió algún agente de Features/ML en esta conversación,
  o que ya están en el libro.

Cómo decidir:
1. Consulta a los agentes de Features/ML que necesites.
2. Si la evidencia es floja o los agentes fallan, deja el dinero en efectivo.
3. Cierra con proponer_asignacion y explica en motivo qué agente te dio cada razón.
"""

    def user_message(self, book: Book) -> str:
        libro = json.dumps(book.snapshot(), ensure_ascii=False)
        return (
            f"Hoy es {date.today().isoformat()}, después del cierre de la BMV. "
            f"Este es el libro: {libro}. "
            "Consulta a los agentes de Features/ML y termina con proponer_asignacion."
        )

    def guard(self, proposal: dict, book: Book, prices_usd: dict[str, float] | None = None) -> dict:
        out = super().guard(proposal, book, prices_usd)
        if not out.get("ok"):
            return out
        permitidos = self.vistos | set(book.positions)
        fuera = [r["ticker"] for r in out["allocations"] if r["ticker"] not in permitidos]
        if fuera:
            return {
                "ok": False,
                "error": (
                    f"{', '.join(fuera)} no salió de ningún agente de Features/ML. "
                    "Solo puedes comprar nombres que te devolvieron ellos."
                ),
            }
        return out


class AgenteTrading(AgenteInversor):
    name = "trading"

    def __init__(self, book: Book, model: str | None = None):
        super().__init__(book, PoliticaTrading(), model)
        self.tools = build_features_ml_tools()
        self.policy.agentes = list(self.tools)

    def dispatch(self, name: str, params: dict | None) -> dict:
        if name not in self.tools:
            return {
                "ok": False,
                "error": f"{name} no es un agente de Features/ML. Solo puedes consultar: {', '.join(self.tools)}",
            }
        out = super().dispatch(name, params)
        if out.get("ok"):
            self.policy.vistos |= tickers_en(out)
        return out

    def act(self, observation: dict | None = None, apply: bool = False) -> dict:
        self.policy.vistos = set()
        result = super().act(observation, apply)
        result["agentes_features_ml"] = list(self.tools)
        result["tickers_vistos"] = sorted(self.policy.vistos)
        return result


def tickers_en(payload: Any) -> set[str]:
    """Todos los valores de la llave "ticker", a cualquier profundidad."""
    out: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "ticker" and isinstance(value, str) and value.strip():
                out.add(value.upper().strip())
            else:
                out |= tickers_en(value)
    elif isinstance(payload, list):
        for item in payload:
            out |= tickers_en(item)
    return out
