"""Orquestador: su propio agent loop sobre Claude Agent SDK.

Esto es un PUNTO DE PARTIDA, no el orquestador final — está pensado para
que lo reescribas/extiendas tú. Hoy expone, como tools (una por agente de
dominio), consultar_mercado/fundamentales/noticias/features_ml/lab/
evaluacion. Cada una, al ser invocada, corre el loop COMPLETO de ese
agente de dominio (DomainAgent.ask) y regresa solo su respuesta final — el
orquestador nunca ve SQL ni datos crudos, solo texto ya sintetizado por el
agente de dominio correspondiente.

Uso:
    python -m src.agents_mcp.orchestrator "¿qué hace este sistema?"
"""
from __future__ import annotations

import asyncio
from typing import Callable

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    TextBlock,
    create_sdk_mcp_server,
    query,
    tool,
)

from src.agents_mcp.domain_agent import DomainAgent
from src.agents_mcp.domains.evaluacion import evaluacion_agent
from src.agents_mcp.domains.features_ml import features_ml_agent
from src.agents_mcp.domains.fundamentales import fundamentales_agent
from src.agents_mcp.domains.lab import lab_agent
from src.agents_mcp.domains.mercado import mercado_agent
from src.agents_mcp.domains.noticias import noticias_agent

ORCHESTRATOR_MODEL = "claude-sonnet-5-5"

SYSTEM_PROMPT = """Eres el orquestador del sistema stock-market. No
conoces los datos directo: para cada pregunta decide a qué agentes de
dominio preguntarle (mercado, fundamentales, noticias, features_ml, lab,
evaluacion), llámalos con su propia pregunta en español, y sintetiza una
respuesta con fuentes y fecha (as_of) si aplica. Nunca inventes un número
que no venga de un agente de dominio. Si dos agentes dan información
contradictoria, dilo explícitamente en vez de elegir una sola versión."""


def _domain_tool(name: str, description: str, agent: DomainAgent) -> Callable:
    """Envuelve un DomainAgent completo como una sola tool del orquestador."""

    @tool(name, description, {"pregunta": str})
    async def _handler(args: dict) -> dict:
        respuesta = await agent.ask(args.get("pregunta", ""))
        return {"content": [{"type": "text", "text": respuesta}]}

    return _handler


consultar_mercado = _domain_tool(
    "consultar_mercado",
    "Pregunta al agente de dominio Mercado (precios OHLCV, FX).",
    mercado_agent,
)
consultar_fundamentales = _domain_tool(
    "consultar_fundamentales",
    "Pregunta al agente de dominio Fundamentales (márgenes, UPA, dividendos, PIT).",
    fundamentales_agent,
)
consultar_noticias = _domain_tool(
    "consultar_noticias",
    "Pregunta al agente de dominio Noticias (titulares, sentimiento por ticker).",
    noticias_agent,
)
consultar_features_ml = _domain_tool(
    "consultar_features_ml",
    "Pregunta al agente de dominio Features/ML (bitácora de modelos entrenados).",
    features_ml_agent,
)
consultar_lab = _domain_tool(
    "consultar_lab",
    "Pregunta al agente de dominio Lab/Trading (decisión, plan, veto, gemelo, outcome).",
    lab_agent,
)
consultar_evaluacion = _domain_tool(
    "consultar_evaluacion",
    "Pregunta al agente de dominio Evaluación/Riesgo (backtests, Deflated Sharpe).",
    evaluacion_agent,
)

_ALL_TOOLS = [
    consultar_mercado,
    consultar_fundamentales,
    consultar_noticias,
    consultar_features_ml,
    consultar_lab,
    consultar_evaluacion,
]

_SERVER_NAME = "orquestador"


def build_options() -> ClaudeAgentOptions:
    server = create_sdk_mcp_server(name=_SERVER_NAME, version="0.1.0", tools=_ALL_TOOLS)
    allowed = [f"mcp__{_SERVER_NAME}__{t.name}" for t in _ALL_TOOLS]
    return ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        mcp_servers={_SERVER_NAME: server},
        allowed_tools=allowed,
        tools=[],  # igual que en DomainAgent: sin esto el orquestador
        # tendría Read/Bash propios y se saltaría a los agentes de dominio.
        model=ORCHESTRATOR_MODEL,
    )


async def preguntar(pregunta: str) -> str:
    """El loop completo del orquestador sobre una pregunta libre."""
    options = build_options()
    pieces: list[str] = []
    async for message in query(prompt=pregunta, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    pieces.append(block.text)
    return "\n".join(pieces).strip()


def main() -> None:
    import sys

    pregunta = " ".join(sys.argv[1:]) or "¿Qué hace este sistema, en una frase?"
    print(asyncio.run(preguntar(pregunta)))


if __name__ == "__main__":
    main()
