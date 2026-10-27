"""Registro de agentes del equipo Features/ML (equipo Rosa).

Es la única puerta del agente de trading: lo que no esté en FEATURES_ML
no lo puede consultar.

Contrato para agregar un agente:
  1. Subclase de ObservationAgent (src/agent/base.py), constructor sin argumentos.
  2. name único y en minúsculas; description en una frase, es lo que lee Claude.
  3. observe(**params) devuelve un dict con "ok". Si ok es False, trae "error".
     Cada nombre que se pueda comprar va en una llave "ticker" con el nombre
     corto del universo (WALMEX, SLV). Solo esos nombres puede comprar el trading.
  4. tool_schema() con _schema(...) de src/agent/tools.py.
  5. Agregar la clase a FEATURES_ML.
"""
from __future__ import annotations

from src.agent.base import ObservationAgent
from src.agent.tools import AgenteRanking, AgenteWoe

FEATURES_ML: list[type[ObservationAgent]] = [
    AgenteRanking,
    AgenteWoe,
]


def build_features_ml_tools() -> dict[str, ObservationAgent]:
    agents = [cls() for cls in FEATURES_ML]
    out: dict[str, ObservationAgent] = {}
    for agent in agents:
        if agent.name in out:
            raise ValueError(f"agente de Features/ML repetido: {agent.name}")
        out[agent.name] = agent
    return out
