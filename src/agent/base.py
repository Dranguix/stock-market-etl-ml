"""Clase base observe / act.

Los subagentes de observación solo implementan observe().
El agente inversor es el único que llama al LLM en act().
"""
from __future__ import annotations

from typing import Any


class Agent:
    name: str = "agent"
    description: str = ""

    def observe(self, **params: Any) -> dict:
        raise NotImplementedError

    def act(self, observation: dict | None = None) -> dict:
        raise NotImplementedError


class ObservationAgent(Agent):
    """Empleado que escribe una parte de la nota. No llama al LLM."""

    def act(self, observation: dict | None = None) -> dict:
        raise RuntimeError(f"{self.name} solo observa")

    def tool_schema(self) -> dict:
        raise NotImplementedError

    def comprables(self, resultado: dict) -> list[str]:
        """Tickers que este resultado autoriza a comprar. Por defecto, ninguno."""
        return []

    def evitados(self, resultado: dict) -> list[str]:
        """Tickers que este resultado prohíbe comprar. Por defecto, ninguno."""
        return []
