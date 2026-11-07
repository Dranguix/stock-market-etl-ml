"""Base para un agente de dominio: su propio agent loop sobre Claude Agent SDK.

Cada equipo de datos (Mercado, Fundamentales, Noticias, Features/ML, Lab,
Evaluación) usa esta clase para armar SU agente: un system prompt propio +
tools propias (hoy, text-to-query stubs), acotadas a su esquema. El
orquestador nunca llama estas tools directo — invoca DomainAgent.ask(), que
corre el loop multi-turno completo contra Claude y regresa solo la
respuesta final.

El loop en sí lo corre el SDK (no hay que reimplementar el while/tool_calls
que ya existe en src/agent/investor.py): create_sdk_mcp_server() registra las
tools in-process, y query()/ClaudeAgentOptions maneja el ida-y-vuelta con el
modelo hasta que no hay más tool calls.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    TextBlock,
    create_sdk_mcp_server,
    query,
)

DEFAULT_DOMAIN_MODEL = "claude-haiku-5-5"


@dataclass
class DomainAgent:
    name: str                       # slug corto: "mercado", "fundamentales", ...
    system_prompt: str
    tools: list[Callable] = field(default_factory=list)   # funciones con @tool
    model: str = DEFAULT_DOMAIN_MODEL

    @property
    def server_name(self) -> str:
        return f"dominio-{self.name}"

    def build_options(self) -> ClaudeAgentOptions:
        server = create_sdk_mcp_server(
            name=self.server_name,
            version="0.1.0",
            tools=self.tools,
        )
        allowed = [f"mcp__{self.server_name}__{t.name}" for t in self.tools]
        return ClaudeAgentOptions(
            system_prompt=self.system_prompt,
            mcp_servers={self.server_name: server},
            allowed_tools=allowed,
            tools=[],  # sin esto, el SDK carga el preset "claude_code" completo
            # (Read/Bash/Glob/...) y el agente lee el filesystem directo,
            # saltándose sus propias tools de dominio. Solo deben existir
            # las tools MCP de este agente.
            model=self.model,
        )

    async def ask(self, pregunta: str) -> str:
        """Corre el loop completo de este agente de dominio sobre `pregunta`
        y regresa su respuesta final en texto. Agrega tus propias tools
        antes de llamar esto — ver src/agents_mcp/domains/*.py."""
        options = self.build_options()
        pieces: list[str] = []
        async for message in query(prompt=pregunta, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        pieces.append(block.text)
        return "\n".join(pieces).strip()


def stub_tool_response(nombre_tool: str, pregunta: str, donde_implementar: str) -> dict[str, Any]:
    """Respuesta estándar de un tool todavía no implementado. Bórrala cuando
    tu equipo conecte el text-to-query real."""
    return {
        "content": [{
            "type": "text",
            "text": (
                f"[STUB] {nombre_tool} todavía no está implementado. "
                f"Pregunta recibida: {pregunta!r}. "
                f"Implementar en {donde_implementar}."
            ),
        }]
    }
