"""Servidor MCP standalone (protocolo MCP oficial, HTTP) para stock-market.

Expone, vía el protocolo MCP real (no el `create_sdk_mcp_server` in-process
que usa `src/agents_mcp/orchestrator.py`), una tool por cada agente de
dominio ya existente en `src/agents_mcp/domains/`. Pensado para que una app
de Claude (claude.ai / Claude Desktop) se conecte por URL como "custom
connector" y le pregunte en lenguaje natural sobre el proyecto.

NOTA DE VERSIÓN DEL SDK `mcp`: el repo trae instalado `mcp==2.3.0` (viene
como dependencia de `claude-agent-sdk`, que ya usan los agentes in-process).
En esa serie, `mcp.server.fastmcp.FastMCP` fue renombrado a
`mcp.server.mcpserver.MCPServer` — ver
https://py.sdk.modelcontextprotocol.io/v2/migration/#fastmcp-renamed-to-mcpserver.
La API que nos importa (decorador `@mcp.tool()`, `mcp.streamable_http_app()`,
`mcp.run(transport=...)`) es equivalente, así que este archivo usa
`MCPServer` en vez de `FastMCP`. Si se hace downgrade a `mcp<2` en el venv,
cambia el import de vuelta a `from mcp.server.fastmcp import FastMCP` y
`mcp = FastMCP("stock-market")` — el resto del archivo no cambia.

Cada tool hace el import de su `_agent` DENTRO de la función (import
perezoso): si un dominio tiene un error de import (p.ej. porque otro
agente está reemplazando sus tools STUB por datos reales en paralelo),
eso tumba solo esa tool al invocarse, no el servidor completo.
"""
from __future__ import annotations

import os

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("stock-market")

# Disclaimer obligatorio en TODA respuesta que toque dinero/decisión — no es
# un README aparte, va pegado al output que de verdad lee quien pregunta.
# Ver investigación de la sesión: "publisher's exclusion" (SEC) y la
# distinción recomendación-general-vs-personalizada (CNBV) solo protegen si
# el límite es visible en cada respuesta, no solo documentado.
DISCLAIMER = (
    "\n\n---\n"
    "⚠️ Herramienta educativa/informativa, NO es asesoría de inversión "
    "personalizada ni gestión de terceros. No hay garantía de resultados "
    "— varios de estos modelos no tienen ventaja estadística probada (ver "
    "agente Evaluación). Nadie aquí es asesor financiero registrado "
    "(CNBV/SEC). La decisión y el riesgo son tuyos; considera consultar a "
    "un profesional antes de invertir dinero real."
)


def _con_disclaimer(texto: str) -> str:
    return f"{texto}{DISCLAIMER}"


@mcp.tool()
async def preguntar_mercado(pregunta: str) -> str:
    """Precios OHLCV, FX — pregunta en lenguaje natural sobre datos de mercado."""
    from src.agents_mcp.domains.mercado import mercado_agent

    return _con_disclaimer(await mercado_agent.ask(pregunta))


@mcp.tool()
async def preguntar_fundamentales(pregunta: str) -> str:
    """Estados financieros, ratios, reportes — pregunta en lenguaje natural sobre fundamentales."""
    from src.agents_mcp.domains.fundamentales import fundamentales_agent

    return _con_disclaimer(await fundamentales_agent.ask(pregunta))


@mcp.tool()
async def preguntar_noticias(pregunta: str) -> str:
    """Noticias y sentimiento de mercado — pregunta en lenguaje natural sobre noticias."""
    from src.agents_mcp.domains.noticias import noticias_agent

    return _con_disclaimer(await noticias_agent.ask(pregunta))


@mcp.tool()
async def preguntar_features_ml(pregunta: str) -> str:
    """Features, modelos ML, pronóstico y recomendación de portafolio —
    pregunta en lenguaje natural sobre ese dominio."""
    # TODO: portafolio — si/cuando `features_ml_agent` exponga una tool
    # dedicada de recomendación de portafolio (hoy solo hay
    # schema_features_ml / query_features_ml como STUBs en
    # src/agents_mcp/domains/features_ml.py), esta misma tool la cubre
    # porque DomainAgent.ask() decide internamente qué tool usar.
    from src.agents_mcp.domains.features_ml import features_ml_agent

    return _con_disclaimer(await features_ml_agent.ask(pregunta))


@mcp.tool()
async def preguntar_lab(pregunta: str) -> str:
    """Experimentos / lab (esquema Postgres de lab) — pregunta en lenguaje natural."""
    from src.agents_mcp.domains.lab import lab_agent

    return _con_disclaimer(await lab_agent.ask(pregunta))


@mcp.tool()
async def preguntar_evaluacion(pregunta: str) -> str:
    """Evaluación de modelos / métricas A-B — pregunta en lenguaje natural."""
    from src.agents_mcp.domains.evaluacion import evaluacion_agent

    return _con_disclaimer(await evaluacion_agent.ask(pregunta))


@mcp.tool()
async def vision_360(ticker: str) -> str:
    """Visión 360 de un ticker: un agente 'quant senior' corre, EN ORDEN FIJO
    (no lo decide el LLM), búsqueda (mercado+noticias+fundamentales) -> ML
    (pronóstico) -> optimización (rol en el portafolio óptimo) -> síntesis
    final citando la fuente de cada afirmación. Usa esta tool cuando pidan
    un análisis completo de un ticker, no solo un dato suelto."""
    from src.agents_mcp.quant_senior import vision_360 as _vision_360

    return _con_disclaimer(await _vision_360(ticker))


def build_app():
    """Construye la app ASGI (Starlette) del transporte streamable-http,
    con el middleware de autenticación por Bearer token ya montado."""
    from mcp_server.auth import BearerTokenAuthMiddleware

    host = os.environ.get("HOST", "0.0.0.0")
    app = mcp.streamable_http_app(host=host)
    app.add_middleware(BearerTokenAuthMiddleware)
    return app


def main() -> None:
    import asyncio

    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8080"))

    app = build_app()
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    asyncio.run(server.serve())


if __name__ == "__main__":
    main()
