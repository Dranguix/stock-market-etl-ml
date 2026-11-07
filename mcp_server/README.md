# MCP server standalone — stock-market

Servidor MCP real (protocolo oficial Model Context Protocol, transporte
`streamable-http`) que envuelve los agentes de dominio ya existentes en
`src/agents_mcp/domains/` para que una app de Claude (claude.ai / Claude
Desktop, como "custom connector") se conecte por URL y les haga preguntas
en lenguaje natural sobre el proyecto stock-market.

Esto es distinto del sistema multiagente in-process
(`src/agents_mcp/orchestrator.py` + `create_sdk_mcp_server`): ese corre
dentro de un proceso Python que ya está llamando a Claude. Este servidor
expone las mismas capacidades vía HTTP, para que cualquier cliente MCP
(incluido claude.ai) se conecte desde fuera.

## Tools expuestas

Una tool por dominio, cada una recibe `pregunta: str` en lenguaje natural:

- `preguntar_mercado` — precios OHLCV, FX.
- `preguntar_fundamentales` — estados financieros, ratios, reportes.
- `preguntar_noticias` — noticias y sentimiento de mercado.
- `preguntar_features_ml` — features, modelos ML, pronóstico y portafolio
  (pronóstico/portafolio dependen de que el dominio ya tenga esas tools —
  ver TODO en `server.py`).
- `preguntar_lab` — experimentos / esquema Postgres de lab.
- `preguntar_evaluacion` — evaluación de modelos / métricas A-B.

Cada tool importa su `_agent` de forma perezosa (dentro de la función), así
que si un dominio tiene un error de import, solo falla esa tool al
invocarse — no tumba el servidor completo.

## Correr local

```bash
cd /Users/rod/Projects/stock-market
source venv/bin/activate
export MCP_AUTH_TOKEN="un-secreto-cualquiera-para-pruebas"
python -m mcp_server.server
```

Por default escucha en `0.0.0.0:${PORT:-8080}` (8080 = default de Cloud Run).
El endpoint MCP queda en `http://localhost:8080/mcp`.

## Autenticación

Todas las requests (salvo `/health` y `/healthz`) deben incluir:

```
Authorization: Bearer <MCP_AUTH_TOKEN>
```

El token correcto viene de la variable de entorno `MCP_AUTH_TOKEN`:

- Si **no está seteada**: el servidor arranca igual, pero usa un token
  default claramente marcado como inseguro
  (`INSECURE-DEFAULT-TOKEN-CAMBIAME`, ver `mcp_server/auth.py`) e imprime un
  `WARNING` en stderr. Sirve para no trabar pruebas rápidas locales.
- Para forzar que el servidor se niegue a arrancar sin un token real
  (recomendado antes de exponerlo fuera de tu laptop), exporta además
  `MCP_REQUIRE_AUTH_TOKEN=1`.

Una request sin el header correcto recibe `401 Unauthorized` antes de
llegar al protocolo MCP (el middleware vive en `mcp_server/auth.py` y se
monta sobre la app ASGI de Starlette que regresa
`mcp.streamable_http_app()`).

## Probar manualmente

El protocolo MCP sobre `streamable-http` usa JSON-RPC 2.0 y espera
`Accept: application/json, text/event-stream`, así que un `curl` plano con
un body cualquiera no basta para una sesión MCP completa, pero sirve para
confirmar que la autenticación responde:

```bash
# Sin token -> 401
curl -i http://localhost:8080/mcp

# Con token correcto -> pasa la autenticación (puede dar 400 si el body
# no es un JSON-RPC válido, pero ya no es 401)
curl -i -X POST http://localhost:8080/mcp \
  -H "Authorization: Bearer un-secreto-cualquiera-para-pruebas" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "curl-test", "version": "0"}}}'
```

Para una prueba real end-to-end (listar tools, llamarlas, etc.), usa un
cliente MCP de verdad en vez de curl plano:

```bash
# Inspector interactivo del SDK oficial de mcp
mcp dev mcp_server/server.py
```

o configura este servidor como "custom connector" en Claude Desktop /
claude.ai apuntando a `http://<host>:<port>/mcp` con el header
`Authorization: Bearer <token>`.

## Nota sobre la versión del SDK `mcp`

El repo trae instalado `mcp==2.3.0` (viene como dependencia de
`claude-agent-sdk`, que ya usan los agentes in-process). En esa serie,
`mcp.server.fastmcp.FastMCP` fue renombrado a
`mcp.server.mcpserver.MCPServer`. `mcp_server/server.py` usa `MCPServer`
por eso — la API relevante (`@mcp.tool()`, `mcp.streamable_http_app()`,
`mcp.run(...)`) es equivalente a la de `FastMCP` en `mcp<2`. Si el venv
baja de versión a `mcp<2`, basta cambiar el import de vuelta a
`from mcp.server.fastmcp import FastMCP`.
