"""Middleware ASGI de autenticación por token secreto (Bearer) para el
servidor MCP standalone.

NOTA DE VERSIÓN: el paquete `mcp` instalado en este repo es la serie 2.x,
donde `FastMCP` fue renombrado a `MCPServer` (`mcp.server.mcpserver`). La
API relevante para nosotros no cambió de forma: `MCPServer.streamable_http_app()`
sigue regresando una app ASGI de Starlette, así que seguimos pudiendo
envolverla con un middleware ASGI normal via `app.add_middleware(...)`
o construyendo el stack a mano como se hace aquí. Si en el futuro se
actualiza `mcp` y este mecanismo cambia otra vez, ajusta este archivo y
deja un comentario explicando qué cambió.

Cómo se valida el token:
- Se lee la variable de entorno MCP_AUTH_TOKEN al arrancar el proceso.
- Si no está seteada, usamos un valor default **claramente inseguro**
  (ver INSECURE_DEFAULT_TOKEN) para que pruebas locales rápidas no truenen,
  pero imprimimos un warning fuerte. En un entorno real (producción) hay
  que exportar MCP_AUTH_TOKEN con un secreto real — si quieres que el
  servidor se niegue a arrancar sin el token, exporta
  MCP_REQUIRE_AUTH_TOKEN=1 y el proceso lanzará RuntimeError al importar
  este módulo si MCP_AUTH_TOKEN no está presente.
- Cada request HTTP al servidor MCP debe incluir el header
  `Authorization: Bearer <token>` con el token correcto, o se responde
  401 antes de que la petición llegue al protocolo MCP.
"""
from __future__ import annotations

import os
import sys

from starlette.types import ASGIApp, Receive, Scope, Send

# Valor default INSEGURO, solo para que `python -m mcp_server.server` no
# truene en una laptop sin configurar nada. NO USAR EN PRODUCCIÓN.
INSECURE_DEFAULT_TOKEN = "INSECURE-DEFAULT-TOKEN-CAMBIAME"


def _load_auth_token() -> str:
    token = os.environ.get("MCP_AUTH_TOKEN")
    if token:
        return token

    if os.environ.get("MCP_REQUIRE_AUTH_TOKEN") == "1":
        raise RuntimeError(
            "MCP_AUTH_TOKEN no está seteada y MCP_REQUIRE_AUTH_TOKEN=1. "
            "Exporta MCP_AUTH_TOKEN con un secreto real antes de arrancar "
            "este servidor en un entorno que no sea tu laptop."
        )

    print(
        "WARNING: MCP_AUTH_TOKEN no está seteada. Usando un token default "
        "INSEGURO solo apto para pruebas locales. Exporta MCP_AUTH_TOKEN "
        "antes de exponer este servidor fuera de tu máquina.",
        file=sys.stderr,
    )
    return INSECURE_DEFAULT_TOKEN


AUTH_TOKEN = _load_auth_token()

# Rutas que no requieren el header Authorization (health check para Cloud
# Run / balanceadores de carga).
PUBLIC_PATHS = {"/health", "/healthz"}


class BearerTokenAuthMiddleware:
    """Middleware ASGI puro (no `BaseHTTPMiddleware`) para no interferir
    con las respuestas streaming (SSE) que usa el transporte
    `streamable-http` de MCP.

    Verifica que el header `Authorization: Bearer <token>` tenga el token
    correcto. Si no, responde 401 directamente sin tocar la app envuelta.
    """

    def __init__(self, app: ASGIApp, token: str = AUTH_TOKEN) -> None:
        self.app = app
        self.token = token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            # websockets / lifespan, etc. — deja pasar sin tocar.
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path in PUBLIC_PATHS:
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        auth_header = headers.get(b"authorization", b"").decode("latin-1")

        expected = f"Bearer {self.token}"
        if auth_header != expected:
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": b'{"error": "unauthorized", '
                    b'"detail": "Falta o es incorrecto el header '
                    b'Authorization: Bearer <token>"}',
                }
            )
            return

        await self.app(scope, receive, send)
