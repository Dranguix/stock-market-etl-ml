# Sistema multiagente conversacional

No confundir con `src/agent/` — ese paquete es el agente de **producción**
(`core_satellite.py`, `veto.py`) que decide el libro real de 50 USD. Este
paquete (`src/agents_mcp/`) es la capa conversacional nueva: analítica sobre
los mismos datos, para preguntas en lenguaje natural, usando
[Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python).

## Instalar

```bash
pip install -r requirements.txt   # agrega claude-agent-sdk y anthropic
export ANTHROPIC_API_KEY=...
```

## Probar un agente de dominio solo (antes de meterlo al orquestador)

```python
import asyncio
from src.agents_mcp.domains.mercado import mercado_agent

print(asyncio.run(mercado_agent.ask("¿qué datos de precios tienes?")))
```

Con los tools en modo STUB, la respuesta va a decir explícitamente que no
está implementado — así se ve si el agente al menos arma el loop y llama al
tool correcto.

## Correr el orquestador de punta a punta

```bash
python -m src.agents_mcp.orchestrator "¿qué hace este sistema?"
```

El orquestador (`orchestrator.py`) es un **punto de partida**, no la versión
final — está pensado para que el profesor lo reescriba/extienda. Expone cada
agente de dominio como una sola tool (`consultar_<dominio>`); al invocarla,
corre el loop COMPLETO de ese agente de dominio y regresa solo su respuesta
final en texto. El orquestador nunca ve SQL ni datos crudos.

## Qué le toca a cada equipo

Cada archivo en `domains/` ya tiene:
1. Un `system_prompt` grounded en el código real de su dominio (con
   referencias a los módulos reales — `src/data/store.py`,
   `src/api/fundamentals.py`, `src/ml/news.py`, etc.)
2. Un tool `schema_<dominio>()` que ya describe el contrato de datos real —
   mantenerlo actualizado si cambian columnas.
3. Un tool `query_<dominio>(pregunta)` en modo **STUB** — devuelve un
   mensaje fijo diciendo que falta implementarlo. Esto es lo que cada
   equipo tiene que reemplazar con texto-a-consulta real (pandas sobre los
   parquets/json, o SQL si migran a Postgres).

Al implementar `query_<dominio>` de verdad:
- Mantén la disciplina point-in-time donde aplique (Fundamentales,
  Noticias histórico) — nunca expongas un dato con fecha de disponibilidad
  posterior a la fecha de corte de la pregunta.
- Si tu dominio necesita anonimizar (Noticias, para evaluar sesgos del LLM
  como hace `veto.py`), hazlo en el tool, no confíes en que el modelo lo
  recuerde solo.
- Un tool por consulta específica está bien si `query_<dominio>` genérico
  no basta (ej. `forecast_features_ml` aparte de `query_features_ml`).

## Por qué cada agente corre su propio loop

`DomainAgent.ask()` (`domain_agent.py`) no reimplementa el while/tool_calls
manual que ya existe en `src/agent/investor.py` — usa
`create_sdk_mcp_server()` + `query()`/`ClaudeAgentOptions` del SDK, que
maneja el ida-y-vuelta con Claude hasta que no hay más tool calls. El
orquestador hace exactamente lo mismo, un nivel arriba: sus "tools" son
agentes de dominio completos, no funciones de datos.
