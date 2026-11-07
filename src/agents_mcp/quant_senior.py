"""Agente supervisor "quant senior": visión 360 de un ticker.

A diferencia de los agentes de dominio (`src/agents_mcp/domains/*.py`), que
cada uno corre SU PROPIO loop de Claude decidiendo qué tool llamar, este
supervisor NO deja que el LLM decida el orden. El orden lo fuerza el código
(ver investigación de la sesión — DAG fijo orquestado externamente, no un
ReAct libre): primero búsqueda (mercado+noticias+fundamentales, en
paralelo), luego ML (pronóstico), luego optimización (rol en el portafolio
óptimo), y solo al final UNA llamada a Claude que sintetiza — sin más tools,
solo con lo que ya se juntó, citando de dónde salió cada dato.

Esto evita dos cosas que la investigación marcó como riesgo:
  1. Que el LLM "rellene" un paso con su propio conocimiento en vez de
     usar la herramienta real (el pronóstico y el portafolio SIEMPRE
     vienen de `pronostico_precio`/`recomendar_portafolio`, nunca
     inventados por el supervisor).
  2. Que presente una predicción débil con más confianza de la que
     merece (el texto de honestidad de cada tool se pasa completo al
     supervisor, y el prompt le exige mantenerlo).
"""
from __future__ import annotations

import asyncio
import os

import anthropic

SUPERVISOR_MODEL = "claude-sonnet-5-5"

SYSTEM_PROMPT = """Eres un quant financiero senior armando una "visión 360"
de un ticker para alguien que va a decidir con dinero real (aunque sea
poco). Tienes 4 bloques de evidencia ya recolectados por herramientas
reales — mercado, noticias, pronóstico ML, y rol en el portafolio óptimo.
No tienes acceso a ninguna otra herramienta ni a tu propio conocimiento de
mercado: SOLO puedes afirmar lo que está en los bloques de abajo. Si un
bloque dice "no disponible" o "error", dilo explícitamente, no lo rellenes.

Reglas que debes seguir sin excepción:
1. Cada afirmación concreta (precio, retorno, sentimiento, peso en
   portafolio) debe llevar su fuente entre corchetes, ej. "[fuente:
   mercado]", "[fuente: pronóstico ML]".
2. Si el bloque de pronóstico dice que el modelo no tiene skill probado
   (1 día, o cualquier ticker que no sea SPY a 60-90d), usa lenguaje
   condicional ("podría", "especulativo", "sin evidencia de ventaja") —
   nunca presentes ese número como una predicción confiable.
3. Estructura la respuesta en: Resumen (2 líneas), Estado actual,
   Noticias relevantes, Pronóstico (con su honestidad), Rol en el
   portafolio óptimo, Conclusión (una recomendación de tono, nunca una
   orden de "compra/vende").
4. Sé breve — esto es una síntesis, no un ensayo."""


async def _gather_busqueda(ticker: str) -> dict[str, str]:
    """Etapa 1: búsqueda, en paralelo — mercado + noticias + fundamentales."""
    from src.agents_mcp.domains.fundamentales import fundamentales_agent
    from src.agents_mcp.domains.mercado import mercado_agent
    from src.agents_mcp.domains.noticias import noticias_agent

    mercado, noticias, fundamentales = await asyncio.gather(
        mercado_agent.ask(f"estado actual de {ticker}: precio, retorno y volatilidad de 30 días"),
        noticias_agent.ask(f"noticias recientes de {ticker}, sentimiento y número de artículos"),
        fundamentales_agent.ask(f"últimos fundamentales disponibles de {ticker}"),
        return_exceptions=True,
    )
    return {
        "mercado": mercado if isinstance(mercado, str) else f"error: {mercado}",
        "noticias": noticias if isinstance(noticias, str) else f"error: {noticias}",
        "fundamentales": fundamentales if isinstance(fundamentales, str) else f"error: {fundamentales}",
    }


async def _gather_ml(ticker: str) -> str:
    """Etapa 2: ML — SOLO se corre con el ticker de la etapa 1, nunca uno
    inventado. Llama el tool directo (no pasa por el loop de Claude del
    agente de dominio) porque el resultado ya es determinista/estructurado."""
    from src.agents_mcp.domains.features_ml import pronostico_precio

    result = await pronostico_precio.handler({"ticker": ticker, "horizonte_dias": 42})
    return result["content"][0]["text"]


async def _gather_optimizacion(ticker: str) -> str:
    """Etapa 3: optimización — rol del ticker en el portafolio ya resuelto
    (covarianza real + lambda validado por backtest). Si el ticker no está
    en el universo de 10, se dice explícito, no se improvisa un peso."""
    from src.agents_mcp.domains.features_ml import recomendar_portafolio

    result = await recomendar_portafolio.handler({})
    texto = result["content"][0]["text"]
    if ticker.upper() not in texto.upper():
        texto += (
            f"\n\n({ticker} no está en el universo de 10 tickers del optimizador "
            "actual — no se puede dar su peso recomendado.)"
        )
    return texto


async def vision_360(ticker: str) -> str:
    """El DAG completo: búsqueda -> ML -> optimización -> síntesis (1 sola
    llamada final a Claude, sin tools, solo con lo ya recolectado)."""
    ticker = ticker.strip().upper()

    # Etapas 1 y 2 pueden correr en paralelo (ML no depende de noticias/
    # fundamentales, solo del ticker); la síntesis sí espera a las 3.
    busqueda, pronostico, portafolio = await asyncio.gather(
        _gather_busqueda(ticker),
        _gather_ml(ticker),
        _gather_optimizacion(ticker),
    )

    bloques = (
        f"=== MERCADO [fuente: mercado] ===\n{busqueda['mercado']}\n\n"
        f"=== NOTICIAS [fuente: noticias] ===\n{busqueda['noticias']}\n\n"
        f"=== FUNDAMENTALES [fuente: fundamentales] ===\n{busqueda['fundamentales']}\n\n"
        f"=== PRONÓSTICO ML [fuente: pronóstico ML] ===\n{pronostico}\n\n"
        f"=== PORTAFOLIO ÓPTIMO [fuente: optimización] ===\n{portafolio}"
    )

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=SUPERVISOR_MODEL,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Ticker: {ticker}\n\n{bloques}\n\nArma la visión 360.",
        }],
    )
    return "".join(b.text for b in response.content if b.type == "text")
