"""Agente de dominio: Mercado (precios OHLCV + FX).

Esquema real: src/data/store.py (grupos bmv/usa/etf/crypto/fibra/macro,
MXN=X en macro) y scripts/fetch_intraday.py (barras de 5m, solo BMV
líquidas, desde 2024-11).

TODO equipo Mercado: reemplazar query_mercado por texto-a-consulta real
(pandas sobre data/market/*.parquet y data/intraday/*.parquet, o SQL si
migran a una base). schema_mercado ya describe el contrato real — mantenlo
actualizado si cambian columnas.
"""
from __future__ import annotations

import numpy as np

from claude_agent_sdk import tool

from src.agents_mcp.domain_agent import DomainAgent
from src.data import store

SYSTEM_PROMPT = """Eres el agente de dominio de Mercado: precios OHLCV
diarios e intradía, y tipo de cambio USD/MXN. Tu única fuente de verdad son
tus tools. Si no tienes el dato, dilo — nunca inventes un precio ni una
fecha."""


@tool(
    "schema_mercado",
    "Describe las tablas y columnas disponibles del dominio Mercado.",
    {},
)
async def schema_mercado(args: dict) -> dict:
    return {
        "content": [{
            "type": "text",
            "text": (
                "data/market/{SYMBOL}.parquet — Date(index), Open, High, Low, "
                "Close, Volume. Grupos (src/data/store.py): bmv, usa, etf, "
                "crypto, fibra, macro (macro incluye USDMXN='MXN=X', "
                "NAFTRAC, WTI='CL=F', SP500='^GSPC'). "
                "data/intraday/{ticker}.parquet — barras de 5 minutos, solo "
                "BMV líquidas (INTRADAY_MAP), histórico desde 2024-11-01."
            ),
        }]
    }


@tool(
    "query_mercado",
    "Da el último cierre, retorno % y volatilidad realizada de un ticker del "
    "store (nombre amigable, ej. 'WALMEX', 'SPY', 'USDMXN') sobre una ventana "
    "de días hábiles.",
    {"ticker": str, "dias": int},
)
async def query_mercado(args: dict) -> dict:
    ticker = str(args.get("ticker", "")).strip()
    dias = args.get("dias") or 30
    try:
        dias = int(dias)
    except (TypeError, ValueError):
        dias = 30

    try:
        df = store.get(ticker)
        if df is None or df.empty or "Close" not in df.columns:
            return {
                "content": [{
                    "type": "text",
                    "text": f"No tengo datos de {ticker!r} en el store local.",
                }]
            }

        closes = df["Close"].dropna()
        if closes.empty:
            return {
                "content": [{
                    "type": "text",
                    "text": f"No tengo datos de cierre utilizables para {ticker!r}.",
                }]
            }

        last_close = float(closes.iloc[-1])
        last_date = closes.index[-1]
        last_date_str = last_date.date().isoformat() if hasattr(last_date, "date") else str(last_date)

        window = closes.tail(dias + 1)
        if len(window) >= 2:
            ret_pct = (window.iloc[-1] / window.iloc[0] - 1.0) * 100.0
            log_rets = np.log(window / window.shift(1)).dropna()
            vol_diaria = float(log_rets.std()) if len(log_rets) >= 2 else None
        else:
            ret_pct = None
            vol_diaria = None

        dias_usados = len(window) - 1 if len(window) >= 2 else 0

        partes = [
            f"{ticker}: último cierre {last_close:,.4f} el {last_date_str}.",
        ]
        if ret_pct is not None:
            partes.append(
                f"Retorno de los últimos {dias_usados} días hábiles: {ret_pct:+.2f}%."
            )
        else:
            partes.append(
                f"No hay suficiente historial para calcular el retorno de {dias} días."
            )
        if vol_diaria is not None:
            partes.append(
                f"Volatilidad diaria realizada (std de log-retornos, {dias_usados} días): "
                f"{vol_diaria * 100:.2f}% diario."
            )
        else:
            partes.append("No hay suficiente historial para calcular volatilidad.")

        return {"content": [{"type": "text", "text": " ".join(partes)}]}

    except Exception as e:
        return {
            "content": [{
                "type": "text",
                "text": f"No pude obtener datos de {ticker!r}: {e}",
            }]
        }


mercado_agent = DomainAgent(
    name="mercado",
    system_prompt=SYSTEM_PROMPT,
    tools=[schema_mercado, query_mercado],
)
