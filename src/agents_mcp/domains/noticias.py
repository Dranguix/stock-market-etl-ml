"""Agente de dominio: Noticias (segmentadas bmv/usa/etf/crypto/fibra).

Esquema real: src/ml/news.py (NEWS_FEATURE_COLS, dedup TF-IDF, decay
temporal), src/ml/news_history.py (panel histórico PIT para backtest).

Dos almacenes distintos:
  - data/news/{slug}_{YYYY-MM-DD}.json + latest.json   (snapshot de producción,
    lo que lee el portero de src/agent/veto.py)
  - data/news/history/articles.parquet + daily_features.parquet  (panel
    histórico point-in-time, vía GDELT, para backtest)

Si la pregunta involucra evaluar sesgos del LLM (como hace veto.py), aplica
la misma anonimización: ticker/alias fuera del texto, nombres como E1/E2.

TODO equipo Noticias: implementar query_noticias de verdad sobre
articles.parquet / los snapshots diarios.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from claude_agent_sdk import tool

from src.agents_mcp.domain_agent import DomainAgent

NEWS_DIR = Path("data/news")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

SYSTEM_PROMPT = """Eres el agente de dominio de Noticias: titulares y
features de sentimiento por ticker, segmentados por universo (bmv, usa,
etf, crypto, fibra). Si te preguntan sobre sesgos o evaluación del portero
de noticias, anonimiza igual que src/agent/veto.py (nunca repitas el nombre
real de la emisora en ese contexto). Si no hay cobertura de noticias para
un ticker/fecha, dilo — no inventes sentimiento."""


@tool(
    "schema_noticias",
    "Describe las columnas, almacenes y la disciplina de anonimización del dominio Noticias.",
    {},
)
async def schema_noticias(args: dict) -> dict:
    return {
        "content": [{
            "type": "text",
            "text": (
                "NEWS_FEATURE_COLS (src/ml/news.py): news_sent_score, "
                "news_count_7d, news_sent_delta, news_sent_vol, "
                "news_relevance_avg, news_esg_score. "
                "Producción: data/news/{bmv|usa|etf|crypto|fibra}_"
                "{YYYY-MM-DD}.json + latest.json. "
                "Histórico PIT: data/news/history/articles.parquet, "
                "daily_features.parquet (GDELT). "
                "Dedup TF-IDF (umbral coseno 0.5) + decay temporal "
                "(lambda=0.3/día) ya aplicados en estas features."
            ),
        }]
    }


def _latest_news_file() -> Path | None:
    """Archivo YYYY-MM-DD.json más reciente en data/news/ (snapshot
    combinado de todos los universos, no los segmentados bmv_/usa_/...)."""
    candidatos = [p for p in NEWS_DIR.glob("*.json") if _DATE_RE.match(p.stem)]
    if not candidatos:
        return None
    return max(candidatos, key=lambda p: p.stem)


def _load_news_file(fecha: str | None) -> tuple[Path | None, list[dict]]:
    if fecha:
        path = NEWS_DIR / f"{fecha}.json"
        if not path.exists():
            return path, []
    else:
        path = _latest_news_file()
        if path is None:
            return None, []
    with open(path, encoding="utf-8") as f:
        return path, json.load(f)


def _fmt_item(item: dict) -> str:
    sources = ", ".join(item.get("sources", [])) or "sin fuente registrada"
    return (
        f"- {item.get('ticker')}: sentimiento={item.get('news_sent_score')}, "
        f"n_articulos={item.get('n_articles')}, "
        f"relevancia_prom={item.get('news_relevance_avg')}, "
        f"delta_sent={item.get('news_sent_delta')}, vol_sent={item.get('news_sent_vol')}, "
        f"fuentes=[{sources}]"
    )


@tool(
    "query_noticias",
    "Da el sentimiento/cobertura de noticias de un ticker (o un resumen de "
    "varios) en una fecha dada. ticker y fecha son opcionales: sin ticker "
    "resume varios; sin fecha usa el snapshot más reciente en data/news/.",
    {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Ticker a consultar, ej. WALMEX. Opcional."},
            "fecha": {"type": "string", "description": "YYYY-MM-DD. Opcional, default: más reciente."},
        },
        "required": [],
    },
)
async def query_noticias(args: dict) -> dict:
    ticker = (args.get("ticker") or "").strip().upper() or None
    fecha = (args.get("fecha") or "").strip() or None

    path, items = _load_news_file(fecha)

    if path is None:
        return {
            "content": [{
                "type": "text",
                "text": "No hay ningún snapshot de noticias en data/news/.",
            }]
        }
    if not items and fecha:
        return {
            "content": [{
                "type": "text",
                "text": f"No existe el archivo de noticias para la fecha {fecha} ({path}).",
            }]
        }

    fecha_real = fecha or path.stem

    if ticker:
        match = [it for it in items if it.get("ticker", "").upper() == ticker]
        if not match:
            return {
                "content": [{
                    "type": "text",
                    "text": (
                        f"No hay noticias registradas para {ticker} en la fecha "
                        f"{fecha_real} (archivo {path.name}, {len(items)} tickers "
                        f"cubiertos ese día, {ticker} no está entre ellos)."
                    ),
                }]
            }
        lineas = [_fmt_item(it) for it in match]
        texto = f"Noticias de {ticker} — {fecha_real} ({path.name}):\n" + "\n".join(lineas)
        return {"content": [{"type": "text", "text": texto}]}

    # Sin ticker: resumen de varios (hasta 15 para no saturar la respuesta).
    muestra = items[:15]
    lineas = [_fmt_item(it) for it in muestra]
    extra = f"\n… y {len(items) - 15} tickers más." if len(items) > 15 else ""
    texto = (
        f"Resumen de noticias — {fecha_real} ({path.name}), {len(items)} tickers "
        f"con cobertura:\n" + "\n".join(lineas) + extra
    )
    return {"content": [{"type": "text", "text": texto}]}


noticias_agent = DomainAgent(
    name="noticias",
    system_prompt=SYSTEM_PROMPT,
    tools=[schema_noticias, query_noticias],
)
