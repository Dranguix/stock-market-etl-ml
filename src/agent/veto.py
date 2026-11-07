"""El LLM como extractor y veto, no como asignador.

Para cada candidato del satélite, Haiku lee los titulares de los últimos días
(solo los publicados hasta asof), los clasifica en un evento estructurado y
puede vetar el nombre si el evento es material y negativo. No pone pesos: la
regla de src/agent/core_satellite.py ya los puso y el siguiente del ranking
toma el lugar del vetado.

Contra los sesgos documentados: los nombres van anonimizados (E1, E2...), el
orden se baraja en cada réplica y se vota entre K réplicas. Cada llamada queda
en data/eval/veto_log.jsonl con el hash del prompt y la respuesta.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ARTICLES = ROOT / "data" / "news" / "history" / "articles.parquet"
VETO_LOG = ROOT / "data" / "eval" / "veto_log.jsonl"
API_URL = "https://api.anthropic.com/v1/messages"

EVENTOS = (
    "recorte_distribucion", "cambio_guia", "fusion_adquisicion", "regulatorio",
    "legal_o_fraude", "resultados", "deuda_o_calificacion", "otro", "ninguno",
)
LOOKBACK_DAYS = 30
MAX_TITLES = 12
REPLICAS = 5

PROMPT = """Clasificas noticias de emisoras de la Bolsa Mexicana. Los nombres van ocultos a propósito: no intentes adivinar la empresa.
Para cada emisora, a partir SOLO de sus titulares:
- evento: el más material de {eventos}
- direccion: "negativa", "neutral" o "positiva"
- veto: true solo si hay un evento material y negativo (recorte de dividendo o distribución, recorte de guía, problema legal, fraude, regulatorio adverso, degradación de deuda). Ante la duda, false.
- motivo: una frase.
Responde solo JSON {{"E1": {{"evento", "direccion", "veto", "motivo"}}, ...}} con todas las emisoras.

{bloques}"""


def _aliases(ticker: str) -> list[str]:
    try:
        from src.ml.news_sources import get_entity_aliases

        names = get_entity_aliases(ticker)
    except Exception:
        names = []
    return sorted({ticker, *names}, key=len, reverse=True)


def _anonymize(title: str, aliases: list[str]) -> str:
    for alias in aliases:
        if len(alias) >= 3:
            title = re.sub(re.escape(alias), "LA EMISORA", title, flags=re.IGNORECASE)
    return title


def headlines_for(ticker: str, asof: pd.Timestamp, days: int = LOOKBACK_DAYS) -> list[str]:
    """Titulares publicados en (asof - days, asof]. Nada posterior a asof."""
    asof = pd.Timestamp(asof).tz_localize(None).normalize() + pd.Timedelta(hours=23, minutes=59)
    start = asof - pd.Timedelta(days=days)
    titles: list[str] = []
    if ARTICLES.exists():
        df = pd.read_parquet(ARTICLES, columns=["ticker", "published_dt", "title"])
        ts = pd.to_datetime(df["published_dt"], utc=True).dt.tz_localize(None)
        rows = df[(df["ticker"] == ticker) & (ts > start) & (ts <= asof)]
        titles += rows["title"].dropna().tolist()
    try:
        from src.ml.databursatil_news import SNAPSHOT_DIR, _articles_from_snapshot, list_snapshot_dates

        for day in list_snapshot_dates():
            if not (start.strftime("%Y-%m-%d") < day <= asof.strftime("%Y-%m-%d")):
                continue
            snap = json.loads((SNAPSHOT_DIR / f"{day}.json").read_text())
            titles += [a.title for a in _articles_from_snapshot(snap, ticker)]
    except Exception:
        pass
    seen, out = set(), []
    for t in titles:
        key = t.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(t.strip())
    return out[-MAX_TITLES:]


def _call(prompt: str, model: str) -> str:
    import httpx

    resp = httpx.post(
        API_URL,
        headers={
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 1500,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=90,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"Anthropic {resp.status_code}: {resp.text[:200]}")
    body = resp.json()
    return "".join(b.get("text", "") for b in body.get("content") or [] if b.get("type") == "text")


def _parse(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}


def classify(
    candidates: list[str],
    asof: pd.Timestamp,
    replicas: int = REPLICAS,
    model: str | None = None,
    call=_call,
) -> dict[str, dict]:
    """{ticker: {veto, votos, replicas, eventos, motivo}}. Sin titulares, el
    nombre no se manda: sin noticia no hay veto."""
    model = model or os.environ.get("ANTHROPIC_MODEL", "claude-haiku-5-5")
    news = {t: headlines_for(t, asof) for t in candidates}
    news = {t: [_anonymize(h, _aliases(t)) for h in hs] for t, hs in news.items() if hs}
    if not news:
        return {}

    rng = random.Random(f"{pd.Timestamp(asof).date()}")
    votes: dict[str, list[dict]] = {t: [] for t in news}
    for k in range(replicas):
        order = list(news)
        rng.shuffle(order)
        labels = {f"E{i + 1}": t for i, t in enumerate(order)}
        bloques = "\n\n".join(
            f"{lab}:\n" + "\n".join(f"- {h}" for h in news[t]) for lab, t in labels.items()
        )
        prompt = PROMPT.format(eventos=", ".join(EVENTOS), bloques=bloques)
        try:
            text = call(prompt, model)
        except Exception as exc:
            _log({"asof": str(pd.Timestamp(asof).date()), "replica": k, "error": str(exc)[:200]})
            continue
        parsed = _parse(text)
        _log({
            "asof": str(pd.Timestamp(asof).date()),
            "replica": k,
            "model": model,
            "labels": labels,
            "prompt_sha": hashlib.sha256(prompt.encode()).hexdigest()[:16],
            "response_sha": hashlib.sha256(text.encode()).hexdigest()[:16],
            "parsed": parsed,
        })
        for lab, t in labels.items():
            row = parsed.get(lab)
            if isinstance(row, dict):
                votes[t].append(row)

    out = {}
    for t, rows in votes.items():
        if not rows:
            continue
        n_veto = sum(1 for r in rows if r.get("veto") is True)
        eventos = [str(r.get("evento")) for r in rows]
        out[t] = {
            "veto": n_veto * 2 > len(rows),
            "votos": n_veto,
            "replicas": len(rows),
            "eventos": eventos,
            "motivo": next((r.get("motivo") for r in rows if r.get("veto") is True), rows[0].get("motivo")),
        }
    return out


def llm_veto(candidates: list[str], asof: pd.Timestamp, **kwargs) -> dict[str, str]:
    """La firma que espera core_satellite.decide: {ticker: motivo} de los vetados."""
    if not os.environ.get("ANTHROPIC_API_KEY") and "call" not in kwargs:
        return {}
    result = classify(candidates, asof, **kwargs)
    return {t: str(r.get("motivo") or "veto") for t, r in result.items() if r["veto"]}


def _log(row: dict) -> None:
    VETO_LOG.parent.mkdir(parents=True, exist_ok=True)
    row = {"ts": datetime.now(timezone.utc).isoformat(), **row}
    with VETO_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
