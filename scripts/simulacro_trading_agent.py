#!/usr/bin/env python3
"""Simulacro del agente de trading: ciclo completo con datos reales, sin gastar créditos.

Un Claude de guion sustituye a la API. Intenta romper las reglas a propósito
y al final propone un reparto con los nombres comprables del ranking. Si la señal es
vieja, el guard lo rechaza y el guion se queda en efectivo.
El libro vive en memoria: no escribe data/agent.

    python -X utf8 scripts/simulacro_trading_agent.py
    python -X utf8 scripts/simulacro_trading_agent.py --hoy 2026-11-02  # como si la señal fuera reciente
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import investor
from src.agent.book import Book
from src.agent.trading import AgenteTrading, consultar


def _call(i: int, name: str, args: dict) -> dict:
    return {"id": f"sim_{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def _tool_results(messages: list[dict]) -> dict[str, dict]:
    return {m["tool_call_id"]: json.loads(m["content"]) for m in messages if m.get("role") == "tool"}


class ClaudeDeGuion:
    """Responde como chat_with_tools, paso por paso, y anota qué le contestó el sistema."""

    def __init__(self) -> None:
        self.paso = 0
        self.enviados: list[dict] = []
        self.ranking: dict = {}

    def __call__(self, messages: list[dict], tools: list[dict], model: str | None = None) -> dict:
        resultados = _tool_results(messages)
        for row in self.enviados:
            if "resultado" not in row and row["id"] in resultados:
                row["resultado"] = resultados[row["id"]]
                if row["name"] == "ver_ranking":
                    self.ranking = row["resultado"]

        self.paso += 1
        calls = self._siguiente(tools)
        for call in calls:
            fn = call["function"]
            self.enviados.append({"id": call["id"], "name": fn["name"], "args": json.loads(fn["arguments"])})
        message = {"role": "assistant", "content": None}
        if calls:
            message["tool_calls"] = calls
        return {"choices": [{"finish_reason": "tool_calls" if calls else "stop", "message": message}]}

    def _siguiente(self, tools: list[dict]) -> list[dict]:
        top = [p["ticker"] for p in self.ranking.get("top") or []]
        comprables = self.ranking.get("comprables") or top
        no_comprables = [t for t in top if t not in comprables]
        evitar = self.ranking.get("evitar") or []
        bloqueado = evitar[0] if evitar else (no_comprables[0] if no_comprables else "MEGACPO")
        p = self.paso
        if p == 1:
            return [_call(p, "ver_libro", {}), _call(p + 10, "ver_noticias", {"ticker": "WALMEX"})]
        if p == 2:
            return [_call(p, "ver_ranking", {"universo": "bmv", "horizonte": "4s"})]
        if p == 3 and comprables:
            return [_call(p, "ver_woe", {"ticker": comprables[0]})]
        if p == 4:
            return [_call(p, "ver_woe", {"ticker": bloqueado})]
        if p == 5:
            return [_call(p, "proponer_asignacion", {"allocations": [{"ticker": bloqueado, "usd": 20}], "cash_usd": 30, "motivo": "nombre no autorizado por comprables"})]
        if p == 6 and top:
            exceso = top[:3] if len(top) >= 3 else top[:2]
            return [_call(p, "proponer_asignacion", {"allocations": [{"ticker": t, "usd": 30} for t in exceso], "motivo": "me paso de 50"})]
        if p == 7 and comprables:
            if len(comprables) >= 2:
                allocs = [{"ticker": comprables[0], "usd": 20}, {"ticker": comprables[1], "usd": 15}]
                cash = 15
                motivo = f"{comprables[0]} y {comprables[1]} están en comprables de ver_ranking bmv 4s; ver_woe de {comprables[0]}."
            else:
                allocs = [{"ticker": comprables[0], "usd": 20}]
                cash = 30
                motivo = f"{comprables[0]} está en comprables de ver_ranking bmv 4s; ver_woe de {comprables[0]}."
            return [_call(p, "proponer_asignacion", {
                "allocations": allocs,
                "cash_usd": cash,
                "motivo": motivo,
            })]
        if p == 8:
            return [_call(p, "proponer_asignacion", {"allocations": [], "cash_usd": 50, "motivo": "señal vieja: efectivo"})]
        return []


def _linea(row: dict, aceptada: bool) -> str:
    res = row.get("resultado")
    if res is None:
        estado = "ACEPTADA" if aceptada and row["name"] == "proponer_asignacion" else "sin respuesta"
    elif res.get("ok"):
        estado = "ok"
    else:
        estado = f"RECHAZADA: {res.get('error')}"
    return f"  {row['name']}({json.dumps(row['args'], ensure_ascii=False)}) -> {estado}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hoy", type=date.fromisoformat, help="Fecha que usa el guard, en ISO. Por defecto, hoy")
    args = parser.parse_args()

    claude = ClaudeDeGuion()
    investor.chat_with_tools = claude  # el ciclo de AgenteInversor ya no sale a la API
    os.environ.setdefault("ANTHROPIC_API_KEY", "simulacro")

    book = Book(payload={"cash_usd": 50, "budget_usd": 50, "positions": {}, "fills": []})
    agent = AgenteTrading(book, hoy=args.hoy)
    result = consultar("Reparte el libro y termina con proponer_asignacion.", agent)
    aplicado = agent.aplicar(result["asignacion"]) if result["asignacion"] else None

    print("Consultas del Claude de guion:")
    for row in claude.enviados:
        print(_linea(row, result["ok"]))
    print()
    print(json.dumps({
        **result,
        "tickers_vistos": sorted(agent.policy.vistos),
        "tickers_evitados": sorted(agent.policy.evitados),
        "libro_final": (aplicado or {}).get("libro"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
