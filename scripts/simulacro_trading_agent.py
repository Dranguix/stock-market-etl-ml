#!/usr/bin/env python3
"""Simulacro del agente de trading: ciclo completo con datos reales, sin gastar créditos.

Un Claude de guion sustituye a la API. Intenta romper las reglas a propósito
y al final propone un reparto válido con los dos primeros del ranking.
El libro vive en memoria: no escribe data/agent.

    python -X utf8 scripts/simulacro_trading_agent.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import investor
from src.agent.book import Book
from src.agent.trading import AgenteTrading


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
        evitar = self.ranking.get("evitar") or []
        p = self.paso
        if p == 1:
            return [_call(p, "ver_libro", {}), _call(p + 10, "ver_noticias", {"ticker": "WALMEX"})]
        if p == 2:
            return [_call(p, "ver_ranking", {"universo": "bmv", "horizonte": "4s"})]
        if p == 3 and top:
            return [_call(p, "ver_woe", {"ticker": top[0]})]
        if p == 4 and evitar:
            return [_call(p, "proponer_asignacion", {"allocations": [{"ticker": evitar[0], "usd": 20}], "motivo": "nombre de la lista evitar"})]
        if p == 5 and len(top) >= 3:
            return [_call(p, "proponer_asignacion", {"allocations": [{"ticker": t, "usd": 25} for t in top[:3]], "motivo": "me paso de 50"})]
        if p == 6 and len(top) >= 2:
            return [_call(p, "proponer_asignacion", {
                "allocations": [{"ticker": top[0], "usd": 20}, {"ticker": top[1], "usd": 15}],
                "cash_usd": 15,
                "motivo": f"{top[0]} y {top[1]} son 1 y 2 en ver_ranking bmv 4s; ver_woe de {top[0]}.",
            })]
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
    claude = ClaudeDeGuion()
    investor.chat_with_tools = claude  # el ciclo de AgenteInversor ya no sale a la API
    os.environ.setdefault("ANTHROPIC_API_KEY", "simulacro")

    book = Book(payload={"cash_usd": 50, "budget_usd": 50, "positions": {}, "fills": []})
    agent = AgenteTrading(book)
    result = agent.act(apply=True)

    print("Consultas del Claude de guion:")
    for row in claude.enviados:
        print(_linea(row, result["ok"]))
    print()
    print(json.dumps({
        "ok": result["ok"],
        "turnos": result["turnos"],
        "asignacion": result["asignacion"],
        "tickers_vistos": result["tickers_vistos"],
        "libro_final": (result.get("aplicado") or {}).get("libro"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
