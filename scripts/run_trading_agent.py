#!/usr/bin/env python3
"""Un ciclo del agente de trading, que solo consulta a los agentes de Features/ML.

    python scripts/run_trading_agent.py --ejemplo
    python scripts/run_trading_agent.py            # llama a Claude si hay ANTHROPIC_API_KEY
    python scripts/run_trading_agent.py --aplicar  # además escribe el libro
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.agent.book import Book
from src.agent.trading import AgenteTrading

BOOK_PATH = ROOT / "data" / "agent" / "trading_book.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ejemplo", action="store_true", help="Imprime la pregunta y dos tool calls, sin llamar a Claude")
    parser.add_argument("--aplicar", action="store_true", help=f"Si el guard acepta, escribe {BOOK_PATH.relative_to(ROOT)}")
    args = parser.parse_args()

    book = Book(BOOK_PATH)
    agent = AgenteTrading(book)

    if args.ejemplo:
        pregunta = agent.observe()
        permitido = {"name": "ver_ranking", "input": {"universo": "bmv", "horizonte": "4s"}}
        bloqueado = {"name": "ver_noticias", "input": {"ticker": "WALMEX"}}
        print(json.dumps({
            "system": pregunta["system"],
            "user": pregunta["user"],
            "tools": [t["function"]["name"] for t in pregunta["tools"]],
            "consulta_permitida": permitido,
            "resultado_permitida": agent.dispatch(permitido["name"], permitido["input"]),
            "consulta_bloqueada": bloqueado,
            "resultado_bloqueada": agent.dispatch(bloqueado["name"], bloqueado["input"]),
            "tickers_vistos": sorted(agent.policy.vistos),
        }, ensure_ascii=False, indent=2))
        return

    result = agent.act(apply=args.aplicar)
    if not result.get("ok") and result.get("pregunta"):
        print(result["error"])
        print(json.dumps({"user": result["pregunta"]["user"]}, ensure_ascii=False, indent=2))
        return
    slim = {k: result.get(k) for k in ("ok", "modelo", "asignacion", "aplicado", "turnos", "agentes_features_ml", "tickers_vistos")}
    print(json.dumps(slim, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
