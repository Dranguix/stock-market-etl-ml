"""Libro de paper trading en dólares, fraccionario. Presupuesto fijo: $50."""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

BUDGET_USD = 50.0


def _today() -> str:
    return date.today().isoformat()


class Book:
    def __init__(self, path: Path | None = None, payload: dict | None = None):
        self.path = path
        if payload is not None:
            self.data = payload
        elif path is not None and path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8"))
        else:
            self.data = {
                "cash_usd": BUDGET_USD,
                "budget_usd": BUDGET_USD,
                "positions": {},
                "fills": [],
                "opened_at": _today(),
            }

    @property
    def cash(self) -> float:
        return float(self.data["cash_usd"])

    @property
    def positions(self) -> dict:
        return self.data["positions"]

    def snapshot(self) -> dict:
        return {
            "cash_usd": round(self.cash, 2),
            "budget_usd": BUDGET_USD,
            "posiciones": {
                t: {
                    "usd": round(float(p["usd"]), 2),
                    "acciones": p["shares"],
                    "precio_usd": p["price_usd"],
                    "abierta_desde": p["opened_at"],
                    "stop_usd": p.get("stop_usd"),
                }
                for t, p in self.positions.items()
            },
        }

    def days_open(self, ticker: str, today: str | None = None) -> int | None:
        pos = self.positions.get(ticker)
        if not pos:
            return None
        start = datetime.fromisoformat(pos["opened_at"]).date()
        end = datetime.fromisoformat(today or _today()).date()
        return (end - start).days

    def apply(self, proposal: dict, prices_usd: dict[str, float], today: str | None = None) -> dict:
        """Rebalancea hacia los dólares pedidos. prices_usd: ticker → precio en USD."""
        today = today or _today()
        cash = float(self.data["cash_usd"])
        targets = {a["ticker"]: float(a["usd"]) for a in proposal["allocations"]}
        stops = {a["ticker"]: a.get("stop_usd") for a in proposal["allocations"]}
        fills = []

        for ticker in list(self.positions):
            if ticker not in targets:
                targets[ticker] = 0.0

        for ticker, usd in targets.items():
            price = prices_usd.get(ticker)
            if price is None or price <= 0:
                if usd == 0 and ticker in self.positions:
                    price = float(self.positions[ticker]["price_usd"])
                else:
                    raise ValueError(f"sin precio USD para {ticker}")
            current = self.positions.get(ticker)
            current_usd = float(current["usd"]) if current else 0.0
            delta = round(usd - current_usd, 2)
            if abs(delta) < 0.5:
                if current and stops.get(ticker):
                    current["stop_usd"] = stops[ticker]
                continue
            shares = round(usd / price, 6)
            if usd <= 0:
                cash += current_usd
                self.positions.pop(ticker, None)
                fills.append({"ticker": ticker, "lado": "venta", "usd": current_usd})
            elif current is None:
                cash -= usd
                self.positions[ticker] = {
                    "usd": usd,
                    "shares": shares,
                    "price_usd": price,
                    "opened_at": today,
                    "stop_usd": stops.get(ticker),
                }
                fills.append({"ticker": ticker, "lado": "compra", "usd": usd})
            else:
                cash -= delta
                current["usd"] = usd
                current["shares"] = shares
                current["price_usd"] = price
                if stops.get(ticker):
                    current["stop_usd"] = stops[ticker]
                fills.append({"ticker": ticker, "lado": "ajuste", "usd": delta})

        self.data["cash_usd"] = round(cash, 4)
        self.data["fills"].extend(fills)
        self.save()
        return {"fills": fills, "libro": self.snapshot()}

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
