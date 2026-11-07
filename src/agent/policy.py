"""Política del inversor: el prompt y las reglas que Python no deja romper."""
from __future__ import annotations

from datetime import date

from src.agent.book import BUDGET_USD, Book

MAX_NAMES = 3
MIN_TICKET = 10.0
MAX_TICKET = 25.0
MIN_HOLD_DAYS = 5
TOL = 0.05


class PoliticaInversor:
    name = "inversor_50"
    respeta_stop = True

    def system_prompt(self) -> str:
        return f"""Eres el agente inversor de un libro de paper trading.
El presupuesto es uno solo: {BUDGET_USD:.0f} dólares. Tú repartes esa cantidad.

Reglas que el sistema va a rechazar si las rompes:
- Long only. El efectivo es una posición válida.
- Máximo {MAX_NAMES} nombres con dólares > 0.
- Cada nombre lleva 0, o entre {MIN_TICKET:.0f} y {MAX_TICKET:.0f} dólares.
- La suma de los dólares de los nombres más el efectivo es exactamente {BUDGET_USD:.0f}.
- No cambies el tamaño de una posición con menos de {MIN_HOLD_DAYS} días, salvo que el precio haya tocado su stop.
- No inventes tickers. Solo nombres que te hayan devuelto las herramientas.

Cómo decidir, pidiendo herramientas. No adivines números que no te hayan devuelto:
1. ver_modelos: el IV de cada variable y el AUC del LSTM. Si el IV es menor a 0.02, esa variable no separa.
2. ver_horizontes: trae el ranking completo de cada plazo, todas las emisoras, no solo el puesto 1. El campo libro es un reparto posible, no una orden. Puedes elegir cualquier nombre de ranking_completo.
3. ver_libro y ver_clima.
4. Pide ver_woe de varios nombres, no solo del primero. También ver_noticias y ver_salida de los que sí podrían llevar dólares.
5. Cierra con proponer_asignacion. Si nada separa, deja los 50 en efectivo.

El WoE te dice por qué un nombre está arriba y su probabilidad de subir en 5 días.
El clima (alcista, lateral, bajista) ya viene dentro de esa historia cuando el dato existe.
La salida es el precio del stop, no una opinión. Si la probabilidad es floja, pon 0 dólares.
"""

    def user_message(self, book: Book) -> str:
        return (
            f"Hoy es {date.today().isoformat()}, después del cierre de la BMV. "
            f"El libro tiene {BUDGET_USD:.0f} dólares en total y "
            f"{book.cash:.2f} dólares en efectivo. "
            "Usa las herramientas y termina con proponer_asignacion."
        )

    def guard(self, proposal: dict, book: Book, prices_usd: dict[str, float] | None = None) -> dict:
        """Devuelve {{ok, motivo, allocations, cash_usd}} o el error para que Claude corrija."""
        raw = proposal.get("allocations") or proposal.get("asignaciones") or []
        if not isinstance(raw, list):
            return {"ok": False, "error": "allocations tiene que ser una lista"}

        from src.agent.tools import find_ticker

        cleaned = []
        seen = set()
        for row in raw:
            ticker = str(row.get("ticker", "")).upper().strip()
            usd = float(row.get("usd", 0) or 0)
            if not ticker or usd <= 0:
                continue
            if find_ticker(ticker) is None:
                return {"ok": False, "error": f"{ticker} no está en los universos"}
            if ticker in seen:
                return {"ok": False, "error": f"{ticker} viene repetido"}
            seen.add(ticker)
            cleaned.append({
                "ticker": ticker,
                "usd": round(usd, 2),
                "horizon": row.get("horizon") or row.get("horizonte") or "4s",
                "stop_usd": row.get("stop_usd"),
            })

        if len(cleaned) > MAX_NAMES:
            return {"ok": False, "error": f"máximo {MAX_NAMES} nombres, llegaron {len(cleaned)}"}

        for row in cleaned:
            if row["usd"] < MIN_TICKET - TOL or row["usd"] > MAX_TICKET + TOL:
                return {
                    "ok": False,
                    "error": f"{row['ticker']} lleva ${row['usd']}. El mínimo es {MIN_TICKET:.0f} y el máximo {MAX_TICKET:.0f}. Si no alcanza, pon 0.",
                }

        invested = round(sum(r["usd"] for r in cleaned), 2)
        if invested > BUDGET_USD + TOL:
            return {"ok": False, "error": f"la suma {invested} pasa de {BUDGET_USD:.0f}"}

        cash = round(BUDGET_USD - invested, 2)
        if "cash_usd" in proposal and proposal["cash_usd"] is not None:
            stated = round(float(proposal["cash_usd"]), 2)
            if abs(stated - cash) > 0.05:
                return {
                    "ok": False,
                    "error": f"cash_usd={stated} no cuadra. Con esos nombres el efectivo es {cash}.",
                }

        held = []
        for row in cleaned:
            days = book.days_open(row["ticker"])
            if days is None or days >= MIN_HOLD_DAYS:
                continue
            prev = float(book.positions[row["ticker"]]["usd"])
            if abs(prev - row["usd"]) < 1:
                continue
            stop = book.positions[row["ticker"]].get("stop_usd")
            price = (prices_usd or {}).get(row["ticker"])
            if self.respeta_stop and stop and price and price <= float(stop):
                continue
            held.append(row["ticker"])
        if held:
            salvo = ", salvo que haya tocado el stop" if self.respeta_stop else ""
            return {
                "ok": False,
                "error": f"{', '.join(held)} lleva menos de {MIN_HOLD_DAYS} días. Déjalo como está{salvo}.",
            }

        return {
            "ok": True,
            "allocations": cleaned,
            "cash_usd": cash,
            "motivo": proposal.get("motivo") or proposal.get("reason") or "",
        }
