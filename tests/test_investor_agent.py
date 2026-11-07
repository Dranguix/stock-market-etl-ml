"""Reglas del libro de $50, scorecard WoE y lectura del ranking ya guardado."""
from __future__ import annotations

from src.agent.book import Book
from src.agent.investor import AgenteInversor
from src.agent.policy import PoliticaInversor
from src.agent.woe_obs import bin_of, scorecard_probability


def test_scorecard_en_la_tasa_base_si_no_hay_woe():
    assert abs(scorecard_probability([], 0.492) - 0.492) < 1e-6


def test_scorecard_sube_si_el_woe_es_positivo():
    base = scorecard_probability([], 0.5)
    alto = scorecard_probability([0.4], 0.5)
    assert alto > base


def test_bin_de_hoy_cae_en_el_intervalo():
    bins = [
        {"label": "bajo", "lo": 0, "hi": 10, "woe": -0.2},
        {"label": "alto", "lo": 10, "hi": 20, "woe": 0.3},
    ]
    assert bin_of(3, bins)["label"] == "bajo"
    assert bin_of(15, bins)["label"] == "alto"
    assert bin_of(25, bins)["label"] == "alto"


def test_guard_rechaza_pasarse_de_50():
    book = Book(payload={"cash_usd": 50, "budget_usd": 50, "positions": {}, "fills": []})
    out = PoliticaInversor().guard({
        "allocations": [
            {"ticker": "WALMEX", "usd": 25},
            {"ticker": "SLV", "usd": 25},
            {"ticker": "INTC", "usd": 10},
        ],
    }, book)
    assert out["ok"] is False


def test_guard_acepta_un_reparto_que_suma_50():
    book = Book(payload={"cash_usd": 50, "budget_usd": 50, "positions": {}, "fills": []})
    out = PoliticaInversor().guard({
        "allocations": [
            {"ticker": "WALMEX", "usd": 20},
            {"ticker": "SLV", "usd": 15},
        ],
        "cash_usd": 15,
        "motivo": "dos nombres",
    }, book)
    assert out["ok"] is True
    assert out["cash_usd"] == 15


def test_libro_fraccionario_no_exige_acciones_enteras():
    book = Book(payload={"cash_usd": 50, "budget_usd": 50, "positions": {}, "fills": []})
    book.apply(
        {"allocations": [{"ticker": "SLV", "usd": 15, "stop_usd": 20}]},
        {"SLV": 28.4},
        today="2026-10-09",
    )
    assert book.positions["SLV"]["shares"] < 1
    assert abs(book.cash - 35) < 0.01


def test_pregunta_ofrece_tools_y_no_pega_el_historico():
    book = Book(payload={"cash_usd": 50, "budget_usd": 50, "positions": {}, "fills": []})
    agent = AgenteInversor(book)
    pregunta = agent.observe()
    names = [t["function"]["name"] for t in pregunta["tools"]]
    assert "ver_woe" in names
    assert "ver_modelos" in names
    assert "ver_horizontes" in names
    assert "proponer_asignacion" in names
    assert len(pregunta["user"]) < 800


def test_ranking_bmv_lee_la_senal_guardada():
    book = Book(payload={"cash_usd": 50, "budget_usd": 50, "positions": {}, "fills": []})
    agent = AgenteInversor(book)
    out = agent.dispatch("ver_ranking", {"universo": "bmv", "horizonte": "4s"})
    if not out.get("ok"):
        return
    assert out["top"]
    assert "ticker" in out["top"][0]
    assert "razones" in out["top"][0]
