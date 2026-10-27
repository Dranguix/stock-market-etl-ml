"""El agente de trading solo consulta a los agentes de Features/ML."""
from __future__ import annotations

import json

import pytest

from src.agent import features_ml, investor
from src.agent.base import ObservationAgent
from src.agent.book import Book
from src.agent.trading import AgenteTrading, tickers_en


class RankingFalso(ObservationAgent):
    name = "ranking_falso"
    description = "Ranking de prueba."

    def observe(self, **params) -> dict:
        return {"ok": True, "top": [{"puesto": 1, "ticker": "WALMEX", "score": 0.7}], "evitar": ["SLV"]}

    def tool_schema(self) -> dict:
        return {"name": self.name, "description": self.description, "input_schema": {"type": "object", "properties": {}}}


def _book() -> Book:
    return Book(payload={"cash_usd": 50, "budget_usd": 50, "positions": {}, "fills": []})


@pytest.fixture
def agente(monkeypatch) -> AgenteTrading:
    monkeypatch.setattr(features_ml, "FEATURES_ML", [RankingFalso])
    return AgenteTrading(_book())


def _call(name: str, args: dict, i: int) -> dict:
    return {"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def _guion(monkeypatch, pasos: list[list[dict]]) -> list[list[dict]]:
    """Claude falso: cada respuesta trae los tool calls del paso; guarda lo que recibió."""
    recibido: list[list[dict]] = []

    def chat(messages, tools, model=None):
        recibido.append(list(messages))
        calls = pasos.pop(0) if pasos else []
        message = {"role": "assistant", "content": None}
        if calls:
            message["tool_calls"] = calls
        return {"choices": [{"finish_reason": "tool_calls" if calls else "stop", "message": message}]}

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(investor, "chat_with_tools", chat)
    return recibido


def test_registro_real_trae_ranking_y_woe():
    assert list(features_ml.build_features_ml_tools()) == ["ver_ranking", "ver_woe"]


def test_claude_solo_ve_agentes_de_features_ml(agente):
    names = [t["function"]["name"] for t in agente.observe()["tools"]]
    assert names == ["ranking_falso", "proponer_asignacion"]
    assert "ranking_falso" in agente.observe()["system"]


def test_dispatch_bloquea_agentes_fuera_del_registro(agente):
    for name in ("ver_noticias", "ver_libro", "ver_salida", "ver_clima"):
        out = agente.dispatch(name, {"ticker": "WALMEX"})
        assert out["ok"] is False
        assert "Features/ML" in out["error"]
    assert agente.policy.vistos == set()


def test_tickers_en_ignora_listas_sin_llave_ticker():
    assert tickers_en({"top": [{"ticker": "walmex"}], "evitar": ["SLV"]}) == {"WALMEX"}


def test_compra_lo_que_devolvio_features_ml(agente, monkeypatch):
    _guion(monkeypatch, [
        [_call("ranking_falso", {}, 1)],
        [_call("proponer_asignacion", {"allocations": [{"ticker": "WALMEX", "usd": 20}], "cash_usd": 30}, 2)],
    ])
    out = agente.act()
    assert out["ok"] is True
    assert out["asignacion"]["allocations"][0]["ticker"] == "WALMEX"
    assert out["tickers_vistos"] == ["WALMEX"]


def test_rechaza_ticker_que_no_vino_de_features_ml(agente, monkeypatch):
    recibido = _guion(monkeypatch, [
        [_call("ranking_falso", {}, 1)],
        [_call("proponer_asignacion", {"allocations": [{"ticker": "SLV", "usd": 20}]}, 2)],
        [_call("proponer_asignacion", {"allocations": [], "cash_usd": 50, "motivo": "efectivo"}, 3)],
    ])
    out = agente.act()
    rechazo = json.loads(recibido[2][-1]["content"])
    assert rechazo["ok"] is False
    assert "SLV" in rechazo["error"]
    assert out["ok"] is True
    assert out["asignacion"]["cash_usd"] == 50


def test_tool_inventada_no_cuenta_como_vista(agente, monkeypatch):
    recibido = _guion(monkeypatch, [
        [_call("ver_noticias", {"ticker": "SLV"}, 1)],
        [_call("proponer_asignacion", {"allocations": [{"ticker": "SLV", "usd": 20}]}, 2)],
    ])
    out = agente.act()
    assert json.loads(recibido[1][-1]["content"])["ok"] is False
    assert out["ok"] is False


def test_registro_rechaza_nombres_repetidos(monkeypatch):
    monkeypatch.setattr(features_ml, "FEATURES_ML", [RankingFalso, RankingFalso])
    with pytest.raises(ValueError):
        features_ml.build_features_ml_tools()
