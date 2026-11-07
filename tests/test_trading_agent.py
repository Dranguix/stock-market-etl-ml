"""El agente de trading solo consulta a los agentes de Features/ML."""
from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

import src.agent.trading as trading
from src.agent import features_ml, investor
from src.agent.base import ObservationAgent
from src.agent.book import Book
from src.agent.tools import AgenteRanking, AgenteWoe
from src.agent.trading import MAX_DIAS_SENAL, AgenteTrading, consultar, dias_habiles

HOY = date(2026, 10, 28)  # miércoles


class RankingFalso(ObservationAgent):
    name = "ranking_falso"
    description = "Ranking de prueba."
    fecha = HOY.isoformat()

    def observe(self, **params) -> dict:
        return {
            "ok": True,
            "fecha_senal": self.fecha,
            "top": [{"puesto": 1, "ticker": "WALMEX", "score": 0.7}],
            "evitar": ["MEGACPO"],
        }

    def tool_schema(self) -> dict:
        return {"name": self.name, "description": self.description, "input_schema": {"type": "object", "properties": {}}}

    def comprables(self, resultado: dict) -> list[str]:
        return [p["ticker"] for p in resultado["top"]]

    def evitados(self, resultado: dict) -> list[str]:
        return resultado["evitar"]


class WoeFalso(ObservationAgent):
    """Como AgenteWoe: devuelve el ticker que le pidieron y no autoriza nada."""

    name = "ver_woe"
    description = "WoE de prueba."

    def observe(self, ticker: str = "", **params) -> dict:
        return {"ok": True, "ticker": ticker, "p_sube_5d": 0.5}

    def tool_schema(self) -> dict:
        return {"name": self.name, "description": self.description, "input_schema": {"type": "object", "properties": {}}}


def _book(**posiciones) -> Book:
    return Book(payload={"cash_usd": 50, "budget_usd": 50, "positions": posiciones, "fills": []})


def _posicion(usd: float, opened_at: str, **extra) -> dict:
    return {"usd": usd, "shares": 1.0, "price_usd": usd, "opened_at": opened_at, **extra}


@pytest.fixture
def agente(monkeypatch) -> AgenteTrading:
    monkeypatch.setattr(features_ml, "FEATURES_ML", [RankingFalso, WoeFalso])
    monkeypatch.setattr(investor, "_prices_for", lambda proposal: {})
    return AgenteTrading(_book(), hoy=HOY)


def _call(name: str, args: dict, i: int) -> dict:
    return {"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def _guion(monkeypatch, pasos: list[list[dict]], texto: str | None = None) -> list[list[dict]]:
    """Claude falso: cada respuesta trae los tool calls del paso; guarda lo que recibió."""
    recibido: list[list[dict]] = []

    def chat(messages, tools, model=None):
        recibido.append(list(messages))
        calls = pasos.pop(0) if pasos else []
        message = {"role": "assistant", "content": None if calls else texto}
        if calls:
            message["tool_calls"] = calls
        return {"choices": [{"finish_reason": "tool_calls" if calls else "stop", "message": message}]}

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(investor, "chat_with_tools", chat)
    return recibido


def _ultimo_tool(recibido: list[list[dict]]) -> dict:
    return json.loads(recibido[-1][-1]["content"])


def test_registro_real_trae_ranking_y_woe():
    assert list(features_ml.build_features_ml_tools()) == ["ver_ranking", "ver_woe"]


def test_registro_rechaza_nombres_repetidos(monkeypatch):
    monkeypatch.setattr(features_ml, "FEATURES_ML", [RankingFalso, RankingFalso])
    with pytest.raises(ValueError):
        features_ml.build_features_ml_tools()


def test_orquestador_descubre_agentes_sin_conocer_clases():
    assert features_ml.SUBDOMINIO == "features_ml"
    assert [a["name"] for a in features_ml.describir_agentes()] == ["ver_ranking", "ver_woe"]
    assert trading.SUBDOMINIO == "trading"
    assert [a["name"] for a in trading.describir_agentes()] == ["trading"]
    assert json.dumps(features_ml.describir_agentes() + trading.describir_agentes())


def test_claude_solo_ve_agentes_de_features_ml(agente):
    names = [t["function"]["name"] for t in agente.observe()["tools"]]
    assert names == ["ranking_falso", "ver_woe", "proponer_asignacion"]
    assert "ranking_falso" in agente.observe()["system"]


def test_dispatch_bloquea_agentes_fuera_del_registro(agente):
    for name in ("ver_noticias", "ver_libro", "ver_salida", "ver_clima"):
        out = agente.dispatch(name, {"ticker": "WALMEX"})
        assert out["ok"] is False
        assert "Features/ML" in out["error"]
    assert agente.policy.vistos == {}
    assert agente.consultados == []


def test_el_ranking_real_solo_autoriza_el_top():
    ranking = AgenteRanking()
    resultado = {"top": [{"ticker": "WALMEX"}], "evitar": ["MEGACPO"]}
    assert ranking.comprables(resultado) == ["WALMEX"]
    assert ranking.evitados(resultado) == ["MEGACPO"]
    assert AgenteWoe().comprables({"ticker": "MEGACPO"}) == []


def test_compra_lo_que_devolvio_features_ml(agente, monkeypatch):
    _guion(monkeypatch, [
        [_call("ranking_falso", {}, 1)],
        [_call("proponer_asignacion", {"allocations": [{"ticker": "WALMEX", "usd": 20}], "cash_usd": 30}, 2)],
    ])
    out = consultar("Reparte el libro", agente)
    assert out["ok"] is True
    assert out["asignacion"]["allocations"][0]["ticker"] == "WALMEX"
    assert out["agentes_consultados"] == ["ranking_falso"]


def test_ver_woe_no_vuelve_comprable_un_ticker_de_evitar(agente, monkeypatch):
    recibido = _guion(monkeypatch, [
        [_call("ver_woe", {"ticker": "MEGACPO"}, 1)],
        [_call("proponer_asignacion", {"allocations": [{"ticker": "MEGACPO", "usd": 20}], "cash_usd": 30}, 2)],
    ], texto="No pude comprar.")
    out = consultar("Reparte el libro", agente)
    assert "MEGACPO" in _ultimo_tool(recibido)["error"]
    assert out["asignacion"] is None


def test_guard_rechaza_lista_evitar_aunque_salga_en_otro_ranking(agente):
    agente.dispatch("ranking_falso", {})
    agente.policy.vistos["MEGACPO"] = HOY.isoformat()
    out = agente.policy.guard({"allocations": [{"ticker": "MEGACPO", "usd": 20}]}, agente.book)
    assert out["ok"] is False
    assert "evitar" in out["error"]


def test_tool_inventada_no_cuenta_como_vista(agente, monkeypatch):
    recibido = _guion(monkeypatch, [
        [_call("ver_noticias", {"ticker": "SLV"}, 1)],
        [_call("proponer_asignacion", {"allocations": [{"ticker": "SLV", "usd": 20}]}, 2)],
    ])
    out = consultar("Reparte el libro", agente)
    assert _ultimo_tool(recibido)["ok"] is False
    assert out["ok"] is False
    assert out["agentes_consultados"] == []


def test_dias_habiles_no_cuenta_fin_de_semana():
    assert dias_habiles("2026-10-23", date(2026, 10, 26)) == 1
    assert dias_habiles("no-es-fecha", HOY) is None


@pytest.mark.parametrize("fecha, ok", [("2026-10-23", True), ("2026-10-22", False), ("2026-06-28", False)])
def test_guard_rechaza_senal_vieja(agente, fecha, ok):
    assert dias_habiles("2026-10-23", HOY) == MAX_DIAS_SENAL
    agente.policy.vistos["WALMEX"] = fecha
    out = agente.policy.guard({"allocations": [{"ticker": "WALMEX", "usd": 20}]}, agente.book)
    assert out["ok"] is ok
    if not ok:
        assert fecha in out["error"]


@pytest.mark.parametrize("fecha", [None, "basura", (HOY + timedelta(days=7)).isoformat()])
def test_guard_rechaza_senal_sin_fecha_valida(agente, fecha):
    agente.policy.vistos["WALMEX"] = fecha
    out = agente.policy.guard({"allocations": [{"ticker": "WALMEX", "usd": 20}]}, agente.book)
    assert out["ok"] is False


def test_senal_vieja_deja_quedarse_en_efectivo(agente):
    agente.policy.vistos["WALMEX"] = "2026-06-28"
    out = agente.policy.guard({"allocations": [], "cash_usd": 50}, agente.book)
    assert out["ok"] is True


def test_senal_vieja_deja_mantener_lo_que_ya_esta_en_el_libro(monkeypatch):
    monkeypatch.setattr(features_ml, "FEATURES_ML", [RankingFalso, WoeFalso])
    agente = AgenteTrading(_book(WALMEX=_posicion(20, "2026-09-01")), hoy=HOY)
    agente.policy.evitados.add("WALMEX")
    assert agente.policy.guard({"allocations": [{"ticker": "WALMEX", "usd": 20}]}, agente.book)["ok"] is True
    assert agente.policy.guard({"allocations": [{"ticker": "WALMEX", "usd": 25}]}, agente.book)["ok"] is False


def test_el_schema_no_ofrece_stop(agente):
    tools = {t["function"]["name"]: t["function"] for t in agente.observe()["tools"]}
    items = tools["proponer_asignacion"]["parameters"]["properties"]["allocations"]["items"]
    assert "stop_usd" not in items["properties"]
    assert "stop" not in agente.observe()["system"]


def test_el_stop_que_llegue_se_descarta(agente):
    agente.dispatch("ranking_falso", {})
    out = agente.policy.guard({"allocations": [{"ticker": "WALMEX", "usd": 20, "stop_usd": 999}]}, agente.book)
    assert out["ok"] is True
    assert out["allocations"][0]["stop_usd"] is None


def test_un_stop_no_salta_los_dias_minimos(monkeypatch):
    monkeypatch.setattr(features_ml, "FEATURES_ML", [RankingFalso, WoeFalso])
    hoy = date.today().isoformat()
    book = _book(WALMEX=_posicion(20, hoy, stop_usd=999))
    agente = AgenteTrading(book)
    agente.dispatch("ranking_falso", {})
    out = agente.policy.guard({"allocations": [{"ticker": "WALMEX", "usd": 10}]}, book, {"WALMEX": 1.0})
    assert out["ok"] is False
    assert "stop" not in out["error"]


def test_consultar_devuelve_json_sin_transcript(agente, monkeypatch):
    _guion(monkeypatch, [
        [_call("ranking_falso", {}, 1)],
        [_call("proponer_asignacion", {"allocations": [{"ticker": "WALMEX", "usd": 20}], "cash_usd": 30, "motivo": "top"}, 2)],
    ])
    out = json.loads(json.dumps(consultar("Reparte el libro", agente)))
    assert set(out) == {"ok", "respuesta", "asignacion", "agentes_consultados", "error"}
    assert out["error"] is None
    assert out["asignacion"]["motivo"] == "top"


def test_consultar_pasa_la_pregunta_y_contesta_texto(agente, monkeypatch):
    recibido = _guion(monkeypatch, [[_call("ranking_falso", {}, 1)]], texto="Arriba va WALMEX.")
    out = consultar("¿Quién va arriba?", agente)
    assert "¿Quién va arriba?" in recibido[0][1]["content"]
    assert out == {
        "ok": True,
        "respuesta": "Arriba va WALMEX.",
        "asignacion": None,
        "agentes_consultados": ["ranking_falso"],
        "error": None,
    }


def test_consultar_no_escribe_el_libro(monkeypatch, tmp_path):
    monkeypatch.setattr(features_ml, "FEATURES_ML", [RankingFalso, WoeFalso])
    ruta = tmp_path / "book.json"
    agente = AgenteTrading(Book(ruta), hoy=HOY)
    _guion(monkeypatch, [
        [_call("ranking_falso", {}, 1)],
        [_call("proponer_asignacion", {"allocations": [{"ticker": "WALMEX", "usd": 20}], "cash_usd": 30}, 2)],
    ])
    assert consultar("Reparte el libro", agente)["ok"] is True
    assert not ruta.exists()
    assert agente.book.positions == {}


def test_aplicar_es_la_accion_que_escribe(monkeypatch, tmp_path):
    monkeypatch.setattr(features_ml, "FEATURES_ML", [RankingFalso, WoeFalso])
    monkeypatch.setattr(trading, "_prices_for", lambda proposal: {"WALMEX": 2.0})
    ruta = tmp_path / "book.json"
    agente = AgenteTrading(Book(ruta), hoy=HOY)
    _guion(monkeypatch, [
        [_call("ranking_falso", {}, 1)],
        [_call("proponer_asignacion", {"allocations": [{"ticker": "WALMEX", "usd": 20}], "cash_usd": 30}, 2)],
    ])
    asignacion = consultar("Reparte el libro", agente)["asignacion"]
    out = agente.aplicar(asignacion)
    assert out["ok"] is True
    assert ruta.exists()
    assert agente.book.positions["WALMEX"]["usd"] == 20


def test_aplicar_vuelve_a_pasar_el_guard(agente):
    out = agente.aplicar({"allocations": [{"ticker": "WALMEX", "usd": 20}]})
    assert out["ok"] is False
    assert agente.book.positions == {}


def test_consultar_sin_llave_devuelve_error(agente, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out = consultar("Reparte el libro", agente)
    assert out["ok"] is False
    assert "ANTHROPIC_API_KEY" in out["error"]
    assert json.dumps(out)


def test_consultar_atrapa_el_fallo_de_la_api(agente, monkeypatch):
    def chat(messages, tools, model=None):
        raise RuntimeError("Anthropic 500")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(investor, "chat_with_tools", chat)
    out = consultar("Reparte el libro", agente)
    assert out["ok"] is False
    assert "Anthropic 500" in out["error"]


def test_consultar_exige_pregunta():
    assert consultar("  ")["error"] == "falta la pregunta"


def test_consultar_con_libro_danado_devuelve_error(monkeypatch, tmp_path):
    ruta = tmp_path / "book.json"
    ruta.write_text("{no es json", encoding="utf-8")
    monkeypatch.setattr(trading, "BOOK_PATH", ruta)
    out = consultar("Reparte el libro")
    assert out["ok"] is False
    assert out["error"].startswith("trading fallo")
    assert ruta.read_text(encoding="utf-8") == "{no es json"


def test_cada_corrida_reinicia_lo_visto(agente, monkeypatch):
    agente.dispatch("ranking_falso", {})
    assert "WALMEX" in agente.policy.vistos
    _guion(monkeypatch, [], texto="Nada.")
    out = consultar("Hola", agente)
    assert agente.policy.vistos == {}
    assert out["agentes_consultados"] == []
