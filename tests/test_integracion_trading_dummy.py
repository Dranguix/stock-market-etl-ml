"""Pruebas offline para el Trading Agent. No ejecutan Claude ni PostgreSQL."""
from __future__ import annotations

import json
from datetime import date

import pytest

from src.agent import trading, tools
from src.agent.book import Book
from src.db import lab_reader


def _datos_snapshot() -> dict:
    return {
        "date": "2026-10-31",
        "universe": "bmv",
        "bets": [
            {"ticker": "POSADAS", "direction": "LONG", "horizon_days": 20},
            {"ticker": "GMEXICOB", "direction": "LONG", "horizon_days": 20},
        ],
        "top10_ranking": [
            {"rank": 1, "ticker": "QQQ", "conviction": 0.8},
            {"rank": 2, "ticker": "POSADAS", "conviction": 0.6},
            {"rank": 3, "ticker": "WALMEX", "conviction": 0.5},
        ],
    }


def test_senal_historica_filtra_universo_y_long(monkeypatch, tmp_path):
    hist = tmp_path / "history"
    hist.mkdir()
    (hist / "2026-10-31.json").write_text(
        json.dumps(_datos_snapshot()), encoding="utf-8"
    )
    monkeypatch.setattr(tools, "SIGNALS", tmp_path)
    monkeypatch.setattr(tools, "SLUGS", {
        "bmv": ("BMV", {"tickers": {
            "POSADAS": "POSADAS.MX", "WALMEX": "WALMEX.MX", "GMEXICOB": "GMEXICOB.MX"
        }}),
        "usa": ("USA", {"tickers": {"QQQ": "QQQ"}}),
    })

    agente = tools.AgenteRanking()
    resultado = agente.observe(universo="bmv", horizonte="4s")
    assert resultado["ok"] is True
    assert [r["ticker"] for r in resultado["top"]] == ["POSADAS", "WALMEX"]
    assert agente.comprables(resultado) == ["POSADAS"]
    assert "QQQ" not in agente.comprables(resultado)
    # GMEXICOB no está en el top10, así que tampoco se autoriza por solo estar en bets.
    assert "GMEXICOB" not in agente.comprables(resultado)
    assert agente.observe(universo="bmv", horizonte="1s")["ok"] is False
    assert tools._signal("usa") == {}  # no hereda señales de BMV


def test_no_hay_escritura_dummy_automatica(monkeypatch, tmp_path):
    monkeypatch.setenv("LAB_FUENTE", "dummy")
    monkeypatch.delenv("LAB_DUMMY_WRITE", raising=False)
    monkeypatch.setenv("LAB_DUMMY_DIR", str(tmp_path / "sandbox"))
    assert trading._destino_lab_dummy() is None
    assert not (tmp_path / "sandbox").exists()


def test_se_prohibe_escribir_en_los_fixtures_originales(monkeypatch, tmp_path):
    carpeta = tmp_path / "dummy_original"
    monkeypatch.setattr(trading, "_DUMMY_DIR", carpeta)
    monkeypatch.setenv("LAB_FUENTE", "dummy")
    monkeypatch.setenv("LAB_DUMMY_WRITE", "1")
    monkeypatch.setenv("LAB_DUMMY_DIR", str(carpeta))
    with pytest.raises(ValueError, match="No se permite escribir"):
        trading._destino_lab_dummy()


def test_aplicar_en_sandbox_y_leer_por_esquema_lab(monkeypatch, tmp_path):
    carpeta = tmp_path / "sandbox"
    monkeypatch.setenv("LAB_FUENTE", "dummy")
    monkeypatch.setenv("LAB_DUMMY_WRITE", "1")
    monkeypatch.setenv("LAB_DUMMY_DIR", str(carpeta))
    monkeypatch.setattr(trading, "_prices_for", lambda _p: {"WALMEX": 2.0})

    book = Book(payload={"cash_usd": 50, "budget_usd": 50, "positions": {}, "fills": []})
    agente = trading.AgenteTrading(book, hoy=date(2026, 11, 2))
    agente.policy.vistos["WALMEX"] = "2026-10-31"  # solo simulación fechada
    resultado = agente.aplicar({
        "allocations": [{"ticker": "WALMEX", "usd": 20}],
        "cash_usd": 30,
        "motivo": "prueba local sin Claude",
    })
    assert resultado["ok"] is True
    assert resultado["lab_dummy"]["estado"] == "registrado"
    assert book.positions["WALMEX"]["usd"] == 20

    fuente = lab_reader.fuente()
    assert fuente.es_dummy is True
    decisiones = fuente.decisiones()
    assert len(decisiones) == 1
    assert decisiones[0]["cash_usd"] == 30
    assert decisiones[0]["raw"]["book_snapshot"]["posiciones"]["WALMEX"]["usd"] == 20
    planes = fuente.planes([decisiones[0]["id"]])
    assert len(planes) == 1
    assert set(planes[0]) == set(lab_reader.TABLAS["entry_plan"])
    assert planes[0]["stop_usd"] is None
    assert planes[0]["p_sube"] is None


def test_signal_salta_snapshot_corrupto_y_usa_anterior_valido(monkeypatch, tmp_path):
    hist = tmp_path / "history"
    hist.mkdir()
    anterior = _datos_snapshot()
    anterior["date"] = "2026-10-30"
    (hist / "2026-10-30.json").write_text(json.dumps(anterior), encoding="utf-8")
    (hist / "2026-10-31.json").write_text("{json corrupto", encoding="utf-8")

    monkeypatch.setattr(tools, "SIGNALS", tmp_path)
    monkeypatch.setattr(tools, "SLUGS", {
        "bmv": ("BMV", {"tickers": {"POSADAS": "POSADAS.MX", "WALMEX": "WALMEX.MX"}}),
    })

    agente = tools.AgenteRanking()
    resultado = agente.observe(universo="bmv", horizonte="4s")
    assert resultado["ok"] is True
    assert resultado["fecha_senal"] == "2026-10-30"


def test_append_to_lab_dummy_usa_horizonte_por_defecto(tmp_path):
    validada = {"allocations": [{"ticker": "WALMEX", "usd": 20}], "cash_usd": 30}
    trading._append_to_lab_dummy(validada, {"libro": {}}, tmp_path, date(2026, 11, 2))
    linea = (tmp_path / "entry_plan.jsonl").read_text(encoding="utf-8").strip()
    plan = json.loads(linea)
    assert plan["horizon"] == "20d"

