
import asyncio
import json

from src.agents_mcp.domains import features_ml as modulo


def consultar(pregunta):
    resultado = asyncio.run(
        modulo.query_features_ml.handler(
            {"pregunta": pregunta}
        )
    )
    return resultado["content"][0]["text"]


def test_consulta_scorecard_local(tmp_path, monkeypatch):
    archivo = tmp_path / "agent_scorecard.md"
    archivo.write_text(
        "MODELO_PRUEBA: evaluacion dummy",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        modulo, "AGENT_SCORECARD_PATH", archivo
    )

    texto = consultar("Muestra el scorecard")

    assert "MODELO_PRUEBA" in texto
    assert "datos locales offline" in texto


def test_consulta_bitacora_local(tmp_path, monkeypatch):
    archivo = tmp_path / "trial_log.jsonl"

    registro = {
        "ts": "2026-10-11T02:11:31",
        "family": "prueba_dummy",
        "hash": "abc123",
        "result": {
            "usd": {
                "libro": {
                    "retorno_pct": 1.25,
                    "sharpe": 0.85,
                }
            }
        },
    }

    archivo.write_text(
        json.dumps(registro) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        modulo, "TRIAL_LOG_PATH", archivo
    )

    texto = consultar("Muestra la bitacora")

    assert "1 corridas registradas" in texto
    assert "prueba_dummy" in texto
    assert "abc123" in texto
    assert "ret=+1.25%" in texto


def test_consulta_senales_locales(tmp_path, monkeypatch):
    historial = tmp_path / "history"
    historial.mkdir()

    snapshot = {
        "date": "2026-10-31",
        "bets": [
            {
                "ticker": "POSADAS",
                "conviction": 0.62,
                "direction": "LONG",
            }
        ],
        "top10_ranking": [
            {"ticker": "POSADAS"}
        ],
    }

    (historial / "2026-10-31.json").write_text(
        json.dumps(snapshot),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        modulo, "SIGNALS_HISTORY_DIR", historial
    )

    texto = consultar("Muestra las senales")

    assert "2026-10-31" in texto
    assert "POSADAS" in texto
    assert "1 snapshots" in texto


def test_consulta_archivos_faltantes_no_falla(tmp_path, monkeypatch):
    monkeypatch.setattr(modulo, "AGENT_SCORECARD_PATH", tmp_path / "no_existe_scorecard.md")
    monkeypatch.setattr(modulo, "TRIAL_LOG_PATH", tmp_path / "no_existe_trial.jsonl")
    monkeypatch.setattr(modulo, "SIGNALS_HISTORY_DIR", tmp_path / "no_existe_history")
    monkeypatch.setattr(modulo, "HORIZONS_PATH", tmp_path / "no_existe_horizons.json")

    texto = consultar("Resumen general")

    assert "Scorecard no generado todavía" in texto
    assert "No se encontró bitácora" in texto


def test_bitacora_ignora_lineas_json_corruptas(tmp_path, monkeypatch):
    archivo = tmp_path / "trial_log.jsonl"
    registro_valido = {
        "ts": "2026-10-12T10:00:00",
        "family": "modelo_valido",
        "hash": "def456",
        "result": {"usd": {"libro": {"retorno_pct": 2.5, "sharpe": 1.1}}},
    }
    archivo.write_text(
        "{linea_corrupta\n" + json.dumps(registro_valido) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(modulo, "TRIAL_LOG_PATH", archivo)

    texto = consultar("Muestra la bitacora")

    assert "2 corridas registradas" in texto
    assert "modelo_valido" in texto
    assert "def456" in texto


def test_recomendar_portafolio_sin_archivo_avisa_como_generarlo(tmp_path, monkeypatch):
    monkeypatch.setattr(modulo, "PORTFOLIO_COV_PATH", tmp_path / "no_existe_cov.json")

    resultado = asyncio.run(modulo.recomendar_portafolio.handler({}))
    texto = resultado["content"][0]["text"]

    assert "portfolio_optimizer_cov.py" in texto

