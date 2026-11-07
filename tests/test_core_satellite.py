from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.agent import core_satellite as cs
from src.agent import veto as veto_mod
from src.ml.trial_stats import deflated_sharpe, expected_max_sharpe, log_trial, n_trials


def _frame(prices, start="2023-01-02"):
    idx = pd.bdate_range(start, periods=len(prices))
    close = pd.Series(prices, index=idx, dtype=float)
    return pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close})


@pytest.fixture
def market(monkeypatch):
    n = 400
    rng = np.random.default_rng(0)
    sats = {f"S{i}": f"S{i}.MX" for i in range(10)}
    monkeypatch.setattr(cs, "satellite_symbols", lambda: dict(sats))
    frames = {
        "SPY": _frame(np.linspace(100, 150, n)),
        "NAFTRAC": _frame(np.linspace(60, 40, n)),
        cs.FX_SYMBOL: _frame(np.full(n, 20.0)),
    }
    for i, t in enumerate(sats):
        drift = np.linspace(0, 0.3 + 0.05 * i, n)
        frames[t] = _frame(50 * np.exp(drift + rng.normal(0, 0.002 * (i + 1), n).cumsum()))
    asof = frames["SPY"].index[-1]
    return frames, asof


def test_nucleo_bajo_tendencia_se_queda_a_la_mitad(market):
    frames, asof = market
    d = cs.decide(frames, asof, cs.Inputs(), equity_usd=50, held_satellite=[], reselect=True)
    assert d.weights["SPY"] == pytest.approx(0.50)
    assert d.weights["NAFTRAC"] == pytest.approx(0.25 * cs.CORE_BEAR_CUT)


def test_satelite_respeta_minimo_por_orden(market):
    frames, asof = market
    # 50 USD * 25% = 12.5; con 100 MXN a 20 = 5 USD por orden caben 2 nombres.
    assert cs.n_satellite(50, 20.0) == 2
    d = cs.decide(frames, asof, cs.Inputs(), equity_usd=50, held_satellite=[], reselect=True)
    assert len(d.satellite) <= 2
    assert sum(d.weights[t] for t in d.satellite) <= cs.SAT_WEIGHT + 1e-9
    assert cs.n_satellite(10, 20.0) == 0


def test_histeresis_mantiene_un_nombre_fuera_del_top20_pero_dentro_del_top40(market):
    frames, asof = market
    table = cs.factor_table(frames, asof, cs.Inputs())
    middle = table.index[int(len(table) * 0.3)]
    assert cs.ENTER_TOP < table.loc[middle, "rank_pct"] <= cs.EXIT_TOP
    d = cs.decide(frames, asof, cs.Inputs(), equity_usd=50, held_satellite=[middle], reselect=True)
    if cs.trend_asof(frames[middle]["Close"], asof):
        assert middle in d.satellite
    assert middle not in d.candidates


def test_veto_solo_quita_y_el_siguiente_toma_el_lugar(market):
    frames, asof = market
    base = cs.decide(frames, asof, cs.Inputs(), equity_usd=50, held_satellite=[], reselect=True)
    assert base.satellite
    first = base.satellite[0]
    d = cs.decide(frames, asof, cs.Inputs(), equity_usd=50, held_satellite=[], reselect=True,
                  veto=lambda cands, _: {first: "recorte de distribución"})
    assert first not in d.satellite
    assert d.vetoed == {first: "recorte de distribución"}
    assert sum(d.weights.values()) <= sum(base.weights.values()) + 1e-9


def test_banda_de_tendencia_evita_whipsaw():
    close = pd.Series(np.r_[np.full(cs.TREND_DAYS, 100.0), 99.5], index=pd.bdate_range("2024-01-01", periods=cs.TREND_DAYS + 1))
    asof = close.index[-1]
    assert cs.in_trend(close, asof, was_in=True) is True
    assert cs.in_trend(close, asof, was_in=False) is False


def test_tendencia_sin_estado_recuerda_el_mes_anterior():
    # Arriba de la media, luego dentro de la banda: sigue dentro porque el mes pasado lo estaba.
    n = cs.TREND_DAYS + 60
    prices = np.r_[np.full(cs.TREND_DAYS, 100.0), np.full(40, 110.0), np.full(20, 103.0)]
    close = pd.Series(prices, index=pd.bdate_range("2024-01-01", periods=n))
    asof = close.index[-1]
    sma = close.tail(cs.TREND_DAYS).mean()
    assert sma * (1 - cs.TREND_BAND) <= close.iloc[-1] <= sma * (1 + cs.TREND_BAND)
    assert cs.trend_asof(close, asof) is True


def test_banda_de_deriva_opera_hasta_el_borde():
    out = cs.apply_drift_band({"SPY": 0.50, "X": 0.10, "Y": 0.10}, {"SPY": 0.58, "X": 0.12, "Z": 0.05})
    assert out["SPY"] == pytest.approx(0.55)
    assert out["X"] == pytest.approx(0.12)
    assert out["Y"] == pytest.approx(0.10)
    assert "Z" not in out


def test_rebalanceo_al_cierre_del_mes():
    assert cs.is_rebalance_day(date(2026, 9, 30), date(2026, 10, 1))
    assert not cs.is_rebalance_day(date(2026, 10, 8), date(2026, 10, 9))


def test_veto_anonimiza_baraja_y_vota(monkeypatch, tmp_path):
    monkeypatch.setattr(veto_mod, "VETO_LOG", tmp_path / "veto.jsonl")
    monkeypatch.setattr(veto_mod, "_aliases", lambda t: [t, "Walmart de México"])
    news = {
        "WALMEX": ["Walmart de México recorta su dividendo", "WALMEX baja guía"],
        "FUNO11": ["FUNO11 coloca bono"],
        "AC": [],
    }
    monkeypatch.setattr(veto_mod, "headlines_for", lambda t, asof: news[t])
    prompts = []

    def fake_call(prompt, model):
        prompts.append(prompt)
        out = {}
        for block in prompt.split("\n\n"):
            if block.startswith("E") and ":" in block.splitlines()[0]:
                label = block.splitlines()[0].rstrip(":")
                out[label] = {"evento": "otro", "direccion": "neutral", "veto": "recorta" in block, "motivo": "x"}
        import json
        return json.dumps(out)

    res = veto_mod.classify(["WALMEX", "FUNO11", "AC"], pd.Timestamp("2026-10-09"), replicas=3, call=fake_call)
    assert len(prompts) == 3
    assert all("WALMEX" not in p and "Walmart de México" not in p for p in prompts)
    assert res["WALMEX"]["veto"] and res["WALMEX"]["votos"] == 3
    assert not res["FUNO11"]["veto"]
    assert "AC" not in res
    assert veto_mod.llm_veto(["WALMEX", "FUNO11"], pd.Timestamp("2026-10-09"), replicas=1, call=fake_call) == {"WALMEX": "x"}


def test_sharpe_por_azar_coincide_con_la_tabla_del_reporte():
    # Bailey et al.: N=10, 24 meses -> ~1.1; N=20, 12 meses -> ~2.0
    assert expected_max_sharpe(10, 2) == pytest.approx(1.11, abs=0.05)
    assert expected_max_sharpe(20, 1) == pytest.approx(1.99, abs=0.1)
    assert expected_max_sharpe(1, 2) == 0.0


def test_dsr_bajo_con_poca_historia_y_muchas_pruebas():
    rng = np.random.default_rng(1)
    r = rng.normal(0.0006, 0.01, 504)  # Sharpe anual ~1
    low = deflated_sharpe(r, 20)["dsr"]
    high = deflated_sharpe(r, 1)["dsr"]
    assert low < high
    assert low < 0.95


def test_bitacora_cuenta_configuraciones_distintas(tmp_path):
    path = tmp_path / "log.jsonl"
    assert log_trial("f", {"a": 1}, {}, path) == 1
    assert log_trial("f", {"a": 1}, {}, path) == 1
    assert log_trial("f", {"a": 2}, {}, path) == 2
    assert n_trials("otra", path) == 0
