"""El score de cada horizonte ordena como se espera."""
from __future__ import annotations

import pandas as pd

from src.agent.horizons import _score


def test_mes_es_el_ano_sin_el_ultimo_mes():
    idx = pd.bdate_range("2020-01-01", periods=260)
    # Sube fuerte al principio y se estanca el último mes: el 12-1 sigue alto.
    close = pd.Series(100.0, index=idx)
    close.iloc[8] = 100
    close.iloc[-21] = 150
    close.iloc[-1] = 150
    score = _score(close, idx[-1], "mes")
    assert score is not None and score > 0.4


def test_manana_prefiere_al_que_cayo():
    idx = pd.bdate_range("2024-01-01", periods=5)
    close = pd.Series([10, 10, 10, 10, 8], index=idx, dtype=float)
    score = _score(close, idx[-1], "manana")
    assert score is not None and score > 0
