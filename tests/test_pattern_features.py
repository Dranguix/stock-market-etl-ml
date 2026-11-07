"""Patrones de 3, 20 y 60 días."""
from __future__ import annotations

import pandas as pd

from src.ml.pattern_features import pattern_frame


def test_racha_y_patron_de_tres_dias():
    # 10 subidas seguidas y luego una baja: la racha pasa de +10 a -1
    close = pd.Series([100 + i for i in range(12)] + [110], index=pd.bdate_range("2020-01-01", periods=13))
    out = pattern_frame(close, horizon=2)
    assert out["racha"].iloc[10] == 10
    assert out["racha"].iloc[12] == -1
    # últimos tres cierres antes del último tramo: el código +++ es 7
    assert out["pat3"].dropna().iloc[-2] == 7


def test_dias_verdes_en_20_no_pasa_de_20():
    close = pd.Series(range(100, 160), index=pd.bdate_range("2020-01-01", periods=60), dtype=float)
    out = pattern_frame(close, horizon=5)
    assert out["up_days_20"].dropna().iloc[-1] <= 20
    assert out["up_days_20"].dropna().iloc[-1] >= 19
