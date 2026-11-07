"""Tests for BMV catalog and SQL ticker sync."""
from __future__ import annotations

from src.data.bmv_catalog import (
    BMV_EMISORA_SERIE,
    BMV_YFINANCE,
    INTRADAY_MAP,
    sql_ticker_list,
)
from src.data.store import (
    ALL_SYMBOLS,
    BMV_TICKERS,
    GBM_ILLIQUID,
    GBM_VENUE,
    gbm_tradable,
)


def test_bmv_catalog_consistency() -> None:
    assert len(BMV_YFINANCE) == 36
    assert BMV_YFINANCE.keys() == BMV_EMISORA_SERIE.keys()
    assert INTRADAY_MAP == BMV_EMISORA_SERIE
    assert BMV_TICKERS == BMV_YFINANCE


def test_gbm_venue_marks_every_symbol() -> None:
    assert set(GBM_VENUE) == set(ALL_SYMBOLS)
    assert set(GBM_VENUE.values()) == {"bmv", "usa", "no"}
    not_an_order = {name for name, venue in GBM_VENUE.items() if venue == "no"}
    assert not_an_order == {"ALFAA", "USDMXN", "NAFTRAC", "WTI", "SP500"}
    assert gbm_tradable("WALMEX") and gbm_tradable("AAPL") and gbm_tradable("NAFTRAC_ETF")
    assert not gbm_tradable("ALFAA") and not gbm_tradable("SP500")
    assert GBM_ILLIQUID == frozenset({"HOTEL", "POSADAS"})
    assert all(gbm_tradable(name) for name in GBM_ILLIQUID)


def test_sql_ticker_list_count() -> None:
    tickers = sql_ticker_list()
    assert len(tickers) == 36
    assert tickers == sorted(set(BMV_EMISORA_SERIE.values()))
