"""Tests de src/ml/flow_features.py — order flow de brokers (hechos BMV)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.flow_features import (
    FLOW_FEATURE_COLS,
    add_flow_features,
    compute_flow_panel,
    load_flow_raw,
)
from scripts.backfill_hechos import aggregate_flows, normalize_ticker, parse_hechos_csv


class TestNormalizeTicker:
    def test_star_series_dropped(self):
        assert normalize_ticker("WALMEX *") == "WALMEX"
        assert normalize_ticker("BBVA   *    ") == "BBVA"

    def test_series_concatenated(self):
        assert normalize_ticker("GFNORTE O") == "GFNORTEO"
        assert normalize_ticker("FEMSA UBD") == "FEMSAUBD"

    def test_empty(self):
        assert normalize_ticker("   ") == ""


class TestParseAggregate:
    CSV = (
        b"20260626\n"
        b"EmisoraSerie;UltHecho;VolHecho;CasaCompra;CasaVende;Hora;Folio;Importe;"
        b"TipoOperacion;Concertacion;Liquidacion;TipoValor;PicoLote\n"
        b"WALMEX *    ;000060.000000;00000000100;GBM  ;ACTIN;09:00:00;0000001;"
        b"0000000006000.0000;CO;CO;48;1E;P\n"
        b"WALMEX *    ;000060.500000;00000000050;GBM  ;SANT ;09:01:00;0000002;"
        b"0000000003025.0000;CO;CO;48;1E;P\n"
        b"GFNORTE O   ;000150.000000;00000000200;ACTIN;GBM  ;09:02:00;0000003;"
        b"0000000030000.0000;CO;CO;48;1E;P\n"
    )

    def test_parse(self):
        trades = parse_hechos_csv(self.CSV)
        assert len(trades) == 3
        assert set(trades["ticker"]) == {"WALMEX", "GFNORTEO"}
        assert trades["vol"].sum() == 350

    def test_aggregate_two_sided(self):
        trades = parse_hechos_csv(self.CSV)
        agg = aggregate_flows(trades, "2026-06-26")
        gbm_walmex = agg[(agg["ticker"] == "WALMEX") & (agg["casa"] == "GBM")]
        assert gbm_walmex["vol_buy"].iloc[0] == 150
        assert gbm_walmex["vol_sell"].iloc[0] == 0
        # volumen comprado == vendido por ticker (cada hecho tiene 2 lados)
        tot = agg.groupby("ticker")[["vol_buy", "vol_sell"]].sum()
        assert (tot["vol_buy"] == tot["vol_sell"]).all()


class TestFlowPanel:
    @pytest.fixture()
    def hechos_dir(self, tmp_path):
        rng = np.random.default_rng(11)
        casas = ["GBM", "ACTIN", "SANT", "BANOR"]
        dates = pd.bdate_range("2026-01-05", periods=30)
        for d in dates:
            rows = []
            for ticker in ("WALMEX", "GFNORTEO", "ALSEA"):
                vols = rng.integers(100, 10_000, size=len(casas))
                for casa, v in zip(casas, vols):
                    rows.append({
                        "date": d.strftime("%Y-%m-%d"), "ticker": ticker, "casa": casa,
                        "vol_buy": int(v), "vol_sell": int(vols.sum() - v) // 3,
                        "imp_buy": float(v) * 50, "imp_sell": float(v) * 49,
                        "n_buy": 5, "n_sell": 5,
                    })
            pd.DataFrame(rows).to_parquet(tmp_path / f"{d.date()}.parquet", index=False)
        return tmp_path

    def test_load_raw_filters_tickers(self, hechos_dir):
        raw = load_flow_raw(["WALMEX"], hechos_dir=hechos_dir)
        assert set(raw["ticker"]) == {"WALMEX"}

    def test_panel_columns_and_bounds(self, hechos_dir):
        panel = compute_flow_panel(hechos_dir=hechos_dir)
        assert set(FLOW_FEATURE_COLS) <= set(panel.columns)
        assert panel[FLOW_FEATURE_COLS].notna().all().all()
        assert panel["flow_imbalance"].abs().max() <= 1.0 + 1e-9
        assert not panel.duplicated(subset=["date", "ticker"]).any()

    def test_persistence_is_rolling_mean(self, hechos_dir):
        panel = compute_flow_panel(hechos_dir=hechos_dir)
        sub = panel[panel["ticker"] == "WALMEX"].sort_values("date")
        manual = sub["flow_imbalance"].rolling(5, min_periods=2).mean()
        pd.testing.assert_series_equal(
            sub["flow_persistence"].iloc[2:], manual.iloc[2:],
            check_names=False,
        )

    def test_smart_flow_zero_without_closes(self, hechos_dir):
        panel = compute_flow_panel(hechos_dir=hechos_dir, close_panel=None)
        assert (panel["flow_smart"] == 0.0).all()

    def test_smart_flow_pit_with_closes(self, hechos_dir):
        dates = pd.bdate_range("2025-06-02", periods=300)
        rng = np.random.default_rng(5)
        closes = pd.DataFrame(
            {t: 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))))
             for t in ("WALMEX", "GFNORTEO", "ALSEA")},
            index=dates,
        )
        panel = compute_flow_panel(close_panel=closes, hechos_dir=hechos_dir)
        assert "flow_smart" in panel.columns
        assert panel["flow_smart"].notna().all()

    def test_add_flow_features_merge(self, hechos_dir):
        panel = compute_flow_panel(hechos_dir=hechos_dir)
        idx = pd.bdate_range("2026-01-05", periods=40)
        feat = pd.DataFrame({"Close": np.linspace(50, 60, 40)}, index=idx)
        merged = add_flow_features(feat, "WALMEX", panel)
        assert all(c in merged.columns for c in FLOW_FEATURE_COLS)
        # días sin hechos → 0.0, no NaN
        assert merged[FLOW_FEATURE_COLS].notna().all().all()

        empty = add_flow_features(feat, "NO_EXISTE", panel)
        assert (empty[FLOW_FEATURE_COLS] == 0.0).all().all()
