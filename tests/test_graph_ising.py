"""Tests de src/ml/graph_ising.py — dinámica, calibración y panel PIT."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.graph_ising import (
    DEFAULT_EVIDENCE_COLS,
    GIM_FEATURE_COLS,
    GraphIsingConfig,
    GraphIsingMarket,
    add_gim_features,
    compute_gim_panel,
)


@pytest.fixture()
def market() -> GraphIsingMarket:
    return GraphIsingMarket(GraphIsingConfig(n_steps=60))


def _random_returns(n_days: int, n_tickers: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    common = rng.normal(0, 0.01, n_days)
    data = {
        f"T{i}": common * (0.5 + 0.5 * rng.random()) + rng.normal(0, 0.01, n_days)
        for i in range(n_tickers)
    }
    idx = pd.bdate_range("2024-01-01", periods=n_days)
    return pd.DataFrame(data, index=idx)


class TestGraph:
    def test_build_graph_row_normalized(self, market):
        rets = _random_returns(80, 10)
        W = market.build_graph(rets)
        assert W.shape == (10, 10)
        assert np.all(np.diag(W) == 0)
        row_sums = W.sum(axis=1)
        assert np.allclose(row_sums[row_sums > 0], 1.0)

    def test_coupling_subcritical(self, market):
        rets = _random_returns(80, 10)
        W = market.build_graph(rets)
        J = market.coupling(W)
        lam = float(np.max(np.abs(np.linalg.eigvals(W)).real))
        assert J * lam / market.cfg.T == pytest.approx(market.cfg.crit_frac, abs=1e-6)


class TestDynamics:
    def test_relax_bounded_and_follows_field(self, market):
        rets = _random_returns(80, 12)
        W = market.build_graph(rets)
        J = market.coupling(W)
        h = np.full(12, 1.5)
        s = market.relax(W, h, J)
        assert np.all(np.abs(s) <= 1.0)
        assert np.all(s > 0)  # campo positivo → presión compradora

        s_neg = market.relax(W, -h, J)
        assert np.all(s_neg < 0)

    def test_herding_amplifies_weak_field(self, market):
        """Con acoplamiento, un campo débil produce |s| mayor que sin vecinos."""
        rets = _random_returns(80, 12)
        W = market.build_graph(rets)
        J = market.coupling(W)
        h = np.full(12, 0.3)
        s_coupled = market.relax(W, h, J)
        s_alone = np.tanh(h / market.cfg.T)
        assert s_coupled.mean() > s_alone.mean()

    def test_susceptibility_positive(self, market):
        rets = _random_returns(80, 12)
        W = market.build_graph(rets)
        J = market.coupling(W)
        s = market.relax(W, np.zeros(12), J)
        chi = market.susceptibility(W, s, J)
        assert np.all(chi > 0)


class TestPanel:
    def _frames(self, n_days: int = 260, n_tickers: int = 8) -> dict[str, pd.DataFrame]:
        rng = np.random.default_rng(3)
        idx = pd.bdate_range("2023-01-02", periods=n_days)
        frames = {}
        for i in range(n_tickers):
            close = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, n_days)))
            frames[f"T{i}"] = pd.DataFrame({
                "Close": close,
                "rsi_14": rng.uniform(20, 80, n_days),
                "ret_5d": rng.normal(0, 2, n_days),
                "vol_ratio_20": rng.uniform(0.5, 2.0, n_days),
            }, index=idx)
        return frames

    def test_panel_shape_and_ranges(self):
        cfg = GraphIsingConfig(woe_refit_every=40, min_train_rows=200)
        panel = compute_gim_panel(self._frames(), config=cfg)
        assert not panel.empty
        assert set(GIM_FEATURE_COLS) <= set(panel.columns)
        assert panel["gim_pressure"].abs().max() <= 1.0
        assert panel["gim_phase"].isin([-1.0, 0.0, 1.0]).all()
        # sin duplicados (date, ticker)
        assert not panel.duplicated(subset=["date", "ticker"]).any()

    def test_panel_needs_min_universe(self):
        frames = self._frames(n_tickers=3)
        panel = compute_gim_panel(frames)
        assert panel.empty

    def test_add_gim_features_merge_and_zeros(self):
        frames = self._frames()
        cfg = GraphIsingConfig(woe_refit_every=40, min_train_rows=200)
        panel = compute_gim_panel(frames, config=cfg)
        t0 = next(iter(frames))
        merged = add_gim_features(frames[t0], t0, panel)
        assert all(c in merged.columns for c in GIM_FEATURE_COLS)
        assert merged[GIM_FEATURE_COLS].notna().all().all()

        empty = add_gim_features(frames[t0], "NO_EXISTE", panel)
        assert (empty[GIM_FEATURE_COLS] == 0.0).all().all()

    def test_evidence_defaults_include_flow_and_news(self):
        assert "flow_imbalance" in DEFAULT_EVIDENCE_COLS
        assert "news_sent_score" in DEFAULT_EVIDENCE_COLS
