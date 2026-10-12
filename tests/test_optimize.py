import json
import math
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

import main
from src.allocators.core_overlay import CoreOverlayAllocator, CoreOverlayParams
from src.backtest.engine import run_backtest
from src.backtest.optimize import SearchResult, random_search_core_overlay, selection_score, split_metrics

CANDIDATE = CoreOverlayParams(
    ma_fast=90, ma_slow=180, qqq_w=0.90, bear_tlt_w=0.55, bear_gld_w=0.35,
    crowd_q=0.90, overlay_scale_k=0.45, overlay_gross_cap=1.0, lev_cap=1.20,
)


def test_turnover_penalty_is_annualized():
    rng = np.random.default_rng(0)
    spy = pd.Series(rng.normal(0.0004, 0.01, 500))
    port = 1.1 * spy + rng.normal(0.0001, 0.002, 500)
    quiet = pd.Series(0.0, index=spy.index)
    busy = pd.Series(0.02, index=spy.index)
    assert selection_score(port, spy, quiet) - selection_score(port, spy, busy) == pytest.approx(0.20 * 0.02 * 252)


@pytest.fixture(scope="module", params=[None, "2010-01-01"], ids=["whole_data", "after_warmup"])
def search(request, market):
    prices, dollar_vol = market
    eval_start = None if request.param is None else pd.Timestamp(request.param)
    return eval_start, random_search_core_overlay(
        prices, dollar_vol, n_candidates=6, patience=6, seed=0, eval_start=eval_start
    )


def test_holdout_is_test_period_of_selected_params(market, search):
    eval_start, result = search
    prices, dollar_vol = market
    weights, _ = CoreOverlayAllocator(result.params).generate_daily_weights(prices, dollar_vol)
    window = prices if eval_start is None else prices.loc[eval_start:]
    expected = split_metrics(run_backtest(window, weights.loc[window.index[0]:]), result.split_date)

    pd.testing.assert_frame_equal(result.holdout, expected)
    # 学習期間と検証期間はウォームアップ後の評価期間を 80:20 に分ける
    assert result.split_date == window.index[int(len(window) * 0.8)]
    assert result.holdout.loc[["TRAIN_Optimized", "TEST_Optimized"], "Days"].sum() == len(window)
    assert result.holdout_alpha == expected.loc["TEST_Optimized", "Alpha(%)"]
    assert result.accepted == (result.holdout_alpha > 0)


def test_missing_holdout_is_not_accepted():
    result = SearchResult(CoreOverlayParams(), 0.0, 1, pd.Timestamp("2015-01-02"), holdout=pd.DataFrame())
    assert math.isnan(result.holdout_alpha)
    assert not result.accepted


def expected_weights(market, params: CoreOverlayParams) -> dict:
    weights, _ = CoreOverlayAllocator(params).generate_daily_weights(*market)
    return {sym: round(float(w), 4) for sym, w in weights.iloc[-1].items()}


@pytest.fixture
def run_optimize(backtest_cli, monkeypatch):
    """random_search を差し替え、検証期間アルファを指定して backtest --optimize を走らせる。"""

    def run(holdout_alpha: float, *extra: str) -> dict:
        def fake_search(prices, dollar_vol, **kwargs):
            holdout = pd.DataFrame({"Alpha(%)": [1.0, holdout_alpha]}, index=["TRAIN_Optimized", "TEST_Optimized"])
            return SearchResult(CANDIDATE, 1.0, 1, prices.index[-200], holdout=holdout)

        monkeypatch.setattr(main, "random_search_core_overlay", fake_search)
        main.main(["backtest", "--strategy", "core_overlay", "--optimize", *extra])
        return json.loads((backtest_cli / "target_weights.json").read_text(encoding="utf-8"))

    return run


@pytest.mark.parametrize("alpha", [-0.5, 0.0])
def test_non_positive_holdout_alpha_exports_default_params(run_optimize, market, capsys, alpha):
    payload = run_optimize(alpha)
    assert payload["params"] == asdict(CoreOverlayParams())
    assert payload["weights"] == expected_weights(market, CoreOverlayParams())
    out = capsys.readouterr().out
    assert "[REJECTED]" in out and "過学習" in out


def test_positive_holdout_alpha_adopts_optimized_params(run_optimize, backtest_cli, market, capsys):
    payload = run_optimize(0.5)
    assert payload["params"] == asdict(CANDIDATE)
    assert payload["weights"] == expected_weights(market, CANDIDATE)
    assert "[ADOPTED]" in capsys.readouterr().out
    metrics = pd.read_csv(backtest_cli / "core_overlay_metrics.csv", index_col=0)
    assert "TEST_Optimized" in metrics.index


def test_force_optimized_overrides_rejection(run_optimize, market, capsys):
    payload = run_optimize(-0.5, "--force-optimized")
    assert payload["params"] == asdict(CANDIDATE)
    assert payload["weights"] == expected_weights(market, CANDIDATE)
    assert "[FORCED]" in capsys.readouterr().out


def test_force_optimized_requires_optimize(backtest_cli):
    with pytest.raises(SystemExit) as exc:
        main.main(["backtest", "--strategy", "core_overlay", "--force-optimized"])
    assert exc.value.code == 2
