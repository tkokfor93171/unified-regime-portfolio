import pandas as pd
import pytest

from src.allocators.core_overlay import CoreOverlayAllocator
from src.allocators.defensive_multi import DefensiveMultiAllocator
from src.allocators.ensemble import EnsembleAllocator
from src.backtest.engine import run_backtest


@pytest.fixture(scope="module")
def singles(market):
    prices, dollar_vol = market
    return {
        "core": CoreOverlayAllocator().generate_daily_weights(prices, dollar_vol),
        "def": DefensiveMultiAllocator().generate_daily_weights(prices, dollar_vol),
    }


def test_core_ratio_one_equals_core_overlay(market, singles):
    prices, dollar_vol = market
    w, regime = EnsembleAllocator(core_ratio=1.0).generate_daily_weights(prices, dollar_vol)
    w_core, crowd_core = singles["core"]

    pd.testing.assert_frame_equal(w, w_core, check_exact=True)
    pd.testing.assert_series_equal(regime, crowd_core, check_exact=True)
    pd.testing.assert_frame_equal(run_backtest(prices, w), run_backtest(prices, w_core), check_exact=True)


def test_core_ratio_zero_equals_defensive_multi(market, singles):
    prices, dollar_vol = market
    w, regime = EnsembleAllocator(core_ratio=0.0).generate_daily_weights(prices, dollar_vol)
    w_def, _ = singles["def"]

    pd.testing.assert_frame_equal(w, w_def, check_exact=True)
    pd.testing.assert_frame_equal(run_backtest(prices, w), run_backtest(prices, w_def), check_exact=True)
    # レジーム別評価は比率によらず core_overlay の日次 Crowding スコアで行う仕様
    pd.testing.assert_series_equal(regime, singles["core"][1], check_exact=True)


@pytest.mark.parametrize("ratio", [0.3, 0.5, 0.7])
def test_blend_is_linear_combination(market, singles, ratio):
    w, _ = EnsembleAllocator(core_ratio=ratio).generate_daily_weights(*market)
    expected = ratio * singles["core"][0] + (1.0 - ratio) * singles["def"][0]
    pd.testing.assert_frame_equal(w, expected, check_exact=True)


@pytest.mark.parametrize("ratio", [-0.1, 1.1, float("nan")])
def test_invalid_core_ratio_is_rejected(ratio):
    with pytest.raises(ValueError):
        EnsembleAllocator(core_ratio=ratio)


def test_params_dict_records_both_strategies():
    params = EnsembleAllocator(core_ratio=0.7).params_dict()
    assert params == {
        "core_ratio": 0.7,
        "core_overlay": CoreOverlayAllocator().params_dict(),
        "defensive_multi": DefensiveMultiAllocator().params_dict(),
    }
