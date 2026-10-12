"""
データを途中で切って計算したウェイトが、全期間で計算したウェイトのその日までと完全一致することを確かめる。
一致しなければ、その日より後のデータがどこかで混入している。
"""
import pandas as pd
import pytest

from src.allocators.core_overlay import CoreOverlayAllocator
from src.allocators.defensive_multi import DefensiveMultiAllocator
from src.allocators.ensemble import EnsembleAllocator
from src.backtest.engine import run_backtest

ALLOCATORS = {
    "core_overlay": CoreOverlayAllocator,
    "defensive_multi": DefensiveMultiAllocator,
    "ensemble": lambda: EnsembleAllocator(core_ratio=0.5),
    "ensemble_inv_vol": lambda: EnsembleAllocator(mode="inv_vol"),
}
# ウォームアップ中・月中・月末・閏日・年末に加え、月次ウェイトの1日前倒しを捉えるため月末の前営業日も入れる
CUTS = [
    "2006-03-15", "2008-03-31", "2009-06-15", "2010-11-01", "2011-09-29", "2012-02-29", "2014-12-31", "2016-05-17",
]


@pytest.fixture(scope="module")
def full_runs(market):
    prices, dollar_vol = market
    return {name: make().generate_daily_weights(prices, dollar_vol) for name, make in ALLOCATORS.items()}


@pytest.fixture(scope="module")
def full_inv_vol_alpha(market):
    allocator = EnsembleAllocator(mode="inv_vol")
    allocator.generate_daily_weights(*market)
    return allocator.alpha_series_


@pytest.mark.parametrize("cut", CUTS)
@pytest.mark.parametrize("name", list(ALLOCATORS))
def test_truncated_history_gives_identical_weights(market, full_runs, name, cut):
    prices, dollar_vol = market
    w_part, regime_part = ALLOCATORS[name]().generate_daily_weights(prices.loc[:cut], dollar_vol.loc[:cut])
    w_full, regime_full = full_runs[name]

    assert w_part.index[-1] == pd.Timestamp(cut)
    pd.testing.assert_frame_equal(w_part, w_full.loc[:cut], check_exact=True)
    pd.testing.assert_series_equal(regime_part, regime_full.loc[:cut], check_exact=True)


@pytest.mark.parametrize("cut", CUTS)
def test_inv_vol_ratio_does_not_use_future_data(market, full_inv_vol_alpha, cut):
    prices, dollar_vol = market
    allocator = EnsembleAllocator(mode="inv_vol")
    allocator.generate_daily_weights(prices.loc[:cut], dollar_vol.loc[:cut])
    pd.testing.assert_series_equal(allocator.alpha_series_, full_inv_vol_alpha.loc[:cut], check_exact=True)
    assert allocator.latest_alpha_ == full_inv_vol_alpha.loc[cut]


def test_synthetic_market_exercises_every_branch(full_runs, full_inv_vol_alpha):
    """合成データが各分岐を通っていないと、上の一致テストが空振りになるため確認しておく。"""
    w_core, _ = full_runs["core_overlay"]
    bull_on = w_core[["QQQ", "IWM"]].sum(axis=1) > 0
    bear_on = w_core[["TLT", "GLD"]].sum(axis=1) > 0
    overlay_on = bull_on | bear_on
    assert bull_on.any() and bear_on.any()
    assert (~overlay_on.loc[overlay_on.idxmax():]).any()

    w_def, _ = full_runs["defensive_multi"]
    risk_on = w_def.drop(columns="BIL").sum(axis=1) > 0
    assert (w_def.loc[risk_on.idxmax():, "BIL"] == 1.0).any()
    assert w_def.loc[risk_on, "BIL"].nunique() > 1

    alpha = full_inv_vol_alpha
    assert alpha.nunique() > 100
    assert alpha.min() < 0.2 and alpha.iloc[0] == 0.5


def test_engine_trades_on_the_day_after_the_signal(market):
    prices, _ = market
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    signal_day = prices.index[100]
    weights.loc[signal_day:, "SPY"] = 1.0

    bt = run_backtest(prices, weights, cost_bps=0.0)
    next_day = prices.index[101]
    assert bt.loc[signal_day, "port_ret"] == 0.0
    assert bt.loc[next_day, "turnover"] == 1.0
    pd.testing.assert_series_equal(
        bt.loc[next_day:, "port_ret"], prices["SPY"].pct_change().loc[next_day:], check_names=False, check_exact=True
    )
