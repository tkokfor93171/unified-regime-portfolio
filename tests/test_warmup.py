"""--start より前のデータをシグナルのウォームアップにだけ使い、損益と成績は --start 以降で計上することを確かめる。"""
import numpy as np
import pandas as pd
import pytest

import main
from src.allocators.defensive_multi import DefensiveMultiAllocator

LOAD_START = "2005-11-23"  # 2010-01-01 の 1,500 暦日前


def slice_market(market, start_date, end_date=None):
    """yfinance と同じく start_date 以降、end_date より前の行だけを返す。"""
    prices, dollar_vol = market
    keep = prices.index >= pd.Timestamp(start_date)
    if end_date is not None:
        keep &= prices.index < pd.Timestamp(end_date)
    return prices.loc[keep], dollar_vol.loc[keep]


@pytest.fixture
def load_calls(backtest_cli, monkeypatch, market):
    calls: list[dict] = []

    def load(start_date=None, end_date=None):
        calls.append({"start_date": start_date, "end_date": end_date})
        return slice_market(market, start_date, end_date)

    monkeypatch.setattr(main, "load_market_data", load)
    return calls


def read_output(out_dir, name: str) -> pd.DataFrame:
    return pd.read_csv(out_dir / name, index_col=0, parse_dates=True)


def test_data_is_loaded_1500_calendar_days_before_start(load_calls):
    main.main(["backtest", "--strategy", "defensive_multi", "--start", "2010-01-01", "--end", "2016-01-01"])
    assert load_calls == [{"start_date": LOAD_START, "end_date": "2016-01-01"}]


def test_pnl_starts_at_start_with_warmed_up_weights(load_calls, backtest_cli, market):
    main.main(["backtest", "--strategy", "defensive_multi", "--start", "2010-01-01"])
    daily = read_output(backtest_cli, "defensive_multi_daily.csv")
    weights = read_output(backtest_cli, "defensive_multi_weights.csv")

    warm, _ = DefensiveMultiAllocator().generate_daily_weights(*slice_market(market, LOAD_START))
    cold, _ = DefensiveMultiAllocator().generate_daily_weights(*slice_market(market, "2010-01-01"))

    assert daily.index[0] == weights.index[0] == pd.Timestamp("2010-01-01")
    pd.testing.assert_frame_equal(weights, warm.loc["2010-01-01":], check_freq=False, check_exact=False, rtol=1e-12)
    # 冷えた状態から始めると初日は何も持てないが、ウォームアップ後は初日からリスク資産を持つ
    assert cold.iloc[0].sum() == 0.0
    assert weights.iloc[0].drop("BIL").sum() > 0.0
    # 損益は初日の終値で建てたポジションから計上する
    assert daily["port_ret"].iloc[0] == 0.0
    assert daily["turnover"].iloc[1] == pytest.approx(weights.iloc[0].abs().sum())


def test_metrics_cover_only_the_backtest_window(load_calls, backtest_cli, market):
    main.main(["backtest", "--strategy", "core_overlay", "--start", "2012-01-01", "--end", "2015-01-01"])
    metrics = pd.read_csv(backtest_cli / "core_overlay_metrics.csv", index_col=0)
    days = len(slice_market(market, "2012-01-01", "2015-01-01")[0])
    assert metrics.loc["Strategy", "Days"] == metrics.loc["SPY_BuyHold", "Days"] == days


def test_inv_vol_ratio_is_warmed_up_on_the_first_day(load_calls, backtest_cli):
    main.main(["backtest", "--strategy", "ensemble", "--ensemble-mode", "inv_vol", "--start", "2010-01-01"])
    daily = read_output(backtest_cli, "ensemble_invvol63_daily.csv")
    assert daily.index[0] == pd.Timestamp("2010-01-01")
    assert daily["core_ratio"].iloc[0] != 0.5


def test_default_window_prints_no_history_warning(load_calls, capsys):
    main.main(["backtest", "--strategy", "defensive_multi"])
    out = capsys.readouterr().out
    assert "signals warmed up over the preceding" in out
    assert "ウォームアップが" not in out and "価格が始まる銘柄" not in out


def test_short_history_before_start_is_warned(load_calls, capsys):
    main.main(["backtest", "--strategy", "defensive_multi", "--start", "2006-06-01"])
    assert "ウォームアップが 1500 暦日に足りません" in capsys.readouterr().out


def test_ticker_listed_after_start_is_warned(backtest_cli, monkeypatch, market, capsys):
    prices, dollar_vol = market
    prices = prices.copy()
    prices.loc[:"2010-12-31", "BIL"] = np.nan
    monkeypatch.setattr(main, "load_market_data", lambda start_date=None, end_date=None: (prices, dollar_vol))
    main.main(["backtest", "--strategy", "defensive_multi", "--start", "2010-01-01"])
    assert "評価開始日より後に価格が始まる銘柄があります: BIL (2011-01-03~)" in capsys.readouterr().out


def test_start_after_the_last_date_stops(load_calls):
    with pytest.raises(SystemExit, match="以降の価格データがありません"):
        main.main(["backtest", "--strategy", "defensive_multi", "--start", "2030-01-01"])


def test_unparseable_start_is_rejected(backtest_cli):
    with pytest.raises(SystemExit) as exc:
        main.main(["backtest", "--start", "2010-13-01"])
    assert exc.value.code == 2
