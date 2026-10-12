# src/signals/features.py
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def rolling_zscore(
    series: pd.Series | pd.DataFrame, window: int, min_periods: int | None = None
) -> pd.Series | pd.DataFrame:
    if min_periods is None:
        min_periods = max(5, window // 2)
    mean = series.rolling(window=window, min_periods=min_periods).mean()
    std = series.rolling(window=window, min_periods=min_periods).std().replace(0, np.nan)
    return (series - mean) / std


def sigmoid(x: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def average_pairwise_correlation(returns: pd.DataFrame, window: int, min_periods: int) -> pd.Series:
    """
    ローリング相関行列の非対角要素(上三角)の平均。
    相関が計算できない期間は 0 埋めせず NaN のまま返す。
    """
    n = returns.shape[1]
    corr = returns.rolling(window=window, min_periods=min_periods).corr()
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    pairs = corr.to_numpy().reshape(len(returns), n, n)[:, upper]
    with warnings.catch_warnings():
        # Warm-up rows are all-NaN and trigger "Mean of empty slice"
        warnings.simplefilter("ignore", category=RuntimeWarning)
        avg = np.nanmean(pairs, axis=1)
    return pd.Series(avg, index=returns.index, name="avg_pairwise_corr")


def compute_daily_crowding_score(prices: pd.DataFrame, dollar_vol: pd.DataFrame, tickers: list[str]) -> pd.Series:
    """
    戦略①用：日次合成Crowdingスコア (0〜1)
    非流動性(0.2) + 21日ボラティリティ(0.3) + 63日ペアワイズ相関(0.5) の252日Zスコア合成
    """
    ret = prices[tickers].pct_change()
    dvol = dollar_vol[tickers].replace(0, np.nan)

    # 1. Amihud非流動性（ドル出来高欠損時は絶対リターン平均でフォールバック）
    illiq_raw = (ret.abs() / dvol).mean(axis=1)
    if illiq_raw.isna().all():
        illiq_raw = ret.abs().mean(axis=1)
    z_illiq = rolling_zscore(illiq_raw, 252).fillna(0.0)

    # 2. ユニバース平均リターンの21日標準偏差
    vol_raw = ret.mean(axis=1).rolling(21, min_periods=10).std()
    z_vol = rolling_zscore(vol_raw, 252).fillna(0.0)

    # 3. 63日窓のペアワイズ相関平均
    avg_corr = average_pairwise_correlation(ret, window=63, min_periods=30)
    z_corr = rolling_zscore(avg_corr, 252).fillna(0.0)

    composite_z = 0.2 * z_illiq + 0.3 * z_vol + 0.5 * z_corr
    return sigmoid(composite_z).rename("crowding_score").fillna(0.5)


def compute_monthly_signals(
    prices: pd.DataFrame, dollar_vol: pd.DataFrame, risk_tickers: list[str]
) -> dict[str, pd.DataFrame | pd.Series]:
    """
    戦略②用：月次モメンタム(12m-6m)、6ヶ月実現ボラ、6ヶ月相関Zスコア、月次Amihud Zスコア
    インデックスは暦月末ラベル(resample("ME"))で、データ末尾の月は途中までの値になる。
    """
    m_prices = prices[risk_tickers].resample("ME").last()
    m_ret = m_prices.pct_change()

    # 1. 中期モメンタム: (12m return - 6m return) の36ヶ月Zスコア
    ret_12m = m_prices.pct_change(12)
    ret_6m = m_prices.pct_change(6)
    mom_z = rolling_zscore(ret_12m - ret_6m, window=36, min_periods=18)

    # 2. 実現ボラティリティ (6ヶ月標準偏差・年率換算)
    realized_vol = m_ret.rolling(6, min_periods=3).std() * np.sqrt(12)

    # 3. 資産間ペアワイズ相関 (6ヶ月窓 → 36ヶ月Zスコア)
    avg_corr = average_pairwise_correlation(m_ret, window=6, min_periods=4)
    crowd_z = rolling_zscore(avg_corr, window=36, min_periods=18)

    # 4. 日次Amihudの月次平均 → 36ヶ月Zスコア
    d_ret = prices[risk_tickers].pct_change()
    d_illiq = (d_ret.abs() / dollar_vol[risk_tickers].replace(0, np.nan)).mean(axis=1)
    m_illiq = d_illiq.resample("ME").mean()
    liq_z = rolling_zscore(m_illiq, window=36, min_periods=18)

    return {
        "mom_z": mom_z,
        "realized_vol": realized_vol,
        "crowd_z": crowd_z,
        "liq_z": liq_z,
    }
