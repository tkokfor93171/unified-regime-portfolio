# src/backtest/engine.py
from __future__ import annotations

import numpy as np
import pandas as pd


def run_backtest(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    cost_bps: float = 10.0,
) -> pd.DataFrame:
    """
    前日確定ウェイト(shift(1))で当日リターンを取得し、ターンオーバーコストを控除する。
    """
    common_cols = [c for c in weights.columns if c in prices.columns]
    ret = prices[common_cols].pct_change().fillna(0.0)
    w_exec = weights[common_cols].reindex(prices.index).shift(1).fillna(0.0)

    turnover = w_exec.diff().abs().sum(axis=1).fillna(0.0)
    costs = turnover * (cost_bps / 10000.0)

    gross_ret = (w_exec * ret).sum(axis=1)
    net_ret = gross_ret - costs
    spy_ret = ret["SPY"]

    return pd.DataFrame({
        "port_ret": net_ret,
        "spy_ret": spy_ret,
        "turnover": turnover,
        "cost": costs,
    })


def compute_metrics(port_ret: pd.Series, spy_ret: pd.Series, turnover: pd.Series | None = None) -> dict:
    n_days = len(port_ret)
    if n_days < 2:
        return {}
    years = n_days / 252.0
    cum = (1.0 + port_ret).cumprod()
    total_ret = float(cum.iloc[-1] - 1.0)
    cagr = float((1.0 + total_ret) ** (1.0 / years) - 1.0) if total_ret > -1.0 else -1.0
    vol = float(port_ret.std() * np.sqrt(252))
    sharpe = float((port_ret.mean() / port_ret.std()) * np.sqrt(252)) if vol > 0 else 0.0

    dd = cum / cum.cummax() - 1.0
    mdd = float(dd.min())
    worst_5 = float(port_ret.quantile(0.05))

    # OLS Alpha / Beta / Active IR vs SPY
    cov = np.cov(port_ret, spy_ret)
    beta = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else 1.0
    alpha_daily = port_ret.mean() - beta * spy_ret.mean()
    alpha_ann = float(alpha_daily * 252)

    active_ret = port_ret - spy_ret
    te = float(active_ret.std() * np.sqrt(252))
    ir = float((active_ret.mean() * 252) / te) if te > 0 else 0.0

    res = {
        "Days": n_days,
        "Total(%)": round(total_ret * 100, 2),
        "CAGR(%)": round(cagr * 100, 2),
        "Vol(%)": round(vol * 100, 2),
        "Sharpe": round(sharpe, 2),
        "MDD(%)": round(mdd * 100, 2),
        "Worst5%(%)": round(worst_5 * 100, 2),
        "Beta": round(beta, 2),
        "Alpha(%)": round(alpha_ann * 100, 2),
        "Active_IR": round(ir, 2),
    }
    if turnover is not None:
        res["Ann_Turnover"] = round(float(turnover.mean() * 252), 2)
    return res


def evaluate_by_crowding_regime(bt_df: pd.DataFrame, crowd_series: pd.Series, q: float = 0.80) -> pd.DataFrame:
    crowd = crowd_series.reindex(bt_df.index)
    thresh = crowd.quantile(q)
    low_mask = crowd <= thresh
    high_mask = crowd > thresh

    low_m = compute_metrics(bt_df.loc[low_mask, "port_ret"], bt_df.loc[low_mask, "spy_ret"])
    high_m = compute_metrics(bt_df.loc[high_mask, "port_ret"], bt_df.loc[high_mask, "spy_ret"])
    return pd.DataFrame.from_dict({"LOW_CROWD": low_m, "HIGH_CROWD": high_m}, orient="index")
