# src/backtest/plot.py
from __future__ import annotations

from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure


def _equity(ret: pd.Series) -> pd.Series:
    return (1.0 + ret.fillna(0.0)).cumprod()


def plot_performance(bt: pd.DataFrame, title: str, save_path: Path) -> Path:
    """累積リターン(対数軸)とドローダウンを戦略とSPYで比較し、PNGで保存する。"""
    eq = _equity(bt["port_ret"])
    spy_eq = _equity(bt["spy_ret"])
    dd = eq / eq.cummax() - 1.0
    spy_dd = spy_eq / spy_eq.cummax() - 1.0

    fig = Figure(figsize=(12, 7), layout="constrained")
    ax_eq, ax_dd = fig.subplots(2, 1, sharex=True, height_ratios=[2, 1])

    ax_eq.plot(eq.index, eq.to_numpy(), label="Strategy", linewidth=1.4)
    ax_eq.plot(spy_eq.index, spy_eq.to_numpy(), label="SPY Buy & Hold", linewidth=1.0, alpha=0.8)
    ax_eq.set_yscale("log")
    ax_eq.set_title(f"{title} - Cumulative Return (log scale)")
    ax_eq.grid(True, alpha=0.3)
    ax_eq.legend()

    ax_dd.fill_between(dd.index, dd.to_numpy(), 0.0, alpha=0.35, label="Strategy")
    ax_dd.plot(spy_dd.index, spy_dd.to_numpy(), label="SPY Buy & Hold", linewidth=1.0, color="tab:orange")
    ax_dd.set_title("Drawdown")
    ax_dd.grid(True, alpha=0.3)
    ax_dd.legend(loc="lower left")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    return save_path


def plot_regime_performance(regime_df: pd.DataFrame, title: str, save_path: Path) -> Path:
    """Crowdingレジーム別(LOW_CROWD / HIGH_CROWD)の主要指標を棒グラフで保存する。"""
    metrics = [m for m in ("CAGR(%)", "Sharpe", "MDD(%)", "Alpha(%)") if m in regime_df.columns]
    fig = Figure(figsize=(3.2 * max(len(metrics), 1), 3.8), layout="constrained")
    axes = fig.subplots(1, max(len(metrics), 1), squeeze=False)[0]

    colors = ["tab:blue" if "LOW" in str(i) else "tab:red" for i in regime_df.index]
    for ax, metric in zip(axes, metrics):
        values = regime_df[metric].astype(float)
        ax.bar([str(i) for i in values.index], values.to_numpy(), color=colors)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"{title} - Performance by Crowding Regime")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    return save_path
