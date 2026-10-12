# src/backtest/optimize.py
"""core_overlay のパラメータを上限付きランダムサーチ(早期打ち切り付き)で選ぶ。"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.allocators.core_overlay import CoreOverlayAllocator, CoreOverlayParams
from src.backtest.engine import run_backtest
from src.signals.features import compute_daily_crowding_score

# 過剰適合を抑えるため意図的に狭い範囲だけを探索する
SEARCH_SPACE: dict[str, list] = {
    "ma_fast": [90, 110, 126],
    "ma_slow": [180, 220, 252],
    "qqq_w": [0.85, 0.90],
    "bear_tlt_w": [0.35, 0.45, 0.55],
    "bear_gld_w": [0.15, 0.25, 0.35],
    "crowd_q": [0.80, 0.85, 0.90],
    "overlay_scale_k": [0.25, 0.35, 0.45],
    "lev_cap": [1.10, 1.20],
}
# overlay_scale_k <= 0.45 なので上限1.0は拘束せず、オーバーレイの大きさは k と lev_cap だけで決まる
FIXED_PARAMS = {"mom_lb": 126, "overlay_gross_cap": 1.0}


@dataclass
class SearchResult:
    params: CoreOverlayParams
    score: float
    n_evaluated: int
    split_date: pd.Timestamp


def selection_score(port_ret: pd.Series, spy_ret: pd.Series, turnover: pd.Series) -> float:
    """
    選択目的関数。
    アクティブリターンとアルファを優遇し、ベータの1からの乖離・ドローダウン・回転率を減点する。
    """
    if len(port_ret) < 120:
        return -math.inf
    x = spy_ret.to_numpy()
    y = port_ret.to_numpy()
    var = float(np.mean((x - x.mean()) ** 2))
    if var <= 0:
        return -math.inf

    beta = float(np.mean((x - x.mean()) * (y - y.mean())) / var)
    alpha_daily = y.mean() - beta * x.mean()
    resid = y - (alpha_daily + beta * x)
    te_ann = float(np.std(resid, ddof=1) * np.sqrt(252))
    alpha_ann = float(alpha_daily * 252)
    ir = alpha_ann / te_ann if te_ann > 0 else 0.0

    active_mean_ann = float((y - x).mean() * 252)
    equity = np.cumprod(1.0 + y)
    mdd = float((equity / np.maximum.accumulate(equity) - 1.0).min())

    return (
        1.20 * active_mean_ann
        + 0.20 * alpha_ann
        + 0.10 * ir
        - 0.25 * abs(beta - 1.0)
        - 0.15 * abs(mdd)
        - 0.20 * float(turnover.mean())
    )


def random_search_core_overlay(
    prices: pd.DataFrame,
    dollar_vol: pd.DataFrame,
    cost_bps: float = 10.0,
    n_candidates: int = 200,
    patience: int = 60,
    test_frac: float = 0.20,
    seed: int = 42,
    progress_every_sec: float = 2.0,
) -> SearchResult:
    """
    先頭 (1 - test_frac) を選択期間として候補を評価し、末尾 test_frac は評価に使わない。
    シグナルはすべて因果的(ローリング・拡張窓・前日ウェイト)なので、全期間で一度だけ計算し
    選択期間の行だけを採点しても末尾期間の情報は混入しない。
    """
    crowd_score = compute_daily_crowding_score(prices, dollar_vol, CoreOverlayAllocator().tickers)
    thresholds = {q: crowd_score.expanding(min_periods=252).quantile(q) for q in SEARCH_SPACE["crowd_q"]}

    split_date = prices.index[int(len(prices) * (1.0 - test_frac))]
    in_train = prices.index < split_date

    rng = np.random.default_rng(seed)
    n_total = min(n_candidates, math.prod(len(v) for v in SEARCH_SPACE.values()))
    seen: set[tuple] = set()
    best: CoreOverlayParams | None = None
    best_score = -math.inf
    no_improve = 0
    t0 = last = time.time()

    print(f"\n=== RANDOM SEARCH (train: {prices.index[0].date()} -> {prices.index[in_train][-1].date()}) ===")
    while len(seen) < n_total:
        draw = {k: rng.choice(v).item() for k, v in SEARCH_SPACE.items()}
        key = tuple(draw.values())
        if key in seen:
            continue
        seen.add(key)

        params = CoreOverlayParams(**FIXED_PARAMS, **draw)
        weights = CoreOverlayAllocator(params).weights_from_crowding(prices, crowd_score, thresholds[params.crowd_q])
        bt = run_backtest(prices, weights, cost_bps=cost_bps).loc[in_train]
        score = selection_score(bt["port_ret"], bt["spy_ret"], bt["turnover"])

        if score > best_score:
            best, best_score, no_improve = params, score, 0
        else:
            no_improve += 1

        now = time.time()
        if now - last >= progress_every_sec:
            print(f"[TRAIN] {len(seen)}/{n_total}  best_score={best_score:.4f}  elapsed={now - t0:.1f}s")
            last = now
        if no_improve >= patience:
            print(f"[EARLY STOP] {patience} 候補連続で改善がないため打ち切りました。")
            break

    if best is None:
        raise RuntimeError("ランダムサーチで評価できた候補がありません。")
    print(f"[DONE] {len(seen)} 候補を評価 ({time.time() - t0:.1f}s)  best_score={best_score:.4f}")
    return SearchResult(params=best, score=best_score, n_evaluated=len(seen), split_date=split_date)
