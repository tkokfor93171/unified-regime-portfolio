from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.allocators.base import BaseAllocator
from src.data.loader import DEFAULT_UNIVERSE
from src.signals.features import compute_daily_crowding_score


@dataclass
class CoreOverlayParams:
    ma_fast: int = 50
    ma_slow: int = 200
    mom_lb: int = 126
    qqq_w: float = 0.60
    bear_tlt_w: float = 0.50
    bear_gld_w: float = 0.50
    crowd_q: float = 0.80
    overlay_scale_k: float = 0.30
    overlay_gross_cap: float = 0.30
    lev_cap: float = 1.30
    rebalance_days: int = 5
    ir_lookback: int = 252
    gate_min_ir: float = 0.0


def _hold_between_rebalances(frame: pd.DataFrame | pd.Series, every_n: int) -> pd.DataFrame | pd.Series:
    """every_n 行ごとの値だけを残し、リバランス間は直前の値で保持する。"""
    keep = np.zeros(len(frame), dtype=bool)
    keep[::every_n] = True
    out = frame.copy()
    out.iloc[~keep] = np.nan
    return out.ffill().fillna(0.0)


class CoreOverlayAllocator(BaseAllocator):
    def __init__(self, params: CoreOverlayParams | None = None):
        self.p = params or CoreOverlayParams()
        self.tickers = ["SPY", "QQQ", "IWM", "TLT", "GLD"]

    def params_dict(self) -> dict:
        return asdict(self.p)

    def generate_daily_weights(
        self, prices: pd.DataFrame, dollar_vol: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        crowd_score = compute_daily_crowding_score(prices, dollar_vol, self.tickers)
        return self.weights_from_crowding(prices, crowd_score), crowd_score

    def weights_from_crowding(
        self,
        prices: pd.DataFrame,
        crowd_score: pd.Series,
        crowd_thresh: pd.Series | None = None,
    ) -> pd.DataFrame:
        """
        計算済みのCrowdingスコアからウェイトを組む。
        ランダムサーチではスコアと crowd_q ごとの閾値を使い回すためにこの入口を使う。
        """
        p = self.p
        df_p = prices[self.tickers]
        ret = df_p.pct_change().fillna(0.0)

        # 1. トレンド & モメンタム判定
        spy = df_p["SPY"]
        ma_f = spy.rolling(p.ma_fast).mean()
        ma_s = spy.rolling(p.ma_slow).mean()
        mom = spy.pct_change(p.mom_lb)

        bull_strong = (ma_f > ma_s) & (mom > 0)
        bear = ma_f <= ma_s
        tlt_ok = df_p["TLT"] > df_p["TLT"].rolling(126).mean()

        # 2. Crowdingフィルター（全期間分位点は先読みになるため拡張窓の分位点を閾値にする）
        if crowd_thresh is None:
            crowd_thresh = crowd_score.expanding(min_periods=252).quantile(p.crowd_q)
        low_crowd = crowd_score <= crowd_thresh

        # 3. 素のオーバーレイウェイト構築
        w_ov_raw = pd.DataFrame(0.0, index=df_p.index, columns=self.tickers)

        bull_mask = bull_strong & low_crowd
        w_ov_raw.loc[bull_mask, "QQQ"] = p.qqq_w
        w_ov_raw.loc[bull_mask, "IWM"] = 1.0 - p.qqq_w

        bear_mask = bear & low_crowd & tlt_ok
        w_ov_raw.loc[bear_mask, "TLT"] = p.bear_tlt_w
        w_ov_raw.loc[bear_mask, "GLD"] = p.bear_gld_w

        # 週次リバランス（5営業日ごとに更新し、間は前方埋め）
        w_ov = _hold_between_rebalances(w_ov_raw, p.rebalance_days)

        # 4. 直近252日アクティブIRゲート (先読み防止のため前日ウェイトでオーバーレイ収益を計算)
        ov_ret = (w_ov.shift(1).fillna(0.0) * ret).sum(axis=1)
        ov_mean = ov_ret.rolling(p.ir_lookback, min_periods=126).mean()
        ov_std = ov_ret.rolling(p.ir_lookback, min_periods=126).std().replace(0, np.nan)
        rolling_ir = (ov_mean / ov_std) * np.sqrt(252)

        k_raw = pd.Series(np.where(rolling_ir > p.gate_min_ir, p.overlay_scale_k, 0.0), index=df_p.index)
        k_t = _hold_between_rebalances(k_raw, p.rebalance_days)

        # 5. グロス・レバレッジ上限の適用とコア(SPY=1.0)との合成
        scaled_ov = w_ov.mul(k_t, axis=0)
        ov_gross = scaled_ov.abs().sum(axis=1).replace(0, 1.0)
        scaled_ov = scaled_ov.mul(np.minimum(1.0, p.overlay_gross_cap / ov_gross), axis=0)

        w_total = scaled_ov.copy()
        w_total["SPY"] += 1.0

        tot_gross = w_total.abs().sum(axis=1).replace(0, 1.0)
        w_total = w_total.mul(np.minimum(1.0, p.lev_cap / tot_gross), axis=0)

        # BIL列を0で追加してフォーマットを統一
        return w_total.reindex(columns=DEFAULT_UNIVERSE, fill_value=0.0)
