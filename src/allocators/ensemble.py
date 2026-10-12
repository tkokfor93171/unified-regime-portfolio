from __future__ import annotations

import numpy as np
import pandas as pd

from src.allocators.base import BaseAllocator
from src.allocators.core_overlay import CoreOverlayAllocator, CoreOverlayParams
from src.allocators.defensive_multi import DefensiveMultiAllocator
from src.data.loader import DEFAULT_UNIVERSE

ENSEMBLE_MODES = ("fixed", "inv_vol")


def sleeve_returns(weights: pd.DataFrame, ret: pd.DataFrame) -> pd.Series:
    """前日確定ウェイト × 当日リターン（コスト控除前）。バックテストエンジンと同じく前日のウェイトを使う。"""
    return (weights.shift(1).fillna(0.0) * ret).sum(axis=1)


class EnsembleAllocator(BaseAllocator):
    """
    core_overlay と defensive_multi を資金比率 alpha_t : (1 - alpha_t) で合成する。
    W_blend(t) = alpha_t * W_core(t) + (1 - alpha_t) * W_def(t)

    mode="fixed"   : alpha_t = core_ratio で一定
    mode="inv_vol" : alpha_t = vol_def / (vol_core + vol_def)（両戦略の直近ボラティリティの逆数で配分）
    """

    def __init__(
        self,
        core_ratio: float = 0.50,
        core_params: CoreOverlayParams | None = None,
        target_vol: float = 0.10,
        bil_cap: float = 0.40,
        mode: str = "fixed",
        vol_window: int = 63,
        rebal_freq: int = 5,
        min_vol: float = 0.05,
    ):
        if mode not in ENSEMBLE_MODES:
            raise ValueError(f"mode は {' / '.join(ENSEMBLE_MODES)} のいずれかで指定してください: {mode!r}")
        if not 0.0 <= core_ratio <= 1.0:
            raise ValueError(f"core_ratio は 0.0〜1.0 で指定してください: {core_ratio}")
        if vol_window < 10:
            raise ValueError(f"vol_window は10以上で指定してください（最小観測数の下限が10日のため）: {vol_window}")
        if rebal_freq < 1:
            raise ValueError(f"rebal_freq は1以上で指定してください: {rebal_freq}")
        if not min_vol > 0.0:
            raise ValueError(f"min_vol は正の値で指定してください: {min_vol}")
        self.mode = mode
        self.core_ratio = core_ratio
        self.vol_window = vol_window
        self.rebal_freq = rebal_freq
        self.min_vol = min_vol
        self.core_alloc = CoreOverlayAllocator(core_params)
        self.def_alloc = DefensiveMultiAllocator(target_vol=target_vol, bil_cap=bil_cap)
        # generate_daily_weights の実行後に日次の配分比率と最新日の値が入る
        self.alpha_series_: pd.Series | None = None
        self.latest_alpha_: float | None = core_ratio if mode == "fixed" else None

    def params_dict(self) -> dict:
        if self.mode == "fixed":
            ratio = {"ensemble_mode": "fixed", "core_ratio": self.core_ratio}
        else:
            ratio = {
                "ensemble_mode": "inv_vol",
                "vol_window": self.vol_window,
                "rebal_freq": self.rebal_freq,
                "min_vol": self.min_vol,
            }
        return {
            **ratio,
            "effective_core_ratio": None if self.latest_alpha_ is None else round(self.latest_alpha_, 4),
            "core_overlay": self.core_alloc.params_dict(),
            "defensive_multi": self.def_alloc.params_dict(),
        }

    def generate_daily_weights(
        self, prices: pd.DataFrame, dollar_vol: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        w_core, crowd_daily = self.core_alloc.generate_daily_weights(prices, dollar_vol)
        w_def, _ = self.def_alloc.generate_daily_weights(prices, dollar_vol)
        w_core = w_core.reindex(columns=DEFAULT_UNIVERSE, fill_value=0.0)
        w_def = w_def.reindex(columns=DEFAULT_UNIVERSE, fill_value=0.0)

        if self.mode == "inv_vol":
            ret = prices.reindex(columns=DEFAULT_UNIVERSE).pct_change().fillna(0.0)
            alpha = self.inverse_vol_alpha(sleeve_returns(w_core, ret), sleeve_returns(w_def, ret))
        else:
            alpha = pd.Series(self.core_ratio, index=w_core.index)

        w_blend = w_core.mul(alpha, axis=0) + w_def.mul(1.0 - alpha, axis=0)
        self.alpha_series_ = alpha.rename("core_ratio")
        self.latest_alpha_ = float(alpha.iloc[-1])
        # レジーム別評価は日次で更新される core_overlay の Crowding スコアで行う（defensive_multi 側は月次の階段状）
        return w_blend, crowd_daily

    def inverse_vol_alpha(self, r_core: pd.Series, r_def: pd.Series) -> pd.Series:
        """両戦略の模擬日次リターンから、core_overlay への配分比率 alpha_t を逆ボラ比率で求める。"""
        min_periods = max(10, self.vol_window // 2)
        # defensive_multi が BIL 100% の間はボラがほぼ0になるため、下限を設けて比率の発散を防ぐ
        vol_core = (r_core.rolling(self.vol_window, min_periods=min_periods).std() * np.sqrt(252)).clip(lower=self.min_vol)
        vol_def = (r_def.rolling(self.vol_window, min_periods=min_periods).std() * np.sqrt(252)).clip(lower=self.min_vol)
        alpha_raw = (vol_def / (vol_core + vol_def)).fillna(0.5)

        # 比率の日々の微小な変動で売買が増えないよう、rebal_freq 営業日ごとにだけ更新する
        keep = np.zeros(len(alpha_raw), dtype=bool)
        keep[:: self.rebal_freq] = True
        return alpha_raw.where(keep).ffill()
