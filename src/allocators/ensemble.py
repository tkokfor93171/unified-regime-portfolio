from __future__ import annotations

import pandas as pd

from src.allocators.base import BaseAllocator
from src.allocators.core_overlay import CoreOverlayAllocator, CoreOverlayParams
from src.allocators.defensive_multi import DefensiveMultiAllocator
from src.data.loader import DEFAULT_UNIVERSE


class EnsembleAllocator(BaseAllocator):
    """
    core_overlay と defensive_multi を資金比率 alpha : (1 - alpha) で合成する。
    W_blend(t) = alpha * W_core(t) + (1 - alpha) * W_def(t)
    """

    def __init__(
        self,
        core_ratio: float = 0.50,
        core_params: CoreOverlayParams | None = None,
        target_vol: float = 0.10,
        bil_cap: float = 0.40,
    ):
        if not 0.0 <= core_ratio <= 1.0:
            raise ValueError(f"core_ratio は 0.0〜1.0 で指定してください: {core_ratio}")
        self.alpha = core_ratio
        self.core_alloc = CoreOverlayAllocator(core_params)
        self.def_alloc = DefensiveMultiAllocator(target_vol=target_vol, bil_cap=bil_cap)

    def params_dict(self) -> dict:
        return {
            "core_ratio": self.alpha,
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
        w_blend = self.alpha * w_core + (1.0 - self.alpha) * w_def
        # レジーム別評価は日次で更新される core_overlay の Crowding スコアで行う（defensive_multi 側は月次の階段状）
        return w_blend, crowd_daily
