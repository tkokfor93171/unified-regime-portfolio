from abc import ABC, abstractmethod

import pandas as pd


class BaseAllocator(ABC):
    @abstractmethod
    def generate_daily_weights(
        self, prices: pd.DataFrame, dollar_vol: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        日次ターゲットウェイトのDataFrameと、評価用の日次レジーム指標(Crowding等)のSeriesを返す。
        ウェイトの列は DEFAULT_UNIVERSE に揃え、t 日の行は t 日終値までの情報だけで決める。
        """

    def params_dict(self) -> dict:
        """target_weights.json に記録するパラメータ。"""
        return {}
