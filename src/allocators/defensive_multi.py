# src/allocators/defensive_multi.py
from __future__ import annotations

import numpy as np
import pandas as pd

from src.allocators.base import BaseAllocator
from src.data.loader import DEFAULT_UNIVERSE
from src.signals.features import compute_monthly_signals


class DefensiveMultiAllocator(BaseAllocator):
    def __init__(self, target_vol: float = 0.10, bil_cap: float = 0.40):
        self.risk_tickers = ["SPY", "QQQ", "TLT", "GLD"]
        self.target_vol = target_vol
        self.bil_cap = bil_cap

    def params_dict(self) -> dict:
        return {"target_vol": self.target_vol, "bil_cap": self.bil_cap}

    def generate_daily_weights(
        self, prices: pd.DataFrame, dollar_vol: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        sigs = compute_monthly_signals(prices, dollar_vol, self.risk_tickers)
        mom_z = sigs["mom_z"]
        r_vol = sigs["realized_vol"]
        crowd_z = sigs["crowd_z"]
        liq_z = sigs["liq_z"]

        # 1. ベースウェイト: モメンタムZ > 0 のみ残し、1/Vol で重み付けして行正規化
        pos_mom = mom_z.clip(lower=0.0)
        inv_vol = (1.0 / r_vol.replace(0, np.nan)).fillna(0.0)
        raw_score = pos_mom * inv_vol
        row_sum = raw_score.sum(axis=1).replace(0, np.nan)
        w_base = raw_score.div(row_sum, axis=0).fillna(0.0)

        # 2. 3つのリスク・マルチプライヤー
        avg_vol = r_vol.mean(axis=1)
        m_vol = (self.target_vol / avg_vol.replace(0, np.nan)).clip(0.0, 1.5).fillna(0.0)
        m_crowd = np.exp(-0.7 * crowd_z.clip(0.0, 3.0).fillna(0.0))
        m_liq = np.exp(-0.6 * liq_z.clip(0.0, 3.0).fillna(0.0))

        w_risk = w_base.mul(m_vol * m_crowd * m_liq, axis=0)

        # 3. グロス100%上限とBIL(現金)配分・40%キャップ処理
        w_risk = w_risk.div(w_risk.sum(axis=1).clip(lower=1.0), axis=0)
        risk_sum = w_risk.sum(axis=1)
        w_bil = (1.0 - risk_sum).clip(lower=0.0)

        # BIL上限40%: 超過分はリスク資産へ比例還流（ただしボラ全欠損やリスク資産ゼロ時はBIL 100%）
        vol_valid = r_vol.notna().any(axis=1)
        cap_mask = (risk_sum > 0) & vol_valid & (w_bil > self.bil_cap)
        w_risk.loc[cap_mask] = w_risk.loc[cap_mask].mul((1.0 - self.bil_cap) / risk_sum[cap_mask], axis=0)
        w_bil = w_bil.mask(cap_mask, self.bil_cap)

        # ボラティリティ全欠損月またはリスク資産ゼロ月は全額BILへフォールバック
        fallback_mask = ~vol_valid | (w_risk.sum(axis=1) <= 0)
        w_risk.loc[fallback_mask] = 0.0
        w_bil = w_bil.mask(fallback_mask, 1.0)

        w_monthly = w_risk.assign(BIL=w_bil).reindex(columns=DEFAULT_UNIVERSE, fill_value=0.0)

        # 4. 日次へ展開。暦月末ラベルは週末や祝日のことがあるので、
        #    ラベル日以降の最初の営業日から適用する（未完了の当月分は使われない）
        w_daily = w_monthly.reindex(prices.index, method="ffill").fillna(0.0)
        daily_crowd = crowd_z.reindex(prices.index, method="ffill").fillna(0.0)
        return w_daily, daily_crowd
