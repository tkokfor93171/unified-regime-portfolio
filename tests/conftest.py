import numpy as np
import pandas as pd
import pytest

from src.data.loader import DEFAULT_UNIVERSE

BETAS = {"SPY": 1.0, "QQQ": 1.2, "IWM": 1.1, "TLT": -0.3, "GLD": 0.1, "BIL": 0.0}
IDIO_VOL = {"SPY": 0.002, "QQQ": 0.005, "IWM": 0.006, "TLT": 0.007, "GLD": 0.009, "BIL": 0.0002}
BASE_DRIFT = {"SPY": 0.0, "QQQ": 0.0001, "IWM": 0.0, "TLT": 0.0002, "GLD": 0.0002, "BIL": 0.00008}


def make_synthetic_market(
    seed: int = 7, start: str = "2005-01-03", end: str = "2016-12-30"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    強気(低ボラ・上昇)と弱気(高ボラ・下落)の局面が交互に続く合成データを作る。
    戻り値は load_market_data と同じ (調整後終値, ドル出来高)。
    """
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(start, end, name="Date")
    n = len(index)

    stress = np.zeros(n, dtype=bool)
    pos, state = 0, False
    while pos < n:
        length = int(rng.integers(60, 250))
        stress[pos : pos + length] = state
        pos, state = pos + length, not state

    market = np.where(stress, -0.0008, 0.0006) + np.where(stress, 0.020, 0.008) * rng.standard_normal(n)
    rets = pd.DataFrame(
        {t: BASE_DRIFT[t] + BETAS[t] * market + IDIO_VOL[t] * rng.standard_normal(n) for t in DEFAULT_UNIVERSE},
        index=index,
    )
    prices = 100.0 * (1.0 + rets).cumprod()
    # 弱気局面では出来高が細り、Amihud 非流動性が上がる
    volume = pd.DataFrame(
        {t: 1e6 * np.exp(0.3 * rng.standard_normal(n)) * np.where(stress, 0.6, 1.0) for t in DEFAULT_UNIVERSE},
        index=index,
    )
    return prices, prices * volume


@pytest.fixture(scope="session")
def market() -> tuple[pd.DataFrame, pd.DataFrame]:
    return make_synthetic_market()
