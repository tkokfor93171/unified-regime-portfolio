# src/data/loader.py
from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import yfinance as yf

DEFAULT_UNIVERSE = ["SPY", "QQQ", "IWM", "TLT", "GLD", "BIL"]


def _field(raw: pd.DataFrame, field: str, tickers: list[str]) -> pd.DataFrame:
    if field not in raw.columns.get_level_values(0):
        raise KeyError(f"yfinance の結果に '{field}' 列がありません: {sorted(set(raw.columns.get_level_values(0)))}")
    frame = raw[field]
    if isinstance(frame, pd.Series):
        frame = frame.to_frame(name=tickers[0])
    return frame.reindex(columns=tickers).astype(float)


def load_market_data(
    tickers: Sequence[str] | None = None,
    start_date: str = "2005-01-01",
    end_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    リターン計算用の調整後終値(Adj Close)と、
    Amihud流動性計算用の未調整ドル出来高(Close * Volume)を取得する。

    Adj Close は分配金・分割の遡及調整で過去の水準が変わるため、出来高との積には使わない。
    """
    tickers = list(tickers or DEFAULT_UNIVERSE)
    raw = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        group_by="column",
        progress=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError("yfinance から価格データを取得できませんでした。ネットワークか銘柄コードを確認してください。")

    raw = raw.sort_index()
    raw.index = pd.to_datetime(raw.index)
    if raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)
    raw.index.name = "Date"

    adj_close = _field(raw, "Adj Close", tickers)
    valid_rows = adj_close.notna().any(axis=1)
    raw, adj_close = raw.loc[valid_rows], adj_close.loc[valid_rows]

    missing = [t for t in tickers if adj_close[t].isna().all()]
    if missing:
        raise RuntimeError(f"価格を取得できなかった銘柄があります: {missing}")

    adj_close = adj_close.ffill()
    raw_close = _field(raw, "Close", tickers).ffill()
    volume = _field(raw, "Volume", tickers).fillna(0.0)

    dollar_volume = raw_close * volume
    return adj_close, dollar_volume
