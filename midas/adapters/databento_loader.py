"""Fetch NQ (E-mini NASDAQ-100) intraday futures bars from Databento.

Returns pandas OHLCV plus the matching forward-return and regime panels that
Midas's offline loop expects. Also persists the raw DBN file for downstream
Nautilus consumption.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import databento as db


DEFAULT_DATASET = "GLBX.MDP3"
DEFAULT_SYMBOL = "MNQ.c.0"         # Micro E-mini NASDAQ-100, continuous front-month
DEFAULT_SCHEMA = "ohlcv-1m"
DEFAULT_FWD_HORIZONS_MIN = (5, 15, 30, 60, 240)


def load_env_api_key(env_path: Path | str = ".env") -> Optional[str]:
    """Minimal .env reader — avoids adding python-dotenv as a dependency."""
    p = Path(env_path)
    if not p.exists():
        return os.environ.get("DATABENTO_API_KEY")
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("DATABENTO_API_KEY"):
            _, _, val = line.partition("=")
            return val.strip().strip('"').strip("'")
    return os.environ.get("DATABENTO_API_KEY")


@dataclass
class DatabentoNQLoader:
    api_key: str
    dataset: str = DEFAULT_DATASET
    symbol: str = DEFAULT_SYMBOL
    schema: str = DEFAULT_SCHEMA
    cache_dir: Path = Path("./data/databento")

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = db.Historical(self.api_key)

    def _cache_path(self, start: str, end: str) -> Path:
        tag = f"{self.symbol.replace('.', '_')}_{self.schema}_{start}_{end}.dbn.zst"
        return self.cache_dir / tag

    def estimate_cost(self, start: str, end: str, schema: Optional[str] = None) -> dict:
        """Pre-fetch cost + size estimate via Databento's metadata endpoint.
        Returns {'cost_usd': float, 'size_bytes': int, 'record_count': int}.
        Use to sanity-check a request — especially for MBO/MBP-10 ranges.
        """
        sch = schema or self.schema
        cost = self._client.metadata.get_cost(
            dataset=self.dataset,
            symbols=[self.symbol],
            stype_in="continuous",
            schema=sch,
            start=start,
            end=end,
        )
        size = self._client.metadata.get_billable_size(
            dataset=self.dataset,
            symbols=[self.symbol],
            stype_in="continuous",
            schema=sch,
            start=start,
            end=end,
        )
        record_count = self._client.metadata.get_record_count(
            dataset=self.dataset,
            symbols=[self.symbol],
            stype_in="continuous",
            schema=sch,
            start=start,
            end=end,
        )
        return {
            "cost_usd": float(cost),
            "size_bytes": int(size),
            "record_count": int(record_count),
            "schema": sch,
            "symbol": self.symbol,
            "start": start,
            "end": end,
        }

    def fetch_dbn(self, start: str, end: str) -> Path:
        """Download (or reuse cached) DBN file. Returns the on-disk path."""
        path = self._cache_path(start, end)
        if path.exists():
            return path
        data = self._client.timeseries.get_range(
            dataset=self.dataset,
            symbols=[self.symbol],
            stype_in="continuous",
            schema=self.schema,
            start=start,
            end=end,
        )
        data.to_file(str(path))
        return path

    def fetch_ohlcv(self, start: str, end: str) -> pd.DataFrame:
        """Return tz-aware UTC DataFrame indexed by bar close time with columns:
        open, high, low, close, volume, vwap, trades.
        """
        path = self.fetch_dbn(start, end)
        store = db.DBNStore.from_file(str(path))
        df = store.to_df()

        # Databento ohlcv-1m schema columns: open, high, low, close, volume
        # Index is the bar's open timestamp (UTC).
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        else:
            df.index = df.index.tz_convert("UTC") if df.index.tz else df.index.tz_localize("UTC")

        # Derive a VWAP from typical price (Databento OHLCV schema has no native vwap).
        typical = (df["high"] + df["low"] + df["close"]) / 3.0
        df["vwap"] = (typical * df["volume"]).cumsum() / df["volume"].replace(0, np.nan).cumsum()
        df["vwap"] = df["vwap"].ffill()
        if "trades" not in df.columns:
            df["trades"] = np.nan

        cols = ["open", "high", "low", "close", "volume", "vwap", "trades"]
        return df[cols].sort_index()

    @staticmethod
    def forward_returns(
        ohlcv: pd.DataFrame,
        horizons_min: tuple[int, ...] = DEFAULT_FWD_HORIZONS_MIN,
    ) -> pd.DataFrame:
        """Per-bar forward log-returns at the requested minute horizons.
        Column names follow Midas's convention: ret_<h>h or ret_<m>m.
        """
        close = ohlcv["close"]
        log_close = np.log(close)
        out: dict[str, pd.Series] = {}
        for m in horizons_min:
            shifted = log_close.shift(-m) - log_close
            name = f"ret_{m // 60}h" if m % 60 == 0 and m >= 60 else f"ret_{m}m"
            out[name] = shifted
        return pd.DataFrame(out, index=ohlcv.index)

    @staticmethod
    def vol_regime(
        ohlcv: pd.DataFrame,
        window_min: int = 60,
        low_pct: float = 0.33,
        high_pct: float = 0.67,
    ) -> pd.Series:
        """Volatility-tercile regime label: LOW_VOL / NORMAL / HIGH_VOL,
        based on rolling realised vol of 1-bar log-returns.
        """
        ret = np.log(ohlcv["close"]).diff()
        rvol = ret.rolling(window_min).std()
        lo = rvol.quantile(low_pct)
        hi = rvol.quantile(high_pct)
        out = pd.Series("NORMAL", index=ohlcv.index, name="regime")
        out[rvol <= lo] = "LOW_VOL"
        out[rvol >= hi] = "HIGH_VOL"
        return out
