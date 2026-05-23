"""Concatenate per-day MBO feature parquets into a single panel and build a
`data_fn` callable for `OfflineCompoundLoop.run(...)`.

The DslEvaluator requires `open/high/low/close/volume` columns; we synthesize
those from MBO outputs (close=mid_price, OHL collapsed to close, volume=
trade_intensity) so the LLM can still call technical operators like `rsi(close)`
alongside microstructure columns like `ofi` and `book_imb_l5`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from midas.adapters.dsl_eval import DslEvaluator
from midas.adapters.mbo_features import MBO_FEATURE_COLUMNS


def load_feature_panel(
    features_dir: Path | str,
    pattern: str = "*.parquet",
    min_rows: int = 1000,
) -> pd.DataFrame:
    """Concatenate every parquet shard under `features_dir` into one panel.

    Days with fewer than `min_rows` bars (e.g. weekend pre-open stubs) are
    dropped — they distort rolling-window statistics. Returns a frame indexed
    by UTC timestamp, sorted, deduplicated.
    """
    features_dir = Path(features_dir)
    shards = sorted(features_dir.glob(pattern))
    if not shards:
        raise FileNotFoundError(f"No parquet files under {features_dir}")

    frames: list[pd.DataFrame] = []
    for p in shards:
        df = pd.read_parquet(p)
        if len(df) < min_rows:
            continue
        frames.append(df)
    if not frames:
        raise ValueError(f"All shards below min_rows={min_rows}")

    panel = pd.concat(frames).sort_index()
    panel = panel[~panel.index.duplicated(keep="last")]
    return panel


def synthesize_ohlcv(panel: pd.DataFrame) -> pd.DataFrame:
    """Add open/high/low/close/volume columns expected by DslEvaluator."""
    out = panel.copy()
    close = out["mid_price"].ffill()
    out["close"] = close
    out["open"] = close.shift(1).fillna(close)
    out["high"] = close
    out["low"] = close
    # trade_intensity = #trades per bar; usable as a volume proxy
    out["volume"] = out["trade_intensity"].fillna(0).astype(float)
    return out


_DEFAULT_HORIZON_MAP = {
    # AlphaEvaluator hard-codes column names ret_{1,4,8,24,48}h. At 1s bars
    # the framework's intended hour horizons would IC to noise — every
    # microstructure signal predicts sub-minute. We relabel minute-scale
    # horizons under the "h" naming so the evaluator runs unchanged while
    # the decay ratio (1 : 4 : 8 : 24 : 48) is preserved at the right scale.
    "ret_1h":  60,     # 1 minute
    "ret_4h":  240,    # 4 minutes
    "ret_8h":  480,    # 8 minutes
    "ret_24h": 1440,   # 24 minutes
    "ret_48h": 2880,   # 48 minutes
}


def compute_forward_returns(
    panel: pd.DataFrame,
    horizon_map: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Future log-returns. Columns named to match AlphaEvaluator expectations."""
    horizon_map = horizon_map or _DEFAULT_HORIZON_MAP
    mid = panel["mid_price"].ffill()
    log_mid = np.log(mid)
    out = pd.DataFrame(index=panel.index)
    for col, bars in horizon_map.items():
        out[col] = log_mid.shift(-bars) - log_mid
    return out


def compute_regimes(
    panel: pd.DataFrame,
    vol_window: int = 600,
    quantiles: tuple[float, float] = (0.33, 0.67),
) -> pd.Series:
    """Tag each bar as LOW_VOL / MID_VOL / HIGH_VOL by rolling realised vol."""
    ret = np.log(panel["mid_price"].ffill()).diff()
    rv = ret.rolling(vol_window).std()
    lo, hi = rv.quantile(quantiles[0]), rv.quantile(quantiles[1])
    regimes = pd.Series("MID_VOL", index=panel.index, dtype="object")
    regimes[rv < lo] = "LOW_VOL"
    regimes[rv > hi] = "HIGH_VOL"
    return regimes


def make_mbo_data_fn(
    features_dir: Path | str,
    horizon_map: dict[str, int] | None = None,
    vol_window: int = 600,
    min_rows: int = 1000,
):
    """Build a `data_fn` suitable for OfflineCompoundLoop.run(data_fn=...).

    Returns a zero-arg callable; calling it returns
        (compute_fn, forward_returns_df, regime_series)
    """
    panel = load_feature_panel(features_dir, min_rows=min_rows)
    panel = synthesize_ohlcv(panel)
    fwd = compute_forward_returns(panel, horizon_map=horizon_map)
    regimes = compute_regimes(panel, vol_window=vol_window)
    evaluator = DslEvaluator(panel)

    def data_fn():
        return evaluator.compute, fwd, regimes

    data_fn.panel = panel
    data_fn.fwd = fwd
    data_fn.regimes = regimes
    return data_fn


__all__ = [
    "load_feature_panel",
    "synthesize_ohlcv",
    "compute_forward_returns",
    "compute_regimes",
    "make_mbo_data_fn",
    "MBO_FEATURE_COLUMNS",
]
