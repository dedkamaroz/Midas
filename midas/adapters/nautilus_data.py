"""Nautilus Trader data adapter for NQ bars.

Two use-cases:

1. Convert a Databento DBN file to Nautilus `Bar` objects via
   `DatabentoDataLoader.from_dbn_file` — useful when you want to feed a
   `BacktestEngine` with real CME tick/bar data.

2. Replay a pandas OHLCV DataFrame through Midas's `OnlineMonitor` without
   spinning up the full Nautilus engine. The synthetic replay is the
   recommended path for the offline → online handoff while you're still
   prototyping; switch to a full BacktestEngine once the strategy needs
   realistic fill simulation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from nautilus_trader.adapters.databento.loaders import DatabentoDataLoader


def load_dbn_to_nautilus_bars(
    dbn_path: Path | str,
    venue: str = "GLBX",
) -> list:
    """Read a Databento DBN file and return Nautilus `Bar` objects.

    The returned list can be passed to `BacktestEngine.add_data(bars)`. The
    instrument referenced by the bars must be added to the engine first via
    `engine.add_instrument(...)`.
    """
    loader = DatabentoDataLoader()
    bars = loader.from_dbn_file(path=str(dbn_path))
    return list(bars)


def replay_bars_to_monitor(
    ohlcv: pd.DataFrame,
    forward_returns: pd.DataFrame,
    regimes: pd.Series,
    feature_compute,
    feature_names: Iterable[str],
    monitor,
    market_context_fn: Optional[callable] = None,
) -> None:
    """Synchronously walk bars and drive `OnlineMonitor.process_update()`.

    Used for backtest replay when you don't need Nautilus's execution sim.
    `feature_compute` is the `compute(expression) -> pd.Series` callable; for
    each named feature we evaluate it once over the full window, then iterate.
    """
    import asyncio

    feature_series = {name: feature_compute(name) for name in feature_names}
    primary_ret = forward_returns.iloc[:, 0]

    async def _run():
        for ts, row in ohlcv.iterrows():
            values = {n: float(feature_series[n].loc[ts]) for n in feature_series}
            await monitor.process_update(
                timestamp=ts.to_pydatetime(),
                feature_values=values,
                forward_return=float(primary_ret.loc[ts]) if ts in primary_ret.index else 0.0,
                regime=str(regimes.loc[ts]) if ts in regimes.index else "NORMAL",
                market_context=(market_context_fn(ts, row) if market_context_fn else {}),
            )

    asyncio.run(_run())
