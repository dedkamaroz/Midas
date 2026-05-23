"""End-to-end Midas backtest on NQ (E-mini NASDAQ-100) 1-minute bars.

Pipeline:
  1. Pull 1m NQ continuous front-month bars from Databento.
  2. Build forward returns + vol-tercile regime labels.
  3. Run Midas's offline loop with a real DSL evaluator (vectorbt indicators).
  4. Mirror each learning into MLflow.
  5. Promote the top candidate and replay it through OnlineMonitor.

Run:
    pip install -e .[nq]
    python examples/nq_backtest.py --start 2025-01-06 --end 2025-01-10
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from midas import create_midas
from midas.adapters import (
    DatabentoNQLoader,
    MlflowKbHook,
    build_data_fn,
    load_env_api_key,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2025-01-06", help="ISO date, inclusive")
    p.add_argument("--end",   default="2025-01-10", help="ISO date, exclusive")
    p.add_argument("--kb",    default="./midas-kb-nq", help="Knowledge base path")
    p.add_argument("--provider", default="mock", choices=["mock", "openai", "anthropic"])
    p.add_argument("--regime", default="HIGH_VOL")
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--estimate", action="store_true",
                   help="Print Databento cost+size estimate for the window and exit")
    p.add_argument("--schema", default=None,
                   help="Override Databento schema (e.g., mbp-10, mbo) for --estimate")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    api_key = load_env_api_key(".env")
    if not api_key:
        raise SystemExit("DATABENTO_API_KEY not found in .env or environment")

    loader = DatabentoNQLoader(api_key=api_key)

    if args.estimate:
        est = loader.estimate_cost(args.start, args.end, schema=args.schema)
        mb = est["size_bytes"] / 1_000_000
        print(f"Databento estimate  symbol={est['symbol']}  schema={est['schema']}")
        print(f"  window     : {est['start']} -> {est['end']}")
        print(f"  records    : {est['record_count']:,}")
        print(f"  size       : {mb:,.2f} MB")
        print(f"  cost       : ${est['cost_usd']:.4f} USD")
        return

    print(f"[1/5] Fetching MNQ 1m bars {args.start} -> {args.end}")
    ohlcv = loader.fetch_ohlcv(args.start, args.end)
    print(f"      {len(ohlcv):,} bars from {ohlcv.index[0]} to {ohlcv.index[-1]}")

    print("[2/5] Building forward returns + regime labels")
    fwd = loader.forward_returns(ohlcv)
    regimes = loader.vol_regime(ohlcv)

    print(f"[3/5] Bootstrapping Midas at {args.kb}")
    midas = create_midas(kb_path=args.kb, provider=args.provider)

    data_fn = build_data_fn(ohlcv, fwd, regimes)

    hook = None if args.no_mlflow else MlflowKbHook(experiment="midas-nq",
                                                   tags={"symbol": "NQ", "venue": "GLBX"})

    print(f"[4/5] Running offline discovery loop (regime={args.regime})")
    learning = midas.offline.run(
        research_goal="Intraday mean-reversion and momentum signals on NQ 1m bars",
        existing_factors=midas.promoter.list_deployed(),
        data_fn=data_fn,
        regime=args.regime,
    )
    if hook is not None:
        hook(learning)
    print(f"      result={learning.result}  pattern={getattr(learning, 'pattern_identified', '')[:80]}")

    candidates = midas.promoter.list_candidates() if hasattr(midas.promoter, "list_candidates") else []
    if candidates:
        top = candidates[0]
        print(f"[5/5] Promoting top candidate -> {top}")
        midas.promoter.promote(top)
    else:
        print("[5/5] No promotable candidate this run — re-run or relax thresholds")


if __name__ == "__main__":
    main()
