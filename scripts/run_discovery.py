"""Run an offline discovery session over the MBO microstructure panel.

Builds a `data_fn` from the 14-column 1s-bar parquet shards under the features
directory, then runs `OfflineCompoundLoop.run` against a rotating set of
research goals. Each goal becomes one LearningDocument in the KB; with
`--iters` refinements per goal, a 30-iteration session = e.g. 6 goals × 5
iters.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from midas.adapters import make_mbo_data_fn, load_env_api_key
from midas.factory import create_midas
from midas.loops import OfflineLoopConfig


def _read_dotenv_var(name: str, env_path: Path | str = ".env") -> str | None:
    """Read a single key=value from .env (no python-dotenv dependency)."""
    p = Path(env_path)
    if not p.exists():
        return None
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        k, _, v = line.partition("=")
        if k.strip() == name:
            return v.strip().strip('"').strip("'")
    return None


DEFAULT_FEATURES_DIR = r"D:\File Transfer\FX Trading\market_data\features\GLBX.MDP3\MNQ.c.0\bar_1s"
DEFAULT_KB           = r"./midas-kb-mnq"

# Each goal exercises a different microstructure angle so the LLM proposer
# explores the feature set rather than re-discovering the same signal six times.
DEFAULT_GOALS = [
    ("short-horizon mean reversion using order-book imbalance",
     ["book_imb_l1", "book_imb_l5", "microprice_dev"]),
    ("order-flow momentum from accumulated OFI",
     ["ofi", "signed_volume"]),
    ("informed-trading detection via VPIN spikes",
     ["vpin_bar", "trade_intensity"]),
    ("queue-dynamics signal from L1 add/cancel asymmetry",
     ["queue_intensity_l1", "bid_size_l1", "ask_size_l1"]),
    ("depth-pressure signal from L5 vs L1 imbalance divergence",
     ["book_imb_l5", "book_imb_l1", "depth_bid_l5", "depth_ask_l5"]),
    ("microprice-deviation reversal at high trade intensity",
     ["microprice_dev", "trade_intensity", "vpin_bar"]),
]

DATA_SCHEMA = (
    "1-second bars (MNQ continuous front-month). "
    "Columns: open, high, low, close (mid_price), volume (trade count); "
    "microstructure — mid_price, microprice, microprice_dev, "
    "book_imb_l1, bid_size_l1, ask_size_l1, "
    "book_imb_l5, depth_bid_l5, depth_ask_l5, "
    "ofi (signed level changes), signed_volume (signed traded size), "
    "trade_intensity (#trades), vpin_bar (|signed|/gross trade vol), "
    "queue_intensity_l1 (net L1 adds-cancels). "
    "IMPORTANT: this is 1-second-bar intraday futures data; the evaluator's "
    "ret_1h/4h/8h/24h/48h columns are relabelled MINUTES (1/4/8/24/48 minutes "
    "forward returns). Lookback windows should be in seconds-to-minutes "
    "(e.g. 30-3600), not days. Expressions should target microstructure "
    "horizons not daily/multi-hour ones."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--features-dir", default=DEFAULT_FEATURES_DIR)
    p.add_argument("--kb",           default=DEFAULT_KB)
    p.add_argument("--provider",     default="anthropic", choices=["anthropic", "openai", "mock"])
    p.add_argument("--model",        default="claude-sonnet-4-6")
    p.add_argument("--iters",        type=int, default=5, help="refinement iterations per goal")
    p.add_argument("--goals",        type=int, default=6, help="number of distinct research goals")
    p.add_argument("--regime",       default="MID_VOL")
    p.add_argument("--sample-frac",  type=float, default=None,
                   help="optional: subsample the panel to this fraction (0-1) for cost control")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ---- 1. Build data_fn from MBO panel
    t0 = time.time()
    print(f"[discovery] loading panel from {args.features_dir}")
    data_fn = make_mbo_data_fn(args.features_dir)
    panel = data_fn.panel
    if args.sample_frac is not None and 0 < args.sample_frac < 1:
        keep = int(len(panel) * args.sample_frac)
        panel = panel.iloc[::max(1, len(panel) // keep)]
        # Rebuild with subsampled panel
        from midas.adapters import DslEvaluator
        from midas.adapters.panel_loader import compute_forward_returns, compute_regimes
        ev = DslEvaluator(panel)
        fwd = compute_forward_returns(panel)
        reg = compute_regimes(panel)
        def data_fn_sub():
            return ev.compute, fwd, reg
        data_fn = data_fn_sub
        data_fn.panel = panel
    print(f"[discovery] panel rows={len(panel):,}  cols={len(panel.columns)}  "
          f"horizon span={panel.index[0]} -> {panel.index[-1]}  "
          f"({time.time()-t0:.1f}s)")

    # ---- 2. Resolve API key
    # Anthropic: prefer the namespaced ANTHROPIC_MIDAS_API_KEY in .env so the
    # key Claude Code itself uses (ANTHROPIC_API_KEY) is kept separate.
    api_key: str | None = None
    if args.provider == "anthropic":
        api_key = _read_dotenv_var("ANTHROPIC_MIDAS_API_KEY") or os.environ.get("ANTHROPIC_MIDAS_API_KEY")
        if not api_key:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        key_source = "ANTHROPIC_MIDAS_API_KEY(.env)" if api_key and _read_dotenv_var("ANTHROPIC_MIDAS_API_KEY") else "ANTHROPIC_API_KEY(env)"
    elif args.provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        key_source = "OPENAI_API_KEY(env)"
    else:
        key_source = "mock"

    if args.provider != "mock" and not api_key:
        sys.exit(f"Missing API key for provider={args.provider}")
    if api_key:
        print(f"[discovery] api key source: {key_source}  len={len(api_key)}  prefix={api_key[:12]}")

    # ---- 3. Bootstrap Midas
    cfg = OfflineLoopConfig(max_iterations=args.iters, model=args.model, verbose=True)
    midas = create_midas(
        kb_path=args.kb,
        api_key=api_key,
        provider=args.provider,
        model=args.model,
        offline_config=cfg,
    )
    print(f"[discovery] Midas wired  kb={args.kb}  model={args.model}")

    # ---- 4. Run discovery across goals
    goals = DEFAULT_GOALS[: args.goals]
    learnings = []
    accepted = []
    t_session = time.time()
    for i, (goal, existing) in enumerate(goals, 1):
        print(f"\n========== Goal {i}/{len(goals)}: {goal} ==========")
        try:
            learning = midas.offline.run(
                research_goal=goal,
                existing_factors=existing,
                data_fn=data_fn,
                regime=args.regime,
                data_schema=DATA_SCHEMA,
            )
            learnings.append(learning)
            if learning.result == "accepted":
                accepted.append(learning)
        except Exception as e:
            print(f"[discovery] goal {i} failed: {type(e).__name__}: {e}")

    # ---- 5. Report
    dt = time.time() - t_session
    print("\n" + "=" * 72)
    print(f"Discovery session complete in {dt/60:.1f} min")
    print(f"  Goals attempted : {len(goals)}")
    print(f"  Learnings saved : {len(learnings)}")
    print(f"  Accepted        : {len(accepted)}")
    for lr in accepted:
        print(f"    [accepted] {lr.expression}  ->  saved as candidate")
    print(f"\nKB: {args.kb}")


if __name__ == "__main__":
    main()
