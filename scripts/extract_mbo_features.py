"""Batch MBO -> bar-aligned microstructure features.

Iterates every .dbn.zst under the input MBO directory, runs
`MboFeatureExtractor`, and writes one parquet file per day to the output dir.
Existing outputs are skipped (resumable).

Default paths match the layout created by scripts/pull_week.py.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from midas.adapters import MboFeatureExtractor


DEFAULT_IN  = r"D:\File Transfer\FX Trading\market_data\databento\GLBX.MDP3\MNQ.c.0\mbo"
DEFAULT_OUT = r"D:\File Transfer\FX Trading\market_data\features\GLBX.MDP3\MNQ.c.0\bar_1s"


def human_bytes(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if f < 1024:
            return f"{f:,.2f} {unit}"
        f /= 1024
    return f"{f:,.2f} TB"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in",  dest="in_dir",  default=DEFAULT_IN,  help="MBO DBN directory")
    p.add_argument("--out", dest="out_dir", default=DEFAULT_OUT, help="Parquet output directory")
    p.add_argument("--bar-freq", default="1s", help="pandas freq for bar bucketing (1s, 100ms, ...)")
    p.add_argument("--pattern", default="*.dbn.zst")
    args = p.parse_args()

    in_dir  = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    extractor = MboFeatureExtractor(bar_freq=args.bar_freq)
    files = sorted(in_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files matching {args.pattern} in {in_dir}")

    print(f"Extracting MBO features  in={in_dir}\n                          out={out_dir}  bar={args.bar_freq}")
    total_secs = 0.0
    for path in files:
        target = out_dir / path.name.replace(".dbn.zst", ".parquet")
        if target.exists():
            print(f"  {path.name}: cached -> {target.name}")
            continue
        t0 = time.time()
        df = extractor.extract_from_dbn(path)
        df.to_parquet(target)
        dt = time.time() - t0
        total_secs += dt
        sz = target.stat().st_size
        print(f"  {path.name}: {len(df):,} bars  {human_bytes(sz)}  ({dt:.1f}s)")
    print(f"\nDone. Total extraction time: {total_secs:.1f}s")


if __name__ == "__main__":
    main()
