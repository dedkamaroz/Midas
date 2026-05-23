"""Pull one calendar week of MNQ futures data from Databento.

Layout written to:
    <root>/databento/GLBX.MDP3/MNQ.c.0/
        ohlcv-1m/<start>_<end>.dbn.zst
        mbo/<YYYY-MM-DD>.dbn.zst         (one file per UTC calendar day)

MBO is sharded per day so an interrupted run resumes cleanly.
"""

from __future__ import annotations

import argparse
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import databento as db

from midas.adapters import load_env_api_key


DATASET = "GLBX.MDP3"
SYMBOL  = "MNQ.c.0"
STYPE   = "continuous"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True, help="ISO date inclusive (e.g. 2026-05-11)")
    p.add_argument("--end",   required=True, help="ISO date exclusive (e.g. 2026-05-18)")
    p.add_argument("--root",  default=r"D:\File Transfer\FX Trading\market_data",
                   help="Top-level market_data directory")
    p.add_argument("--schemas", nargs="+", default=["ohlcv-1m", "mbo"],
                   help="Which schemas to pull")
    return p.parse_args()


def daterange(start: str, end: str):
    d0 = date.fromisoformat(start)
    d1 = date.fromisoformat(end)
    cur = d0
    while cur < d1:
        yield cur
        cur += timedelta(days=1)


def out_dir(root: Path, schema: str) -> Path:
    p = root / "databento" / DATASET / SYMBOL / schema
    p.mkdir(parents=True, exist_ok=True)
    return p


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:,.2f} {unit}"
        n /= 1024
    return f"{n:,.2f} TB"


def fetch(client: db.Historical, schema: str, start: str, end: str, out_path: Path) -> dict:
    if out_path.exists() and out_path.stat().st_size > 0:
        return {"status": "cached", "size": out_path.stat().st_size}
    t0 = time.time()
    data = client.timeseries.get_range(
        dataset=DATASET,
        symbols=[SYMBOL],
        stype_in=STYPE,
        schema=schema,
        start=start,
        end=end,
    )
    data.to_file(str(out_path))
    return {"status": "downloaded", "size": out_path.stat().st_size, "secs": time.time() - t0}


def main() -> None:
    args = parse_args()
    root = Path(args.root)

    api_key = load_env_api_key(Path(__file__).resolve().parent.parent / ".env")
    if not api_key:
        raise SystemExit("DATABENTO_API_KEY not found")
    client = db.Historical(api_key)

    total_bytes = 0
    print(f"Pulling MNQ {args.start} -> {args.end}  root={root}")

    # Bar-aggregate schemas (one file per requested window)
    for schema in (s for s in args.schemas if s.startswith("ohlcv-")):
        target = out_dir(root, schema) / f"{args.start}_{args.end}.dbn.zst"
        print(f"  {schema} -> {target.name}")
        r = fetch(client, schema, args.start, args.end, target)
        total_bytes += r["size"]
        print(f"    {r['status']}  {human_bytes(r['size'])}"
              + (f"  ({r.get('secs', 0):.1f}s)" if r["status"] == "downloaded" else ""))

    # Tick-level schemas (sharded per UTC calendar day for resumability)
    for schema in (s for s in args.schemas if s in ("mbo", "mbp-1", "mbp-10", "trades", "tbbo")):
        sch_dir = out_dir(root, schema)
        for d in daterange(args.start, args.end):
            d_str  = d.isoformat()
            d_next = (d + timedelta(days=1)).isoformat()
            target = sch_dir / f"{d_str}.dbn.zst"
            print(f"  {schema} {d_str} -> {target.name}")
            r = fetch(client, schema, d_str, d_next, target)
            # Drop tiny placeholder files left by no-data windows (weekends/holidays)
            if r["status"] == "downloaded" and r["size"] < 1024 and target.exists():
                target.unlink()
                print(f"    skipped (no-data window, {r['size']} B placeholder removed)")
                continue
            total_bytes += r["size"]
            print(f"    {r['status']}  {human_bytes(r['size'])}"
                  + (f"  ({r.get('secs', 0):.1f}s)" if r["status"] == "downloaded" else ""))

    print(f"\nDone. Total on-disk: {human_bytes(total_bytes)}")


if __name__ == "__main__":
    main()
