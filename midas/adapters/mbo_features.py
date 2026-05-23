"""Reconstruct an aggregate L2 book from Databento MBO and emit bar-aligned
microstructure features that Midas's DSL can use as columns.

The extractor walks MBO events in time order, maintains a per-side price-level
book (with per-order-id tracking so cancels and modifies are exact), and
snapshots the BBO state at each bar boundary. OFI is accumulated event-by-event
using the Cont-Kukanov-Stoikov definition.

Output columns (1-second bars by default):
    mid_price        — (best_bid + best_ask) / 2, end-of-bar
    microprice       — size-weighted bbo price, end-of-bar
    microprice_dev   — microprice − mid_price (short-horizon directional signal)
    book_imb_l1      — (bid_sz − ask_sz) / (bid_sz + ask_sz), end-of-bar
    bid_size_l1      — top-of-book bid size, end-of-bar
    ask_size_l1      — top-of-book ask size, end-of-bar
    ofi              — Cont-Kukanov-Stoikov order-flow imbalance, summed over bar
    signed_volume    — Σ trade_size × sign(aggressor); +ve = net market buy
    trade_intensity  — count of trade events in the bar
    vpin_bar         — |signed_volume| / gross_traded_volume in the bar (NaN if no trades)
    queue_intensity_l1 — (adds − cancels) at L1 across both sides, per bar
    book_imb_l5      — (Σtop5_bid_sz − Σtop5_ask_sz) / total, end-of-bar
    depth_bid_l5     — Σ size across top 5 bid levels, end-of-bar
    depth_ask_l5     — Σ size across top 5 ask levels, end-of-bar
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sortedcontainers import SortedDict

import databento as db


MBO_FEATURE_COLUMNS = (
    "mid_price",
    "microprice",
    "microprice_dev",
    "book_imb_l1",
    "bid_size_l1",
    "ask_size_l1",
    "ofi",
    "signed_volume",
    "trade_intensity",
    "vpin_bar",
    "queue_intensity_l1",
    "book_imb_l5",
    "depth_bid_l5",
    "depth_ask_l5",
)


class _BookSide:
    """Aggregate price-level book with per-order-id tracking.

    Keeping individual order state means Cancel and Modify produce exact level
    decrements rather than approximations.
    """

    __slots__ = ("is_bid", "levels", "orders")

    def __init__(self, is_bid: bool):
        self.is_bid = is_bid
        self.levels: SortedDict = SortedDict()  # price -> aggregate size
        self.orders: dict = {}                  # order_id -> (price, size)

    def add(self, order_id: int, price: float, size: int) -> None:
        prior = self.orders.get(order_id)
        if prior is not None:                   # treat duplicate Add as overwrite
            self._dec(prior[0], prior[1])
        self.orders[order_id] = (price, size)
        self.levels[price] = self.levels.get(price, 0) + size

    def cancel(self, order_id: int) -> None:
        prior = self.orders.pop(order_id, None)
        if prior is None:                       # stale cancel — order never seen
            return
        self._dec(prior[0], prior[1])

    def modify(self, order_id: int, price: float, size: int) -> None:
        prior = self.orders.get(order_id)
        if prior is not None:
            self._dec(prior[0], prior[1])
        self.orders[order_id] = (price, size)
        self.levels[price] = self.levels.get(price, 0) + size

    def fill(self, order_id: int, size: int) -> None:
        """Fill reduces the resting order's remaining size; level reflects it too."""
        prior = self.orders.get(order_id)
        if prior is None:
            return
        p, s = prior
        new_s = s - size
        self._dec(p, size)
        if new_s <= 0:
            self.orders.pop(order_id, None)
        else:
            self.orders[order_id] = (p, new_s)

    def clear(self) -> None:
        self.levels.clear()
        self.orders.clear()

    def _dec(self, price: float, size: int) -> None:
        cur = self.levels.get(price, 0)
        new = cur - size
        if new <= 0:
            self.levels.pop(price, None)
        else:
            self.levels[price] = new

    def best(self) -> tuple[float, int]:
        if not self.levels:
            return (np.nan, 0)
        # SortedDict is ascending; bids want max-price, asks want min-price.
        idx = -1 if self.is_bid else 0
        price, size = self.levels.peekitem(idx)
        return (price, size)

    def topn_size(self, n: int = 5) -> int:
        if not self.levels:
            return 0
        m = len(self.levels)
        k = min(n, m)
        if self.is_bid:
            # top of book = highest price = last index
            return sum(self.levels.peekitem(-1 - i)[1] for i in range(k))
        return sum(self.levels.peekitem(i)[1] for i in range(k))


class MboFeatureExtractor:
    """Stream an MBO DBN file through a reconstruction loop, emit bar-aligned features."""

    def __init__(self, bar_freq: str = "1s"):
        self.bar_freq = bar_freq

    # ---- public ------------------------------------------------------------

    def extract_from_dbn(self, dbn_path: Path | str) -> pd.DataFrame:
        store = db.DBNStore.from_file(str(dbn_path))
        df = store.to_df()
        return self._extract(df)

    def extract_directory(
        self,
        in_dir: Path | str,
        out_dir: Path | str,
        pattern: str = "*.dbn.zst",
    ) -> list[Path]:
        in_dir = Path(in_dir)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for path in sorted(in_dir.glob(pattern)):
            target = out_dir / (path.name.replace(".dbn.zst", ".parquet"))
            if target.exists():
                written.append(target)
                continue
            features = self.extract_from_dbn(path)
            features.to_parquet(target)
            written.append(target)
        return written

    # ---- core loop ---------------------------------------------------------

    def _extract(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=list(MBO_FEATURE_COLUMNS))

        # The DBN store sets ts_recv as the index.
        ts_index = df.index
        bar_keys = ts_index.floor(self.bar_freq).to_numpy()

        actions   = df["action"].to_numpy()
        sides     = df["side"].to_numpy()
        prices    = df["price"].to_numpy()
        sizes     = df["size"].to_numpy().astype(np.int64)
        order_ids = df["order_id"].to_numpy().astype(np.int64)

        bids = _BookSide(is_bid=True)
        asks = _BookSide(is_bid=False)

        cur_bid_px = np.nan; cur_bid_sz = 0
        cur_ask_px = np.nan; cur_ask_sz = 0
        cur_bar = None

        ofi_acc = 0.0
        signed_vol_acc = 0.0
        gross_vol_acc = 0.0
        trade_count_acc = 0
        queue_intensity_acc = 0

        rows: list[tuple] = []

        def snapshot(bar_ts) -> None:
            if np.isnan(cur_bid_px) or np.isnan(cur_ask_px):
                mid = micro = imb = np.nan
            else:
                mid = 0.5 * (cur_bid_px + cur_ask_px)
                denom = cur_bid_sz + cur_ask_sz
                if denom > 0:
                    micro = (cur_bid_px * cur_ask_sz + cur_ask_px * cur_bid_sz) / denom
                    imb = (cur_bid_sz - cur_ask_sz) / denom
                else:
                    micro = mid
                    imb = 0.0
            d_bid5 = bids.topn_size(5)
            d_ask5 = asks.topn_size(5)
            denom5 = d_bid5 + d_ask5
            imb5 = ((d_bid5 - d_ask5) / denom5) if denom5 > 0 else np.nan
            vpin = (abs(signed_vol_acc) / gross_vol_acc) if gross_vol_acc > 0 else np.nan
            rows.append((
                bar_ts,
                mid,
                micro,
                (micro - mid) if not np.isnan(micro) else np.nan,
                imb,
                cur_bid_sz,
                cur_ask_sz,
                ofi_acc,
                signed_vol_acc,
                trade_count_acc,
                vpin,
                queue_intensity_acc,
                imb5,
                d_bid5,
                d_ask5,
            ))

        n = len(df)
        for i in range(n):
            bar = bar_keys[i]
            if cur_bar is None:
                cur_bar = bar
            elif bar != cur_bar:
                snapshot(cur_bar)
                ofi_acc = 0.0
                signed_vol_acc = 0.0
                gross_vol_acc = 0.0
                trade_count_acc = 0
                queue_intensity_acc = 0
                cur_bar = bar

            act  = actions[i]
            side = sides[i]
            px   = prices[i]
            sz   = int(sizes[i])
            oid  = int(order_ids[i])

            # Book mutations
            if act == "A":
                if side == "B":
                    if np.isnan(cur_bid_px) or px >= cur_bid_px:
                        queue_intensity_acc += 1
                    bids.add(oid, px, sz)
                elif side == "A":
                    if np.isnan(cur_ask_px) or px <= cur_ask_px:
                        queue_intensity_acc += 1
                    asks.add(oid, px, sz)
            elif act == "C":
                if side == "B":
                    prior = bids.orders.get(oid)
                    if prior is not None and prior[0] == cur_bid_px:
                        queue_intensity_acc -= 1
                    bids.cancel(oid)
                elif side == "A":
                    prior = asks.orders.get(oid)
                    if prior is not None and prior[0] == cur_ask_px:
                        queue_intensity_acc -= 1
                    asks.cancel(oid)
            elif act == "M":
                if side == "B":   bids.modify(oid, px, sz)
                elif side == "A": asks.modify(oid, px, sz)
            elif act == "F":
                # A passive order was filled; reduce remaining size on the book.
                if side == "B":   bids.fill(oid, sz)
                elif side == "A": asks.fill(oid, sz)
            elif act == "R":
                bids.clear()
                asks.clear()
            elif act == "T":
                # Trade: side denotes the aggressor.
                if side == "B":   signed_vol_acc += sz
                elif side == "A": signed_vol_acc -= sz
                gross_vol_acc += sz
                trade_count_acc += 1
                continue                       # trades don't move the book

            # Re-snapshot BBO and accumulate OFI (Cont-Kukanov-Stoikov)
            new_bid_px, new_bid_sz = bids.best()
            new_ask_px, new_ask_sz = asks.best()

            if not np.isnan(new_bid_px) and not np.isnan(cur_bid_px):
                if   new_bid_px > cur_bid_px: ofi_acc += new_bid_sz
                elif new_bid_px == cur_bid_px: ofi_acc += (new_bid_sz - cur_bid_sz)
                else:                          ofi_acc -= cur_bid_sz

            if not np.isnan(new_ask_px) and not np.isnan(cur_ask_px):
                if   new_ask_px < cur_ask_px: ofi_acc -= new_ask_sz
                elif new_ask_px == cur_ask_px: ofi_acc -= (new_ask_sz - cur_ask_sz)
                else:                          ofi_acc += cur_ask_sz

            cur_bid_px, cur_bid_sz = new_bid_px, new_bid_sz
            cur_ask_px, cur_ask_sz = new_ask_px, new_ask_sz

        if cur_bar is not None:
            snapshot(cur_bar)

        cols = ("ts",) + MBO_FEATURE_COLUMNS
        out = pd.DataFrame(rows, columns=cols).set_index("ts")
        return out
