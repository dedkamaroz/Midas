"""Evaluate Midas DSL expressions against an OHLCV DataFrame.

Operators match those whitelisted in `midas.proposer._VALID_OPS`. Indicators
(RSI/MACD/BBANDS/ATR/OBV) are delegated to vectorbt; everything else is
vectorised pandas/numpy.

The evaluator parses expressions as a tiny S-expression-style call tree:
    ts_zscore(div(sub(close, vwap), atr(high, low, close, 24)), 48)
"""

from __future__ import annotations

import ast
import operator as _op
from typing import Callable, Optional

import numpy as np
import pandas as pd
import vectorbt as vbt


class DslEvaluator:
    """Evaluate Midas DSL strings against a fixed OHLCV DataFrame."""

    def __init__(self, ohlcv: pd.DataFrame):
        missing = {"open", "high", "low", "close", "volume"} - set(ohlcv.columns)
        if missing:
            raise ValueError(f"OHLCV missing columns: {missing}")
        self.df = ohlcv

    # ---- public ------------------------------------------------------------

    def compute(self, expression: str) -> pd.Series:
        """Parse and evaluate. Returns a pd.Series aligned to self.df.index."""
        tree = ast.parse(expression.strip(), mode="eval").body
        result = self._eval(tree)
        if np.isscalar(result):
            result = pd.Series(result, index=self.df.index)
        return pd.Series(np.asarray(result), index=self.df.index, name=expression[:40])

    # ---- core dispatcher ---------------------------------------------------

    def _eval(self, node):
        if isinstance(node, ast.Call):
            name = node.func.id
            args = [self._eval(a) for a in node.args]
            fn = _OPS.get(name)
            if fn is None:
                raise ValueError(f"Unknown operator: {name}")
            return fn(self, *args)
        if isinstance(node, ast.Name):
            col = node.id
            if col in self.df.columns:
                return self.df[col]
            if col in {"true", "false"}:
                return col == "true"
            raise ValueError(f"Unknown column: {col}")
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -self._eval(node.operand)
        raise ValueError(f"Unsupported AST node: {type(node).__name__}")


# ───────────────────────── operator implementations ──────────────────────────

def _to_series(x, like: pd.Series) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    return pd.Series(x, index=like.index)


def _binop(fn):
    def wrapper(self, a, b):
        if isinstance(a, pd.Series):
            return fn(a, b)
        if isinstance(b, pd.Series):
            return fn(_to_series(a, b), b)
        return fn(a, b)
    return wrapper


_OPS: dict[str, Callable] = {
    # arithmetic
    "add":   _binop(_op.add),
    "sub":   _binop(_op.sub),
    "mul":   _binop(_op.mul),
    "div":   _binop(lambda a, b: a / b.replace(0, np.nan) if isinstance(b, pd.Series) else a / b),
    "power": _binop(_op.pow),
    "log":   lambda s, x: np.log(x.replace(0, np.nan)) if isinstance(x, pd.Series) else np.log(x),
    "abs":   lambda s, x: x.abs() if isinstance(x, pd.Series) else abs(x),
    "sign":  lambda s, x: np.sign(x),

    # time-series
    "delay":     lambda s, x, n: x.shift(int(n)),
    "delta":     lambda s, x, n: x - x.shift(int(n)),
    "returns":   lambda s, x, n: x.pct_change(int(n)),
    "ts_mean":   lambda s, x, n: x.rolling(int(n)).mean(),
    "ts_std":    lambda s, x, n: x.rolling(int(n)).std(),
    "ts_max":    lambda s, x, n: x.rolling(int(n)).max(),
    "ts_min":    lambda s, x, n: x.rolling(int(n)).min(),
    "ts_rank":   lambda s, x, n: x.rolling(int(n)).rank(pct=True),
    "ts_zscore": lambda s, x, n: (x - x.rolling(int(n)).mean()) / x.rolling(int(n)).std(),
    "ema":       lambda s, x, n: x.ewm(span=int(n), adjust=False).mean(),
    "ts_corr":   lambda s, a, b, n: a.rolling(int(n)).corr(b),
    "ts_cov":    lambda s, a, b, n: a.rolling(int(n)).cov(b),

    # cross-sectional — single-asset universe collapses these to z-score-style ops
    "cs_rank":       lambda s, x: x.rank(pct=True),
    "cs_zscore":     lambda s, x: (x - x.mean()) / x.std(),
    "cs_demean":     lambda s, x: x - x.mean(),
    "cs_neutralize": lambda s, x: x - x.mean(),

    # conditional
    "if_else": lambda s, c, a, b: pd.Series(np.where(c, a, b), index=(a if isinstance(a, pd.Series) else b).index),
    "clip":    lambda s, x, lo, hi: x.clip(lo, hi) if isinstance(x, pd.Series) else max(lo, min(hi, x)),
    "max":     _binop(np.maximum),
    "min":     _binop(np.minimum),

    # comparison — return 0/1 series, intended to gate other features via mul()/if_else()
    "gt":  _binop(lambda a, b: (a > b).astype(float)),
    "lt":  _binop(lambda a, b: (a < b).astype(float)),
    "gte": _binop(lambda a, b: (a >= b).astype(float)),
    "lte": _binop(lambda a, b: (a <= b).astype(float)),
    "eq":  _binop(lambda a, b: (a == b).astype(float)),
    "neq": _binop(lambda a, b: (a != b).astype(float)),

    # technical (vectorbt-backed)
    "rsi":        lambda s, close, n: vbt.RSI.run(close, window=int(n)).rsi,
    "atr":        lambda s, hi, lo, cl, n: vbt.ATR.run(hi, lo, cl, window=int(n)).atr,
    "macd":       lambda s, close, fast, slow, sig: vbt.MACD.run(close, fast_window=int(fast), slow_window=int(slow), signal_window=int(sig)).macd,
    "bbands_pct": lambda s, close, n: vbt.BBANDS.run(close, window=int(n)).percent_b,
    "obv":        lambda s, close, vol: (np.sign(close.diff().fillna(0)) * vol).cumsum(),
}


# ───────────────────────── data_fn helper for OfflineLoop ────────────────────

def build_data_fn(
    ohlcv: pd.DataFrame,
    forward_returns: pd.DataFrame,
    regimes: pd.Series,
) -> Callable[[], tuple[Callable[[str], pd.Series], pd.DataFrame, pd.Series]]:
    """Adapter to plug a Databento-fetched panel into `OfflineCompoundLoop.run(data_fn=…)`."""
    evaluator = DslEvaluator(ohlcv)

    def data_fn():
        return evaluator.compute, forward_returns, regimes

    return data_fn
