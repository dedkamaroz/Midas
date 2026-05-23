"""NQ futures adapters: Databento data, vectorbt/pandas DSL evaluator,
Nautilus bar loading, and MLflow learning-document tracking.

Install with: pip install -e .[nq]
"""

from midas.adapters.databento_loader import DatabentoNQLoader, load_env_api_key
from midas.adapters.dsl_eval import DslEvaluator, build_data_fn
from midas.adapters.mbo_features import MBO_FEATURE_COLUMNS, MboFeatureExtractor
from midas.adapters.mlflow_tracking import MlflowKbHook, track_offline_run
from midas.adapters.nautilus_data import load_dbn_to_nautilus_bars
from midas.adapters.panel_loader import (
    compute_forward_returns,
    compute_regimes,
    load_feature_panel,
    make_mbo_data_fn,
    synthesize_ohlcv,
)

__all__ = [
    "DatabentoNQLoader",
    "load_env_api_key",
    "DslEvaluator",
    "build_data_fn",
    "MBO_FEATURE_COLUMNS",
    "MboFeatureExtractor",
    "MlflowKbHook",
    "track_offline_run",
    "load_dbn_to_nautilus_bars",
    "load_feature_panel",
    "synthesize_ohlcv",
    "compute_forward_returns",
    "compute_regimes",
    "make_mbo_data_fn",
]
