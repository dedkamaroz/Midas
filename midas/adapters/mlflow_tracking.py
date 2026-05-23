"""Mirror Midas offline learnings into MLflow runs.

Wraps `OfflineCompoundLoop.run()` so each iteration's `LearningDocument` is
logged as an MLflow run with composite scores as metrics, the DSL expression
as a param, and the learning markdown as an artifact.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

import mlflow


class MlflowKbHook:
    """Logs one MLflow run per offline iteration.

    Usage:
        hook = MlflowKbHook(experiment="midas-nq")
        learning = midas.offline.run(..., on_iteration=hook)
    """

    def __init__(
        self,
        experiment: str = "midas",
        tracking_uri: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
    ):
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        self.tags = tags or {}

    def __call__(self, learning: Any) -> None:
        """Hook signature compatible with whatever per-iteration callback the
        offline loop exposes. Pulls fields off the LearningDocument defensively.
        """
        name = getattr(learning, "feature_name", None) or getattr(learning, "name", "candidate")
        with mlflow.start_run(run_name=str(name)):
            for k, v in self.tags.items():
                mlflow.set_tag(k, v)
            for field in ("expression", "regime", "result", "pattern_identified"):
                val = getattr(learning, field, None)
                if val is not None:
                    mlflow.log_param(field, str(val)[:500])

            metrics = getattr(learning, "metrics", None) or {}
            if isinstance(metrics, dict):
                for k, v in metrics.items():
                    try:
                        mlflow.log_metric(k, float(v))
                    except (TypeError, ValueError):
                        continue

            md = getattr(learning, "markdown", None) or getattr(learning, "to_markdown", None)
            if callable(md):
                md = md()
            if isinstance(md, str):
                mlflow.log_text(md, "learning.md")


@contextmanager
def track_offline_run(experiment: str = "midas", run_name: str = "session") -> Iterator[None]:
    """Outer MLflow run wrapping a whole offline session — for top-level params
    and aggregate metrics. Per-iteration runs nest inside via MlflowKbHook.
    """
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name, nested=False):
        yield
