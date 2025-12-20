"""Model evaluation utilities (leakage-safe splitting, walk-forward helpers)."""

from cift.ml.evaluation.splits import EventInterval, PurgedKFold

__all__ = ["EventInterval", "PurgedKFold"]
