from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class EventInterval:
    """A labeled sample's information interval.

    A sample at index i is considered to use information from `start` through `end`
    (inclusive-ish; we treat them as continuous floats).
    """

    start: float
    end: float

    def overlaps(self, other: "EventInterval") -> bool:
        return not (self.end < other.start or self.start > other.end)


class PurgedKFold:
    """Purged + embargoed K-Fold splitter for time series.

    This is a practical implementation for time-indexed samples where labels
    overlap future intervals.

    Parameters
    ----------
    n_splits:
        Number of folds.
    embargo:
        Embargo window (same units as event start/end), applied after each
        test fold end boundary.
    """

    def __init__(self, n_splits: int = 5, embargo: float = 0.0):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = int(n_splits)
        self.embargo = float(embargo)

    def split(
        self,
        indices: Sequence[int] | np.ndarray,
        events: Sequence[EventInterval],
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        idx = np.asarray(indices, dtype=np.int64)
        n = idx.size
        if n != len(events):
            raise ValueError("indices and events must have same length")
        if n < self.n_splits:
            raise ValueError("not enough samples for n_splits")

        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=np.int64)
        fold_sizes[: n % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            start = current
            stop = current + int(fold_size)
            test_mask = np.zeros(n, dtype=bool)
            test_mask[start:stop] = True
            test_idx = idx[test_mask]

            # Define test interval bounds in time.
            test_events = [events[i] for i in range(start, stop)]
            test_start = min(e.start for e in test_events)
            test_end = max(e.end for e in test_events)
            embargo_end = test_end + self.embargo

            train_mask = ~test_mask

            # Purge: remove training samples whose event overlaps test interval.
            for i in range(n):
                if not train_mask[i]:
                    continue
                e = events[i]
                if not (e.end < test_start or e.start > test_end):
                    train_mask[i] = False

            # Embargo: remove training samples whose start is within (test_end, embargo_end].
            if self.embargo > 0:
                for i in range(n):
                    if not train_mask[i]:
                        continue
                    e = events[i]
                    if test_end < e.start <= embargo_end:
                        train_mask[i] = False

            train_idx = idx[train_mask]
            yield train_idx, test_idx

            current = stop


def build_forward_return_events(
    timestamps: Iterable[float] | np.ndarray,
    *,
    horizon: float,
) -> List[EventInterval]:
    """Helper for common label: forward return over fixed horizon.

    timestamps should be monotonic and in the same time units as horizon.
    """
    ts = np.asarray(list(timestamps) if not isinstance(timestamps, np.ndarray) else timestamps, dtype=np.float64)
    if ts.ndim != 1:
        raise ValueError("timestamps must be 1D")
    if ts.size == 0:
        return []
    return [EventInterval(start=float(t), end=float(t + horizon)) for t in ts]
