import numpy as np

from cift.ml.evaluation.splits import EventInterval, PurgedKFold


def test_purged_kfold_purges_overlapping_intervals():
    # 6 samples, each uses [t, t+2]
    events = [EventInterval(i, i + 2) for i in range(6)]
    splitter = PurgedKFold(n_splits=3, embargo=0.0)
    indices = np.arange(6)

    folds = list(splitter.split(indices, events))
    # Fold 1 test indices = [0,1]
    train0, test0 = folds[0]
    assert test0.tolist() == [0, 1]

    # Test interval spans [0,3]; any training event overlapping should be purged.
    # event(2) is [2,4] overlaps; event(3) is [3,5] overlaps boundary; both purged.
    assert 2 not in train0
    assert 3 not in train0


def test_purged_kfold_embargo_removes_near_future_starts():
    events = [EventInterval(i, i + 1) for i in range(8)]
    splitter = PurgedKFold(n_splits=4, embargo=1.0)
    indices = np.arange(8)

    train0, test0 = list(splitter.split(indices, events))[0]
    # Fold 1 test indices = [0,1]
    assert test0.tolist() == [0, 1]

    # test_end is max end over [0,1] = 2
    # embargo_end = 3, so any training sample with start in (2,3] is embargoed -> start=3
    assert 3 not in train0
