"""Tests for reproducible source data and the shared split contract."""

import pandas as pd

from noshows.data import make_appointments
from noshows.pipeline import FEATURE_COLUMNS, LEAKAGE_COLUMNS, TARGET_COLUMN
from noshows.train import RANDOM_SEED, TEST_SIZE, split_data


def test_data_is_reproducible_and_nonfeatures_are_excluded():
    first = make_appointments(n=400, seed=RANDOM_SEED)
    second = make_appointments(n=400, seed=RANDOM_SEED)

    pd.testing.assert_frame_equal(first, second)
    assert set(FEATURE_COLUMNS).isdisjoint((*LEAKAGE_COLUMNS, TARGET_COLUMN))


def test_split_has_expected_sizes_and_preserves_class_balance():
    data = make_appointments(n=4000, seed=RANDOM_SEED)
    X = data[list(FEATURE_COLUMNS)]
    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = split_data(X, y)

    assert len(X_train) == 3000
    assert len(X_test) == 1000
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert abs(y_train.mean() - y.mean()) < 0.01
    assert abs(y_test.mean() - y.mean()) < 0.01
    assert TEST_SIZE == 0.25
