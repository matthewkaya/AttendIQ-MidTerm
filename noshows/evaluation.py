"""Cost evaluation and training-only threshold selection."""

from __future__ import annotations

import numpy as np

REMINDER_CALL_COST = 8
UNCALLED_NO_SHOW_COST = 35
THRESHOLDS = tuple(float(round(value, 2)) for value in np.arange(0.05, 0.951, 0.01))


def _binary_array(values, name):
    """Return a validated one-dimensional binary array."""
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.isin(array, [0, 1, False, True]).all():
        raise ValueError(f"{name} must contain only binary values 0 and 1")
    return array.astype(int)


def _aligned_binary_inputs(y_true, call_flags):
    labels = _binary_array(y_true, "y_true")
    calls = _binary_array(call_flags, "call_flags")
    if labels.size != calls.size:
        raise ValueError("y_true and call_flags must have the same length")
    return labels, calls


def calculate_total_cost(y_true, call_flags):
    """Return call and uncalled-no-show costs for aligned binary inputs."""
    labels, calls = _aligned_binary_inputs(y_true, call_flags)
    return int(
        REMINDER_CALL_COST * calls.sum()
        + UNCALLED_NO_SHOW_COST * ((labels == 1) & (calls == 0)).sum()
    )


def calculate_call_none_cost(y_true):
    """Return total cost when nobody is called."""
    labels = _binary_array(y_true, "y_true")
    return calculate_total_cost(labels, np.zeros(labels.size, dtype=int))


def calculate_call_all_cost(y_true):
    """Return total cost when everybody is called."""
    labels = _binary_array(y_true, "y_true")
    return calculate_total_cost(labels, np.ones(labels.size, dtype=int))


def calculate_threshold_cost(y_true, probabilities, threshold):
    """Return policy cost for calls at or above ``threshold``."""
    labels = _binary_array(y_true, "y_true")
    scores = _probability_array(probabilities)
    if labels.size != scores.size:
        raise ValueError("y_true and probabilities must have the same length")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("threshold must be numeric and between 0 and 1")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be numeric and between 0 and 1")
    return calculate_total_cost(labels, scores >= threshold)


def _probability_array(probabilities):
    scores = np.asarray(probabilities, dtype=float)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("probabilities must be a non-empty one-dimensional array")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("probabilities must be finite values between 0 and 1")
    return scores


def optimize_threshold_oof(y_true, oof_probabilities):
    """Select a cost-minimizing threshold from training-only OOF predictions."""
    labels = _binary_array(y_true, "y_true")
    scores = _probability_array(oof_probabilities)
    if labels.size != scores.size:
        raise ValueError("y_true and oof_probabilities must have the same length")

    # Thresholds are evaluated in ascending order. Keeping the first minimum
    # makes ties deterministic and favors calling more patients among tied policies.
    costs = [
        calculate_threshold_cost(labels, scores, threshold) for threshold in THRESHOLDS
    ]
    best_index = int(np.argmin(costs))
    return THRESHOLDS[best_index], costs[best_index]
