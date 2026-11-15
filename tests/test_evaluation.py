"""Tests for policy cost calculations and OOF threshold selection."""

import numpy as np
import pytest

from noshows.evaluation import (
    THRESHOLDS,
    calculate_call_all_cost,
    calculate_call_none_cost,
    calculate_threshold_cost,
    calculate_total_cost,
    optimize_threshold_oof,
)


def test_cost_matches_hand_calculated_example():
    # Two calls cost $16; one uncalled no-show costs $35: total $51.
    assert calculate_total_cost([1, 0, 1, 0], [1, 1, 0, 0]) == 51


def test_call_none_baseline_costs_every_no_show():
    assert calculate_call_none_cost([0, 1, 1, 0]) == 70


def test_call_all_baseline_costs_every_call():
    assert calculate_call_all_cost([0, 1, 1, 0]) == 32


def test_threshold_policy_calls_at_or_above_threshold():
    assert calculate_threshold_cost([1, 0, 1], [0.8, 0.7, 0.2], 0.7) == 51


@pytest.mark.parametrize(
    ("labels", "calls"),
    [([1, 0], [1]), ([1], [0, 1]), ([], []), ([[1]], [[1]])],
)
def test_cost_rejects_bad_lengths_or_dimensions(labels, calls):
    with pytest.raises(ValueError):
        calculate_total_cost(labels, calls)


@pytest.mark.parametrize("values", [[0, 2], [0, -1], [0, float("nan")]])
def test_cost_rejects_non_binary_values(values):
    with pytest.raises(ValueError, match="binary"):
        calculate_total_cost(values, [0] * len(values))


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), True])
def test_threshold_cost_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="threshold"):
        calculate_threshold_cost([0], [0.5], threshold)


def test_threshold_grid_includes_required_endpoints():
    assert THRESHOLDS[0] == 0.05
    assert THRESHOLDS[-1] == 0.95
    assert len(THRESHOLDS) == 91


def test_oof_threshold_optimizer_uses_cost_minimum_and_returns_cost():
    threshold, cost = optimize_threshold_oof([1, 1, 0, 0], [0.9, 0.8, 0.7, 0.1])

    assert threshold == 0.71
    assert cost == 16


def test_oof_threshold_optimizer_breaks_ties_at_lowest_threshold():
    # No calls at any candidate threshold all have equal cost.
    threshold, cost = optimize_threshold_oof([1], [0.01])

    assert threshold == 0.05
    assert cost == 35


def test_oof_threshold_optimizer_handles_call_everyone_extreme():
    threshold, cost = optimize_threshold_oof([1], [0.99])

    assert threshold == 0.05
    assert cost == 8


def test_oof_threshold_optimizer_rejects_invalid_probabilities():
    with pytest.raises(ValueError, match="probabilities"):
        optimize_threshold_oof([0, 1], [0.2, 1.2])


def test_oof_threshold_optimizer_does_not_mutate_inputs():
    labels = np.array([0, 1])
    probabilities = np.array([0.2, 0.8])
    original_labels = labels.copy()
    original_probabilities = probabilities.copy()

    optimize_threshold_oof(labels, probabilities)

    np.testing.assert_array_equal(labels, original_labels)
    np.testing.assert_array_equal(probabilities, original_probabilities)
