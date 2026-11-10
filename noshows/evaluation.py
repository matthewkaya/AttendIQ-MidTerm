"""Cost evaluation and threshold selection contracts (Joshua's module)."""

REMINDER_CALL_COST = 8
UNCALLED_NO_SHOW_COST = 35


def calculate_total_cost(y_true, call_flags):
    """Return call and uncalled-no-show costs for aligned binary inputs."""
    raise NotImplementedError("Assigned to Joshua")


def optimize_threshold_oof(y_true, oof_probabilities):
    """Select a cost-minimizing threshold from training-only OOF predictions."""
    raise NotImplementedError("Assigned to Joshua")
