"""Validated appointment input contract (Stebin's module)."""


class Appointment:
    """Immutable appointment with field-specific validation.

    Expected implementation: validate against the team guide, accept missing
    distance, and expose one-row DataFrame conversion in shared feature order.
    """

    def __init__(self, **fields):
        raise NotImplementedError("Assigned to Stebin")
