"""One booked appointment, checked at the front desk and then frozen.

Numeric limits and category spellings are the ones the page is allowed to
accept. They are wider than the training rows. ``appointment_id`` only labels
a row, and ``fee_charged`` is known after the visit, so neither can be stored.

Whole-number fields are kept as ``int``. A whole-number float such as ``40.0``
is accepted because the Gradio number box sends floats. ``distance_km`` is a
``float``, or ``None`` when the desk left it blank. ``to_frame`` writes that
blank as NaN and uses ``FEATURE_COLUMNS`` order from the pipeline contract.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from noshows.pipeline import FEATURE_COLUMNS

INTEGER_LIMITS: dict[str, tuple[int, int]] = {
    "lead_days": (0, 120),
    "age": (12, 110),
    "prior_no_shows": (0, 30),
    "hour": (8, 17),
    "first_visit": (0, 1),
}
DISTANCE_LIMITS: tuple[float, float] = (0.1, 150.0)
CATEGORY_CHOICES: dict[str, tuple[str, ...]] = {
    "weekday": ("mon", "tue", "wed", "thu", "fri"),
    "booking": ("online", "phone", "front_desk"),
    "insurance": ("private", "provincial", "self_pay"),
    "treatment": ("physio", "massage", "acupuncture"),
}

_FIELDS = (
    "lead_days",
    "age",
    "prior_no_shows",
    "hour",
    "first_visit",
    "distance_km",
    "weekday",
    "booking",
    "insurance",
    "treatment",
)
_MISSING = object()


def _sealed_message(name: str) -> str:
    return f"{name} cannot be changed after the appointment is created."


def _require_whole_number(field: str, value: Any, low: int, high: int) -> int:
    if (
        value is _MISSING
        or value is None
        or (isinstance(value, str) and not value.strip())
    ):
        raise ValueError(f"{field} is required.")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a whole number.")
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError(f"{field} must be a whole number.")
        value = int(value)
    if not low <= value <= high:
        if field == "first_visit":
            raise ValueError("first_visit must be 0 or 1.")
        raise ValueError(f"{field} must be between {low} and {high}.")
    return int(value)


def _optional_distance(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("distance_km must be a number, or left blank.")
    if not math.isfinite(float(value)):
        raise ValueError("distance_km must be a finite number, or left blank.")
    number = float(value)
    low, high = DISTANCE_LIMITS
    if not low <= number <= high:
        raise ValueError("distance_km must be between 0.1 and 150, or left blank.")
    return number


def _require_category(field: str, value: Any) -> str:
    choices = CATEGORY_CHOICES[field]
    if value is _MISSING or not isinstance(value, str) or value not in choices:
        allowed = ", ".join(choices)
        raise ValueError(f"{field} must be one of: {allowed}.")
    return value


class Appointment:
    """A booking the desk can score, with no fields left open to edit later."""

    __slots__ = (*_FIELDS, "_sealed")

    def __init__(
        self,
        lead_days: Any = _MISSING,
        age: Any = _MISSING,
        prior_no_shows: Any = _MISSING,
        hour: Any = _MISSING,
        first_visit: Any = _MISSING,
        weekday: Any = _MISSING,
        booking: Any = _MISSING,
        insurance: Any = _MISSING,
        treatment: Any = _MISSING,
        distance_km: Any = None,
        **extra: Any,
    ) -> None:
        if extra:
            rejected = ", ".join(sorted(extra))
            raise ValueError(f"{rejected} is not known when the appointment is booked.")
        self.lead_days = _require_whole_number(
            "lead_days", lead_days, *INTEGER_LIMITS["lead_days"]
        )
        self.age = _require_whole_number("age", age, *INTEGER_LIMITS["age"])
        self.prior_no_shows = _require_whole_number(
            "prior_no_shows", prior_no_shows, *INTEGER_LIMITS["prior_no_shows"]
        )
        self.hour = _require_whole_number("hour", hour, *INTEGER_LIMITS["hour"])
        self.first_visit = _require_whole_number(
            "first_visit", first_visit, *INTEGER_LIMITS["first_visit"]
        )
        self.distance_km = _optional_distance(distance_km)
        self.weekday = _require_category("weekday", weekday)
        self.booking = _require_category("booking", booking)
        self.insurance = _require_category("insurance", insurance)
        self.treatment = _require_category("treatment", treatment)
        self._sealed = True

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError(_sealed_message(name))
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        raise AttributeError(_sealed_message(name))

    def _identity(self) -> tuple[Any, ...]:
        return tuple(getattr(self, name) for name in _FIELDS)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Appointment):
            return NotImplemented
        return self._identity() == other._identity()

    def __hash__(self) -> int:
        return hash(self._identity())

    def __repr__(self) -> str:
        parts = ", ".join(f"{name}={getattr(self, name)!r}" for name in _FIELDS)
        return f"Appointment({parts})"

    def to_frame(self) -> pd.DataFrame:
        """Return one row in the pipeline's feature order.

        A missing distance is NaN so the numeric imputer can fill it. The
        frame is a copy: editing it does not change this appointment.
        """
        values: dict[str, Any] = {
            "lead_days": self.lead_days,
            "age": self.age,
            "prior_no_shows": self.prior_no_shows,
            "hour": self.hour,
            "first_visit": self.first_visit,
            "distance_km": (
                float("nan") if self.distance_km is None else self.distance_km
            ),
            "weekday": self.weekday,
            "booking": self.booking,
            "insurance": self.insurance,
            "treatment": self.treatment,
        }
        missing = [column for column in FEATURE_COLUMNS if column not in values]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(
                f"Appointment has no value for pipeline column(s): {joined}."
            )
        return pd.DataFrame([values], columns=list(FEATURE_COLUMNS))
