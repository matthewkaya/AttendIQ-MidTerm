"""Boundary and rejection tests for a single booked appointment."""

import math

import pandas as pd
import pytest

from noshows.appointment import Appointment
from noshows.pipeline import FEATURE_COLUMNS


def valid_fields(**overrides):
    fields = {
        "lead_days": 7,
        "age": 40,
        "prior_no_shows": 0,
        "hour": 10,
        "first_visit": 0,
        "distance_km": 5.0,
        "weekday": "mon",
        "booking": "online",
        "insurance": "private",
        "treatment": "physio",
    }
    fields.update(overrides)
    return fields


def test_accepts_a_complete_booking():
    appointment = Appointment(**valid_fields())

    assert appointment.lead_days == 7
    assert appointment.age == 40
    assert appointment.weekday == "mon"
    assert appointment.treatment == "physio"


def test_stores_a_whole_number_float_as_an_int():
    appointment = Appointment(**valid_fields(age=40.0, hour=10.0))

    assert isinstance(appointment.age, int)
    assert appointment.age == 40
    assert appointment.hour == 10


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lead_days", 0),
        ("lead_days", 120),
        ("age", 12),
        ("age", 110),
        ("prior_no_shows", 0),
        ("prior_no_shows", 30),
        ("hour", 8),
        ("hour", 17),
        ("first_visit", 0),
        ("first_visit", 1),
        ("distance_km", 0.1),
        ("distance_km", 150),
        ("weekday", "fri"),
        ("booking", "front_desk"),
        ("insurance", "self_pay"),
        ("treatment", "acupuncture"),
    ],
)
def test_accepts_each_field_at_its_boundary(field, value):
    appointment = Appointment(**valid_fields(**{field: value}))

    assert getattr(appointment, field) == value


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lead_days", -1),
        ("lead_days", 121),
        ("age", 11),
        ("age", 111),
        ("prior_no_shows", -1),
        ("prior_no_shows", 31),
        ("hour", 7),
        ("hour", 18),
        ("first_visit", -1),
        ("first_visit", 2),
        ("distance_km", 0.09),
        ("distance_km", 150.1),
        ("lead_days", 1.5),
        ("lead_days", True),
        ("lead_days", "7"),
        ("age", None),
        ("hour", float("inf")),
        ("first_visit", 0.5),
        ("distance_km", "5"),
        ("distance_km", True),
        ("weekday", "Mon"),
        ("weekday", "monday"),
        ("weekday", 1),
        ("booking", "walk_in"),
        ("insurance", "OHIP"),
        ("treatment", "chiro"),
    ],
)
def test_rejects_impossible_values_and_names_the_field(field, value):
    with pytest.raises(ValueError, match=field):
        Appointment(**valid_fields(**{field: value}))


@pytest.mark.parametrize(
    "field",
    [
        "lead_days",
        "age",
        "prior_no_shows",
        "hour",
        "first_visit",
        "weekday",
        "booking",
        "insurance",
        "treatment",
    ],
)
def test_rejects_a_missing_required_field(field):
    fields = valid_fields()
    del fields[field]

    with pytest.raises(ValueError, match=field):
        Appointment(**fields)


@pytest.mark.parametrize("field", ["appointment_id", "fee_charged", "no_show"])
def test_rejects_columns_that_are_not_known_at_booking(field):
    with pytest.raises(ValueError, match=field):
        Appointment(**valid_fields(), **{field: 1})


def test_fields_cannot_be_changed_or_removed():
    appointment = Appointment(**valid_fields())

    with pytest.raises(AttributeError, match="age"):
        appointment.age = 41
    with pytest.raises(AttributeError, match="weekday"):
        del appointment.weekday
    assert appointment.age == 40


@pytest.mark.parametrize("missing", [None, "", "   ", float("nan")])
def test_blank_distance_is_kept_and_becomes_missing_in_the_frame(missing):
    appointment = Appointment(**valid_fields(distance_km=missing))
    frame = appointment.to_frame()

    assert appointment.distance_km is None
    assert len(frame) == 1
    assert math.isnan(frame.loc[0, "distance_km"])


def test_to_frame_uses_the_pipeline_column_order():
    appointment = Appointment(**valid_fields())
    frame = appointment.to_frame()

    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == list(FEATURE_COLUMNS)
    assert len(frame) == 1
    assert frame.loc[0, "lead_days"] == 7
    assert frame.loc[0, "booking"] == "online"
    assert frame.loc[0, "distance_km"] == 5.0


def test_editing_the_frame_does_not_change_the_appointment():
    appointment = Appointment(**valid_fields())
    frame = appointment.to_frame()
    frame.loc[0, "age"] = 90

    assert appointment.age == 40


def test_equal_bookings_match_and_can_be_hashed():
    first = Appointment(**valid_fields())
    second = Appointment(**valid_fields())

    assert first == second
    assert len({first, second}) == 1
