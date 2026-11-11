"""Boundary and rejection tests for a single booked appointment."""

import pytest

from noshows.appointment import Appointment


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
