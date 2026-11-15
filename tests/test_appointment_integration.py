"""Appointment rows scored by the saved pipeline, predictor, and page.

Joshua's cost functions are still unimplemented, so this check stops at the
booking, the saved model, and the page handler.
"""

import math
from pathlib import Path

import pytest

from noshows.appointment import Appointment
from noshows.pipeline import FEATURE_COLUMNS, LEAKAGE_COLUMNS
from noshows.predictor import NoShowPredictor

ROOT = Path(__file__).resolve().parents[1]


def booking(**overrides):
    fields = {
        "lead_days": 7,
        "age": 46,
        "prior_no_shows": 0,
        "hour": 12,
        "first_visit": 0,
        "distance_km": 5.0,
        "weekday": "wed",
        "booking": "front_desk",
        "insurance": "provincial",
        "treatment": "acupuncture",
    }
    fields.update(overrides)
    return Appointment(**fields)


def risky_booking():
    return booking(
        lead_days=60,
        age=20,
        prior_no_shows=6,
        hour=8,
        first_visit=1,
        distance_km=40.0,
        weekday="mon",
        booking="online",
        insurance="self_pay",
        treatment="physio",
    )


@pytest.fixture(scope="module")
def predictor():
    return NoShowPredictor(
        model_path=ROOT / "models" / "no_show.joblib",
        metadata_path=ROOT / "models" / "model_info.json",
    )


def test_frame_matches_saved_feature_order_and_leaves_out_leakage(predictor):
    frame = booking().to_frame()

    assert list(frame.columns) == list(FEATURE_COLUMNS)
    assert list(frame.columns) == [
        *predictor.feature_metadata["numeric_features"],
        *predictor.feature_metadata["categorical_features"],
    ]
    assert set(frame.columns).isdisjoint(LEAKAGE_COLUMNS)
    assert len(frame) == 1


def test_saved_predictor_scores_a_real_appointment(predictor):
    result = predictor.predict(booking())

    assert 0 <= result["probability"] <= 1
    assert result["call_recommended"] is (result["probability"] >= predictor.threshold)


def test_riskier_booking_scores_higher_than_a_safer_one(predictor):
    safe = predictor.predict(booking())
    risky = predictor.predict(risky_booking())

    assert risky["probability"] > safe["probability"]
    assert risky["call_recommended"] is True
    assert safe["call_recommended"] is False


def test_blank_distance_is_accepted_by_the_fitted_pipeline(predictor):
    appointment = booking(distance_km=None)
    frame = appointment.to_frame()
    transformed = predictor._model.named_steps["preprocessor"].transform(frame)
    result = predictor.predict(appointment)

    assert appointment.distance_km is None
    assert math.isnan(frame.loc[0, "distance_km"])
    assert transformed.shape[0] == 1
    assert 0 <= result["probability"] <= 1


def test_saved_model_is_not_loaded_again_for_a_second_booking(predictor, monkeypatch):
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("the saved model was loaded again")

    monkeypatch.setattr("noshows.predictor.joblib.load", fail_if_called)

    predictor.predict(booking())
    predictor.predict(risky_booking())
