"""Focused Gradio interface tests with mocked predictor dependencies."""

from unittest.mock import Mock

import gradio as gr
import pytest

import app
from noshows.pipeline import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def _predictor():
    predictor = Mock()
    predictor.feature_metadata = {
        "numeric_features": tuple(NUMERIC_FEATURES),
        "categorical_features": tuple(CATEGORICAL_FEATURES),
    }
    predictor.predict.return_value = {
        "probability": 0.24,
        "call_recommended": True,
    }
    return predictor


class _ValidatingAppointment:
    created = None

    def __init__(self, **fields):
        if fields["age"] > 110:
            raise ValueError("age must be 110 or less")
        type(self).created = fields


def test_builds_inputs_from_predictor_feature_metadata():
    demo = app.build_app(_predictor())
    controls = [
        block
        for block in demo.blocks.values()
        if (getattr(block, "elem_id", None) or "").startswith("appointment-")
    ]

    assert [control.elem_id.removeprefix("appointment-") for control in controls] == [
        *NUMERIC_FEATURES,
        *CATEGORICAL_FEATURES,
    ]
    assert isinstance(controls[0], gr.Number)
    assert isinstance(controls[4], gr.Dropdown)
    assert controls[4].choices == [("0", 0), ("1", 1)]
    assert isinstance(controls[6], gr.Dropdown)


def test_missing_model_artifact_keeps_interface_available(monkeypatch):
    monkeypatch.setattr(
        app,
        "NoShowPredictor",
        Mock(side_effect=FileNotFoundError("models/no_show.joblib")),
    )

    demo = app.build_app()
    messages = [
        block.value for block in demo.blocks.values() if isinstance(block, gr.Markdown)
    ]

    assert any("Model unavailable" in message for message in messages)
    assert (
        len([block for block in demo.blocks.values() if isinstance(block, gr.Number)])
        == 5
    )


def test_handler_passes_validated_appointment_and_shows_probability(monkeypatch):
    predictor = _predictor()
    monkeypatch.setattr(app, "Appointment", _ValidatingAppointment)
    fields = (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES)
    handler = app._prediction_handler(predictor, fields)

    probability, recommendation = handler(
        7, 40, 0, 10, 0, None, "mon", "online", "private", "physio"
    )

    assert probability == "24.0%"
    assert "missing" in recommendation
    assert "Call this patient" in recommendation
    assert _ValidatingAppointment.created["distance_km"] is None
    predictor.predict.assert_called_once()


def test_handler_shows_no_call_when_predictor_does_not_recommend_one(monkeypatch):
    predictor = _predictor()
    predictor.predict.return_value["call_recommended"] = False
    monkeypatch.setattr(app, "Appointment", _ValidatingAppointment)
    fields = (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES)
    handler = app._prediction_handler(predictor, fields)

    _, recommendation = handler(
        7, 40, 0, 10, 0, 5.0, "mon", "online", "private", "physio"
    )

    assert recommendation == "No call recommended"


def test_handler_displays_field_validation_error_without_scoring(monkeypatch):
    predictor = _predictor()
    monkeypatch.setattr(app, "Appointment", _ValidatingAppointment)
    fields = (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES)
    handler = app._prediction_handler(predictor, fields)

    probability, recommendation = handler(
        7, 120, 0, 10, 0, 5.0, "mon", "online", "private", "physio"
    )

    assert probability == ""
    assert "age must be 110 or less" in recommendation
    predictor.predict.assert_not_called()


def test_handler_reports_unavailable_predictor_clearly():
    fields = (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES)
    handler = app._prediction_handler(None, fields, "model artifact is missing")

    probability, recommendation = handler(*([None] * len(fields)))

    assert probability == ""
    assert "Model unavailable" in recommendation
    assert "model artifact is missing" in recommendation


def test_unknown_model_feature_fails_clearly():
    with pytest.raises(ValueError, match="No input definition"):
        app._feature_names(
            {"numeric_features": ("unrecognized",), "categorical_features": ()}
        )
