"""Gradio Blocks interface for Maple Physio appointment risk checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import gradio as gr

from noshows.appointment import Appointment
from noshows.pipeline import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from noshows.predictor import NoShowPredictor

# Assignment validation schema keyed by field. Which fields are rendered, and
# their order, come from the predictor's validated model metadata.
_INPUT_SCHEMA: dict[str, dict[str, Any]] = {
    "lead_days": {"kind": "number", "minimum": 0, "maximum": 120, "step": 1},
    "age": {"kind": "number", "minimum": 12, "maximum": 110, "step": 1},
    "prior_no_shows": {
        "kind": "number",
        "minimum": 0,
        "maximum": 30,
        "step": 1,
    },
    "hour": {"kind": "number", "minimum": 8, "maximum": 17, "step": 1},
    "first_visit": {"kind": "choice", "choices": (0, 1)},
    "distance_km": {
        "kind": "number",
        "minimum": 0.1,
        "maximum": 150,
        "step": 0.1,
        "optional": True,
    },
    "weekday": {"kind": "choice", "choices": ("mon", "tue", "wed", "thu", "fri")},
    "booking": {
        "kind": "choice",
        "choices": ("online", "phone", "front_desk"),
    },
    "insurance": {
        "kind": "choice",
        "choices": ("private", "provincial", "self_pay"),
    },
    "treatment": {
        "kind": "choice",
        "choices": ("physio", "massage", "acupuncture"),
    },
}
_DEFAULT_VALUES = {
    "lead_days": 7,
    "age": 40,
    "prior_no_shows": 0,
    "hour": 10,
    "distance_km": None,
}


def _model_feature_metadata(predictor: NoShowPredictor) -> dict[str, tuple[str, ...]]:
    """Read feature names from the loaded predictor, with scaffold fallback."""
    metadata = getattr(predictor, "feature_metadata", None)
    if metadata is None:
        return {
            "numeric_features": tuple(NUMERIC_FEATURES),
            "categorical_features": tuple(CATEGORICAL_FEATURES),
        }
    try:
        numeric = tuple(metadata["numeric_features"])
        categorical = tuple(metadata["categorical_features"])
    except (KeyError, TypeError) as exc:
        raise ValueError("Model feature metadata is incomplete") from exc
    if numeric != tuple(NUMERIC_FEATURES) or categorical != tuple(CATEGORICAL_FEATURES):
        raise ValueError("Model feature metadata does not match the application schema")
    return {"numeric_features": numeric, "categorical_features": categorical}


def _feature_names(feature_metadata: Mapping[str, Sequence[str]]) -> tuple[str, ...]:
    """Return ordered input fields and reject unsupported schema entries."""
    try:
        fields = tuple(feature_metadata["numeric_features"]) + tuple(
            feature_metadata["categorical_features"]
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("Model feature metadata is incomplete") from exc
    if not fields or len(fields) != len(set(fields)):
        raise ValueError("Model feature metadata must contain unique feature names")
    unsupported = [field for field in fields if field not in _INPUT_SCHEMA]
    if unsupported:
        raise ValueError(
            "No input definition is available for model feature(s): "
            + ", ".join(unsupported)
        )
    return fields


def _create_input(field: str) -> gr.components.Component:
    schema = _INPUT_SCHEMA[field]
    label = field.replace("_", " ").title()
    if schema["kind"] == "choice":
        return gr.Dropdown(
            choices=list(schema["choices"]),
            value=schema["choices"][0],
            label=label,
            elem_id=f"appointment-{field}",
        )

    optional = schema.get("optional", False)
    info = "Optional. Leave blank if the distance is unknown." if optional else None
    return gr.Number(
        label=label,
        value=_DEFAULT_VALUES.get(field),
        minimum=schema["minimum"],
        maximum=schema["maximum"],
        step=schema["step"],
        precision=0 if schema["step"] == 1 else 1,
        info=info,
        elem_id=f"appointment-{field}",
    )


def _prediction_handler(
    predictor: NoShowPredictor | None,
    fields: Sequence[str],
    unavailable_message: str | None = None,
):
    def predict_appointment(*values: object) -> tuple[str, str]:
        if unavailable_message or predictor is None:
            return (
                "",
                f"Model unavailable: {unavailable_message or 'Predictor not loaded.'}",
            )
        if len(values) != len(fields):
            return (
                "",
                "Input error: The submitted fields do not match the model schema.",
            )

        appointment_values = dict(zip(fields, values, strict=True))
        if appointment_values.get("distance_km") == "":
            appointment_values["distance_km"] = None
        missing_distance = appointment_values.get("distance_km") is None
        try:
            appointment = Appointment(**appointment_values)
            result = predictor.predict(appointment)
        except (TypeError, ValueError) as exc:
            return "", f"Input error: {exc}"

        recommendation = (
            "Call this patient" if result["call_recommended"] else "No call recommended"
        )
        if missing_distance:
            recommendation = (
                "Warning: distance is missing and will be imputed. " + recommendation
            )
        return f"{result['probability']:.1%}", recommendation

    return predict_appointment


def build_app(predictor: NoShowPredictor | None = None) -> gr.Blocks:
    """Build the interactive risk checker; missing artifacts remain visible."""
    startup_error = None
    if predictor is None:
        try:
            predictor = NoShowPredictor()
        except (OSError, ValueError, ImportError) as exc:
            startup_error = str(exc)

    try:
        metadata = (
            _model_feature_metadata(predictor)
            if predictor is not None
            else {
                "numeric_features": tuple(NUMERIC_FEATURES),
                "categorical_features": tuple(CATEGORICAL_FEATURES),
            }
        )
        fields = _feature_names(metadata)
    except ValueError as exc:
        startup_error = startup_error or str(exc)
        fields = tuple(NUMERIC_FEATURES) + tuple(CATEGORICAL_FEATURES)

    handler = _prediction_handler(predictor, fields, startup_error)
    with gr.Blocks(title="Maple Physio no-show risk") as demo:
        gr.Markdown(
            "# Maple Physio no-show risk\n"
            "Enter a booked appointment to estimate the chance of a missed visit "
            "and whether a reminder call is recommended."
        )
        if startup_error:
            gr.Markdown(
                f"**Model unavailable:** {startup_error} "
                "Check that the trained model artifacts are installed."
            )
        with gr.Row():
            with gr.Column():
                inputs = [_create_input(field) for field in fields]
                submit = gr.Button("Check appointment", variant="primary")
            with gr.Column():
                probability_output = gr.Textbox(
                    label="Chance of no-show", interactive=False
                )
                recommendation_output = gr.Textbox(
                    label="Recommendation", interactive=False
                )
        submit.click(
            handler,
            inputs=inputs,
            outputs=[probability_output, recommendation_output],
        )
    return demo


try:
    import spaces

    @spaces.GPU
    def _spaces_gpu_placeholder() -> None:
        """Keep the ZeroGPU Space startup check satisfied."""

except ImportError:

    def _spaces_gpu_placeholder() -> None:
        """No-op when running outside Hugging Face Spaces."""


if __name__ == "__main__":
    build_app().launch()
