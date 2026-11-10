"""Load the saved model contract and predict no-show risk for appointments."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib

from noshows.pipeline import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES

if TYPE_CHECKING:
    from noshows.appointment import Appointment


class NoShowPredictor:
    """Score validated appointments using a saved pipeline and threshold.

    ``predict`` returns ``{"probability": float, "call_recommended": bool}``.
    A call is recommended when the probability for label 1 is at least the
    threshold stored in the model metadata.
    """

    def __init__(
        self,
        model_path: str | Path = "models/no_show.joblib",
        metadata_path: str | Path = "models/model_info.json",
    ) -> None:
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self._metadata = self._load_metadata()
        self.threshold = float(self._metadata["threshold"])
        self._model = joblib.load(self.model_path)
        if not callable(getattr(self._model, "predict_proba", None)):
            raise ValueError("Saved model must provide predict_proba()")  # noqa: TRY004

    def _load_metadata(self) -> dict[str, Any]:
        try:
            with self.metadata_path.open(encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Could not read model metadata: {self.metadata_path}"
            ) from exc

        if not isinstance(metadata, dict):
            raise ValueError("Model metadata must be a JSON object")  # noqa: TRY004
        required = {
            "sklearn_version",
            "seed",
            "threshold",
            "numeric_features",
            "categorical_features",
            "best_settings",
            "test_results",
        }
        missing = sorted(required - metadata.keys())
        if missing:
            raise ValueError(
                f"Model metadata is missing required keys: {', '.join(missing)}"
            )

        threshold = metadata["threshold"]
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError("Model metadata threshold must be numeric")  # noqa: TRY004
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError(
                "Model metadata threshold must be finite and between 0 and 1"
            )
        if not isinstance(metadata["sklearn_version"], str):
            raise ValueError("Model metadata sklearn_version must be a string")  # noqa: TRY004
        if isinstance(metadata["seed"], bool) or not isinstance(metadata["seed"], int):
            raise ValueError("Model metadata seed must be an integer")  # noqa: TRY004
        if metadata["numeric_features"] != list(NUMERIC_FEATURES):
            raise ValueError(
                "Model metadata numeric_features do not match the pipeline contract"
            )
        if metadata["categorical_features"] != list(CATEGORICAL_FEATURES):
            raise ValueError(
                "Model metadata categorical_features do not match the pipeline contract"
            )
        if not isinstance(metadata["best_settings"], dict):
            raise ValueError("Model metadata best_settings must be an object")  # noqa: TRY004
        if not isinstance(metadata["test_results"], dict):
            raise ValueError("Model metadata test_results must be an object")  # noqa: TRY004
        return metadata

    def predict(self, appointment: Appointment) -> dict[str, float | bool]:
        """Return the positive-label probability and threshold-based action."""
        from noshows.appointment import Appointment as AppointmentType

        if not isinstance(appointment, AppointmentType):
            raise TypeError("appointment must be an Appointment instance")

        frame = appointment.to_frame()
        if len(frame) != 1:
            raise ValueError("Appointment.to_frame() must return exactly one row")
        if list(frame.columns) != list(FEATURE_COLUMNS):
            raise ValueError(
                "Appointment columns do not match the pipeline feature order"
            )

        classes = list(getattr(self._model, "classes_", ()))
        if 1 not in classes:
            raise ValueError("Saved model must expose probability for positive label 1")
        probabilities = self._model.predict_proba(frame)
        try:
            probability = float(probabilities[0][classes.index(1)])
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError(
                "Saved model returned an invalid probability result"
            ) from exc
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Saved model returned a probability outside [0, 1]")
        return {
            "probability": probability,
            "call_recommended": probability >= self.threshold,
        }

    def __repr__(self) -> str:
        return (
            f"NoShowPredictor(model_path={str(self.model_path)!r}, "
            f"metadata_path={str(self.metadata_path)!r}, "
            f"threshold={self.threshold!r})"
        )
