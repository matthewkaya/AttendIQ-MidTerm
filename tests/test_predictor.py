"""Focused tests for Muharrem's predictor and artifact metadata contract."""

import json
from unittest.mock import Mock

import pandas as pd
import pytest
import sklearn

from noshows import predictor as predictor_module
from noshows.appointment import Appointment
from noshows.pipeline import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES
from noshows.predictor import NoShowPredictor


def _metadata(**updates):
    metadata = {
        "sklearn_version": sklearn.__version__,
        "seed": 11,
        "threshold": 0.5,
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "best_settings": {"classifier__C": 1.0},
        "test_results": {"roc_auc": 0.8},
    }
    metadata.update(updates)
    return metadata


class _ProbabilityModel:
    def __init__(self, probabilities=(0.4, 0.6)):
        self.classes_ = [0, 1]
        self.probabilities = probabilities
        self.received_frame = None

    def predict_proba(self, frame):
        self.received_frame = frame
        return [self.probabilities]


def _make_predictor(tmp_path, monkeypatch, *, metadata=None, model=None):
    metadata_path = tmp_path / "model_info.json"
    metadata_path.write_text(json.dumps(metadata or _metadata()), encoding="utf-8")
    model_path = tmp_path / "no_show.joblib"
    model_path.touch()
    loaded_model = model or _ProbabilityModel()
    load = Mock(return_value=loaded_model)
    monkeypatch.setattr(predictor_module.joblib, "load", load)
    predictor = NoShowPredictor(model_path=model_path, metadata_path=metadata_path)
    return predictor, loaded_model, load


def _appointment(monkeypatch, frame=None):
    appointment = object.__new__(Appointment)
    if frame is None:
        frame = pd.DataFrame([[0] * len(FEATURE_COLUMNS)], columns=FEATURE_COLUMNS)
    monkeypatch.setattr(Appointment, "to_frame", lambda self: frame, raising=False)
    return appointment


def test_loads_pipeline_once_and_reads_saved_threshold(tmp_path, monkeypatch):
    predictor, _, load = _make_predictor(
        tmp_path, monkeypatch, metadata=_metadata(threshold=0.72)
    )

    assert load.call_count == 1
    assert predictor.threshold == 0.72


def test_metadata_is_not_reopened_during_prediction(tmp_path, monkeypatch):
    predictor, _, load = _make_predictor(tmp_path, monkeypatch)
    predictor.metadata_path.rename(tmp_path / "metadata_moved.json")

    result = predictor.predict(_appointment(monkeypatch))

    assert result["probability"] == 0.6
    load.assert_called_once_with(predictor.model_path)


@pytest.mark.parametrize(
    ("probability", "expected"),
    [(0.49, False), (0.5, True), (0.51, True)],
)
def test_predict_returns_probability_and_threshold_decision(
    tmp_path, monkeypatch, probability, expected
):
    predictor, model, _ = _make_predictor(
        tmp_path,
        monkeypatch,
        model=_ProbabilityModel((1 - probability, probability)),
    )
    appointment = _appointment(monkeypatch)

    result = predictor.predict(appointment)

    assert result == {"probability": probability, "call_recommended": expected}
    assert list(model.received_frame.columns) == list(FEATURE_COLUMNS)


@pytest.mark.parametrize("threshold", [-0.01, 1.01, float("nan"), float("inf"), True])
def test_rejects_invalid_threshold_metadata(tmp_path, monkeypatch, threshold):
    metadata_path = tmp_path / "model_info.json"
    metadata_path.write_text(
        json.dumps(_metadata(threshold=threshold)), encoding="utf-8"
    )
    load = Mock()
    monkeypatch.setattr(predictor_module.joblib, "load", load)

    with pytest.raises(ValueError, match="threshold"):
        NoShowPredictor(tmp_path / "model.joblib", metadata_path)
    load.assert_not_called()


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("numeric_features", ["age"], "numeric_features"),
        ("categorical_features", ["weekday"], "categorical_features"),
        ("seed", "11", "seed"),
        ("best_settings", [], "best_settings"),
        ("test_results", [], "test_results"),
        ("sklearn_version", 1.9, "sklearn_version"),
    ],
)
def test_rejects_metadata_that_conflicts_with_contract(
    tmp_path, monkeypatch, key, value, message
):
    metadata_path = tmp_path / "model_info.json"
    metadata_path.write_text(json.dumps(_metadata(**{key: value})), encoding="utf-8")
    load = Mock()
    monkeypatch.setattr(predictor_module.joblib, "load", load)

    with pytest.raises(ValueError, match=message):
        NoShowPredictor(tmp_path / "model.joblib", metadata_path)
    load.assert_not_called()


@pytest.mark.parametrize("contents", ["{", "[]", "null", "{}"])
def test_rejects_malformed_or_incomplete_metadata(tmp_path, monkeypatch, contents):
    metadata_path = tmp_path / "model_info.json"
    metadata_path.write_text(contents, encoding="utf-8")
    load = Mock()
    monkeypatch.setattr(predictor_module.joblib, "load", load)

    with pytest.raises(ValueError, match="metadata"):
        NoShowPredictor(tmp_path / "model.joblib", metadata_path)
    load.assert_not_called()


def test_rejects_metadata_from_a_different_sklearn_version(tmp_path, monkeypatch):
    metadata_path = tmp_path / "model_info.json"
    metadata_path.write_text(
        json.dumps(_metadata(sklearn_version="0.0.0")), encoding="utf-8"
    )
    load = Mock()
    monkeypatch.setattr(predictor_module.joblib, "load", load)

    with pytest.raises(ValueError, match="scikit-learn version"):
        NoShowPredictor(tmp_path / "model.joblib", metadata_path)
    load.assert_not_called()


def test_wraps_missing_model_artifact_with_clear_error(tmp_path, monkeypatch):
    metadata_path = tmp_path / "model_info.json"
    metadata_path.write_text(json.dumps(_metadata()), encoding="utf-8")
    missing_model_path = tmp_path / "missing.joblib"
    load = Mock(side_effect=FileNotFoundError(missing_model_path))
    monkeypatch.setattr(predictor_module.joblib, "load", load)

    with pytest.raises(ValueError, match="Could not load model artifact"):
        NoShowPredictor(missing_model_path, metadata_path)
    load.assert_called_once_with(missing_model_path)


def test_rejects_missing_metadata_artifact(tmp_path, monkeypatch):
    load = Mock()
    monkeypatch.setattr(predictor_module.joblib, "load", load)

    with pytest.raises(ValueError, match="Could not read model metadata"):
        NoShowPredictor(tmp_path / "no_show.joblib", tmp_path / "missing.json")
    load.assert_not_called()


def test_rejects_loaded_artifact_without_prediction_interface(tmp_path, monkeypatch):
    metadata_path = tmp_path / "model_info.json"
    metadata_path.write_text(json.dumps(_metadata()), encoding="utf-8")
    model_path = tmp_path / "no_show.joblib"
    load = Mock(return_value=object())
    monkeypatch.setattr(predictor_module.joblib, "load", load)

    with pytest.raises(ValueError, match="predict_proba"):
        NoShowPredictor(model_path, metadata_path)
    load.assert_called_once_with(model_path)


def test_rejects_non_appointment_input(tmp_path, monkeypatch):
    predictor, _, _ = _make_predictor(tmp_path, monkeypatch)

    with pytest.raises(TypeError, match="Appointment"):
        predictor.predict(object())


def test_rejects_appointment_without_dataframe_method(tmp_path, monkeypatch):
    predictor, _, _ = _make_predictor(tmp_path, monkeypatch)
    appointment = object.__new__(Appointment)

    with pytest.raises(TypeError, match="to_frame"):
        predictor.predict(appointment)


def test_rejects_frame_with_wrong_column_order(tmp_path, monkeypatch):
    predictor, _, _ = _make_predictor(tmp_path, monkeypatch)
    columns = list(FEATURE_COLUMNS)
    columns[0], columns[1] = columns[1], columns[0]
    appointment = _appointment(
        monkeypatch, pd.DataFrame([[0] * len(columns)], columns=columns)
    )

    with pytest.raises(ValueError, match="feature order"):
        predictor.predict(appointment)


def test_rejects_frame_with_more_than_one_row(tmp_path, monkeypatch):
    predictor, _, _ = _make_predictor(tmp_path, monkeypatch)
    frame = pd.DataFrame([[0] * len(FEATURE_COLUMNS)] * 2, columns=FEATURE_COLUMNS)
    appointment = _appointment(monkeypatch, frame)

    with pytest.raises(ValueError, match="exactly one row"):
        predictor.predict(appointment)


def test_rejects_non_dataframe_from_appointment(tmp_path, monkeypatch):
    predictor, _, _ = _make_predictor(tmp_path, monkeypatch)
    appointment = _appointment(monkeypatch, frame={"values": [0]})

    with pytest.raises(ValueError, match="pandas DataFrame"):
        predictor.predict(appointment)


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan")])
def test_rejects_invalid_model_probability(tmp_path, monkeypatch, probability):
    predictor, _, _ = _make_predictor(
        tmp_path,
        monkeypatch,
        model=_ProbabilityModel((1 - probability, probability)),
    )

    with pytest.raises(ValueError, match="probability"):
        predictor.predict(_appointment(monkeypatch))


def test_wraps_incompatible_model_prediction_failure(tmp_path, monkeypatch):
    model = _ProbabilityModel()
    model.predict_proba = Mock(side_effect=RuntimeError("incompatible input"))
    predictor, _, _ = _make_predictor(tmp_path, monkeypatch, model=model)

    with pytest.raises(ValueError, match="valid probability"):
        predictor.predict(_appointment(monkeypatch))


def test_repr_identifies_artifact_paths_and_threshold(tmp_path, monkeypatch):
    predictor, _, _ = _make_predictor(
        tmp_path, monkeypatch, metadata=_metadata(threshold=0.63)
    )

    representation = repr(predictor)
    assert str(predictor.model_path) in representation
    assert str(predictor.metadata_path) in representation
    assert "threshold=0.63" in representation
