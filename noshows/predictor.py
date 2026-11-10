"""Serialized-model prediction interface (Muharrem's module)."""


class NoShowPredictor:
    """Load model and metadata once; predict from validated Appointment values."""

    def __init__(self, model_path="models/model.joblib", metadata_path="models/model_info.json"):
        raise NotImplementedError("Assigned to Muharrem")

    def predict(self, appointment):
        """Return no-show probability and the threshold-based call action."""
        raise NotImplementedError("Assigned to Muharrem")

    def __repr__(self):
        return "NoShowPredictor(uninitialized)"
