"""Shared feature metadata and preprocessing contracts.

Implementation is assigned to the machine learning owner. Keep this module
free of model-specific behavior until the assignment PDF and data source have
been verified.
"""

# Initial feature contract from the team plan. Confirm against the assignment
# and instructor-provided data.py before implementing the pipeline.
NUMERIC_FEATURES = (
    "lead_days", "age", "prior_no_shows", "hour", "first_visit", "distance_km",
)
CATEGORICAL_FEATURES = ("weekday", "booking", "insurance", "treatment")
TARGET_COLUMN = "no_show"
LEAKAGE_COLUMNS = ("appointment_id", "fee_charged")
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor():
    """Build the shared numeric/categorical preprocessing transformer."""
    raise NotImplementedError("Assigned to Ibrahim; confirm feature contract against the PDF and data.py")
