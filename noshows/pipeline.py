"""Shared feature metadata and preprocessing for AttendIQ."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = (
    "lead_days",
    "age",
    "prior_no_shows",
    "hour",
    "first_visit",
    "distance_km",
)

CATEGORICAL_FEATURES = (
    "weekday",
    "booking",
    "insurance",
    "treatment",
)

TARGET_COLUMN = "no_show"
LEAKAGE_COLUMNS = ("appointment_id", "fee_charged")
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor():
    """Create preprocessing for numerical and categorical features."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(NUMERIC_FEATURES)),
            ("categorical", categorical_pipeline, list(CATEGORICAL_FEATURES)),
        ]
    )

    return preprocessor
