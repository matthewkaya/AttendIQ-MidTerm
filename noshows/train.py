"""Reproducible model training and evaluation for AttendIQ."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from noshows.data import make_appointments
from noshows.pipeline import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    build_preprocessor,
)

RANDOM_SEED = 11
TEST_SIZE = 0.25
N_SPLITS = 5
DEFAULT_THRESHOLD = 0.5

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"


def make_pipeline(classifier):
    """Build a preprocessing and classification pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )


def calculate_metrics(y_true, probabilities, threshold=0.5):
    """Calculate classification metrics for a probability threshold."""
    predictions = (np.asarray(probabilities) >= threshold).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }


def save_json(path, data):
    """Write JSON with readable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, allow_nan=False)


def main():
    """Generate data, compare models, tune, evaluate, and save artifacts."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("AttendIQ - Machine Learning Training")
    print("=" * 50)

    # 1. Generate the professor-provided synthetic data.
    # Do not modify noshows/data.py.
    df = make_appointments(n=4000, seed=RANDOM_SEED)

    # Only use approved features, excluding IDs and post-visit leakage.
    X = df[list(FEATURE_COLUMNS)]
    y = df[TARGET_COLUMN]

    print(f"Total appointments: {len(df)}")
    print(f"No-show rate: {y.mean():.2%}")

    # 2. Reserve the test data before model selection.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    print(f"Training appointments: {len(X_train)}")
    print(f"Test appointments: {len(X_test)}")

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    # 3. Compare the baseline and three candidate models.
    # The balanced Logistic Regression candidate explicitly
    # addresses possible class imbalance.
    candidates = {
        "Dummy Classifier": DummyClassifier(strategy="prior"),
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_SEED,
            class_weight="balanced",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            random_state=RANDOM_SEED,
        ),
    }

    scoring = {
        "roc_auc": "roc_auc",
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }

    cv_rows = []

    print("\n5-Fold Cross-Validation Results")
    print("-" * 50)

    for name, classifier in candidates.items():
        pipeline = make_pipeline(classifier)

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            error_score="raise",
        )

        row = {"model": name}

        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std())

        cv_rows.append(row)

        print(f"{name}: ROC-AUC={row['roc_auc_mean']:.4f}, F1={row['f1_mean']:.4f}")

    cv_results = pd.DataFrame(cv_rows)
    cv_results.to_csv(REPORT_DIR / "cv_results.csv", index=False)

    # 4. Tune Logistic Regression using training data only.
    print("\nHyperparameter Tuning")
    print("-" * 50)

    logistic_pipeline = make_pipeline(
        LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_SEED,
        )
    )

    # Eight parameter combinations: 4 C values x 2 weights.
    parameter_grid = {
        "classifier__C": [0.01, 0.1, 1.0, 10.0],
        "classifier__class_weight": [None, "balanced"],
    }

    search = GridSearchCV(
        estimator=logistic_pipeline,
        param_grid=parameter_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )

    search.fit(X_train, y_train)

    print(f"Best parameters: {search.best_params_}")
    print(f"Best CV ROC-AUC: {search.best_score_:.4f}")

    best_settings = search.best_params_

    # 5. Generate out-of-fold training probabilities.
    # Joshua can use this file for cost-based threshold selection
    # without using the held-out test set.
    tuned_pipeline = make_pipeline(
        LogisticRegression(
            C=best_settings["classifier__C"],
            class_weight=best_settings["classifier__class_weight"],
            max_iter=2000,
            random_state=RANDOM_SEED,
        )
    )

    oof_probabilities = cross_val_predict(
        tuned_pipeline,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    oof_results = pd.DataFrame(
        {
            "y_true": y_train.to_numpy(),
            "probability": oof_probabilities,
        }
    )

    oof_results.to_csv(
        REPORT_DIR / "oof_predictions.csv",
        index=False,
    )

    oof_metrics = calculate_metrics(
        y_train,
        oof_probabilities,
        DEFAULT_THRESHOLD,
    )

    # 6. Evaluate the fitted model on the held-out test set.
    # Keep threshold 0.5 until Joshua provides the
    # training-only optimized threshold.
    best_model = search.best_estimator_
    test_probabilities = best_model.predict_proba(X_test)[:, 1]

    test_metrics = calculate_metrics(
        y_test,
        test_probabilities,
        DEFAULT_THRESHOLD,
    )

    print("\nHeld-Out Test Results")
    print("-" * 50)

    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")

    # 7. Save the complete fitted preprocessing + classifier pipeline.
    model_path = MODEL_DIR / "no_show.joblib"
    joblib.dump(best_model, model_path)

    # 8. Save metadata compatible with NoShowPredictor.
    metadata = {
        "sklearn_version": sklearn.__version__,
        "seed": RANDOM_SEED,
        "threshold": DEFAULT_THRESHOLD,
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "best_settings": best_settings,
        "test_results": test_metrics,
    }

    metadata_path = MODEL_DIR / "model_info.json"
    save_json(metadata_path, metadata)

    # 9. Save the training and evaluation summary.
    report = {
        "seed": RANDOM_SEED,
        "total_samples": int(len(df)),
        "training_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "cv_folds": N_SPLITS,
        "model_selection_metric": "roc_auc",
        "selected_model": "Logistic Regression",
        "best_settings": best_settings,
        "best_cv_roc_auc": float(search.best_score_),
        "oof_results": oof_metrics,
        "test_results": test_metrics,
        "threshold": DEFAULT_THRESHOLD,
        "threshold_status": "Default; awaiting OOF cost optimization",
    }

    save_json(REPORT_DIR / "metrics.json", report)

    print("\nSaved Artifacts")
    print("-" * 50)
    print(f"Model: {model_path}")
    print(f"Metadata: {metadata_path}")
    print(f"CV results: {REPORT_DIR / 'cv_results.csv'}")
    print(f"OOF predictions: {REPORT_DIR / 'oof_predictions.csv'}")
    print(f"Metrics: {REPORT_DIR / 'metrics.json'}")
    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()
