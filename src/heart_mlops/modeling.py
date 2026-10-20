"""Reusable preprocessing and classifier factories."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heart_mlops.data import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS

def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """Create a train-fold-fitted transformer; no preprocessing is fit on all rows."""
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(steps=numeric_steps), NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

def build_pipeline(model_name: str, random_state: int = 42) -> Pipeline:
    """Create one full preprocessing + classifier pipeline."""
    normalized = model_name.strip().lower().replace(" ", "_")
    if normalized in {"logistic_regression", "logisticregression", "logistic", "lr"}:
        classifier = LogisticRegression(max_iter=2000, solver="liblinear", random_state=random_state)
        scale_numeric = True
    elif normalized in {"random_forest", "randomforest", "rf"}:
        classifier = RandomForestClassifier(random_state=random_state, n_jobs=1)
        scale_numeric = False
    else:
        raise ValueError(
            f"Unknown model_name={model_name!r}; use 'logistic_regression' or 'random_forest'."
        )
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=scale_numeric)),
            ("classifier", classifier),
        ]
    )

def parameter_grids() -> dict[str, dict[str, list]]:
    """Small, deterministic grids suited to the 303-row assignment dataset."""
    return {
        "LogisticRegression": {
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__class_weight": [None, "balanced"],
        },
        "RandomForest": {
            "classifier__n_estimators": [100, 250],
            "classifier__max_depth": [None, 6],
            "classifier__min_samples_leaf": [1, 3],
            "classifier__class_weight": [None, "balanced"],
        },
    }
