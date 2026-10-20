import numpy as np
import pandas as pd

from heart_mlops.data import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS
from heart_mlops.modeling import build_pipeline

def test_pipeline_fits_with_missing_features_and_predicts_probabilities():
    rng = np.random.default_rng(7)
    rows = 100
    values = {}
    for column in NUMERIC_COLUMNS:
        values[column] = rng.normal(0, 1, rows)
    categorical_levels = {
        "sex": [0, 1], "cp": [1, 2, 3, 4], "fbs": [0, 1], "restecg": [0, 1, 2],
        "exang": [0, 1], "slope": [1, 2, 3], "ca": [0, 1, 2, 3], "thal": [3, 6, 7],
    }
    for column in CATEGORICAL_COLUMNS:
        values[column] = rng.choice(categorical_levels[column], rows)
    frame = pd.DataFrame(values, columns=FEATURE_COLUMNS)
    frame.loc[0, "chol"] = np.nan
    target = ((frame["age"].fillna(0) + frame["thalach"].fillna(0)) > 0).astype(int)
    model = build_pipeline("logistic_regression")
    model.fit(frame, target)
    probabilities = model.predict_proba(frame.iloc[:5])
    assert probabilities.shape == (5, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert set(model.predict(frame.iloc[:5])).issubset({0, 1})
