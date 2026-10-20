import pandas as pd

from heart_mlops.data import FEATURE_COLUMNS, clean_dataframe

def _row(target):
    values = {
        "age": 57, "sex": 1, "cp": 2, "trestbps": 130, "chol": 236,
        "fbs": 0, "restecg": 0, "thalach": 174, "exang": 0,
        "oldpeak": 0.0, "slope": 2, "ca": 0, "thal": 3, "num": target,
    }
    return values

def test_cleaning_binarizes_original_target_and_keeps_missing_features():
    first = _row(0)
    second = _row(2)
    second["ca"] = "?"
    cleaned = clean_dataframe(pd.DataFrame([first, second]))
    assert list(cleaned.columns) == FEATURE_COLUMNS + ["target"]
    assert cleaned["target"].tolist() == [0, 1]
    assert pd.isna(cleaned.loc[1, "ca"])

def test_cleaning_drops_rows_with_missing_target():
    valid = _row(0)
    invalid = _row("?")
    cleaned = clean_dataframe(pd.DataFrame([valid, invalid]))
    assert len(cleaned) == 1
    assert cleaned["target"].tolist() == [0]
