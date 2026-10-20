"""Download, parse, and clean the UCI Cleveland Heart Disease dataset."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd
import requests

UCI_DATASET_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
DATASET_CITATION = (
    "Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). Heart Disease. "
    "UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X"
)
RAW_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
    "exang", "oldpeak", "slope", "ca", "thal", "num",
]
FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
    "exang", "oldpeak", "slope", "ca", "thal",
]
NUMERIC_COLUMNS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLUMNS = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal",
]

def sha256_file(path: str | Path) -> str:
    """Return a SHA-256 digest for file provenance."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def download_raw_data(
    raw_dir: str | Path = "data/raw", force: bool = False, timeout: int = 60
) -> Path:
    """Download the official UCI archive and extract its processed Cleveland file."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / "processed.cleveland.data"
    if destination.exists() and not force:
        return destination

    response = requests.get(UCI_DATASET_URL, timeout=timeout)
    response.raise_for_status()
    try:
        archive = zipfile.ZipFile(io.BytesIO(response.content))
    except zipfile.BadZipFile as exc:
        raise RuntimeError(
            "The UCI response was not a ZIP archive. Check network access and the dataset URL."
        ) from exc

    names = archive.namelist()
    matches = [name for name in names if name.lower().endswith("processed.cleveland.data")]
    if not matches:
        matches = [name for name in names if name.lower().endswith("cleveland.data")]
    if not matches:
        raise FileNotFoundError(
            f"Could not find a Cleveland data file in UCI archive. Files: {names[:20]}"
        )
    with archive.open(matches[0], "r") as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target)
    return destination

def read_raw_data(path: str | Path) -> pd.DataFrame:
    """Read the no-header UCI text file and normalize '?' to missing values."""
    frame = pd.read_csv(
        path,
        header=None,
        names=RAW_COLUMNS,
        na_values=["?", " ?", "? "],
        skipinitialspace=True,
    )
    return frame

def clean_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize numeric columns and binarize original target values 0..4.

    Target 0 means absence; values 1..4 mean presence. Missing feature values are
    retained as NaN so that imputers are fitted only on each training fold.
    """
    working = frame.copy()
    target_column = "num" if "num" in working.columns else "target"
    required = set(FEATURE_COLUMNS + [target_column])
    missing_columns = sorted(required.difference(working.columns))
    if missing_columns:
        raise ValueError(f"Input is missing required columns: {missing_columns}")

    for column in FEATURE_COLUMNS:
        working[column] = pd.to_numeric(working[column], errors="coerce")

    raw_target = pd.to_numeric(working[target_column], errors="coerce")
    valid_target = raw_target.notna()
    cleaned = working.loc[valid_target, FEATURE_COLUMNS].copy()
    cleaned["target"] = (raw_target.loc[valid_target] > 0).astype("int64")
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    cleaned = cleaned[FEATURE_COLUMNS + ["target"]]
    if cleaned.empty:
        raise ValueError("No rows with a valid target remained after cleaning.")
    if not set(cleaned["target"].unique()).issubset({0, 1}):
        raise ValueError("Target conversion failed; expected labels 0 and 1.")
    return cleaned

def prepare_dataset(
    raw_dir: str | Path = "data/raw",
    output_path: str | Path = "data/processed/heart_disease_clean.csv",
    force_download: bool = False,
) -> tuple[Path, dict]:
    """Download (when needed), clean, save CSV, and write a provenance manifest."""
    raw_path = download_raw_data(raw_dir=raw_dir, force=force_download)
    raw_frame = read_raw_data(raw_path)
    cleaned = clean_dataframe(raw_frame)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)
    manifest = {
        "source_url": UCI_DATASET_URL,
        "citation": DATASET_CITATION,
        "raw_file": str(raw_path),
        "raw_sha256": sha256_file(raw_path),
        "clean_file": str(output_path),
        "rows_after_cleaning": int(len(cleaned)),
        "feature_count": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "target_mapping": {"0": 0, "1_to_4": 1},
        "missing_feature_cells": int(cleaned[FEATURE_COLUMNS].isna().sum().sum()),
        "class_counts": {str(k): int(v) for k, v in cleaned["target"].value_counts().sort_index().items()},
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_path, manifest

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output", default="data/processed/heart_disease_clean.csv")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()
    path, manifest = prepare_dataset(args.raw_dir, args.output, args.force_download)
    print(f"Cleaned dataset saved: {path}")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
