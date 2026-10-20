"""Create reproducible EDA plots and summary statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from heart_mlops.data import FEATURE_COLUMNS, NUMERIC_COLUMNS  # noqa: E402

def run_eda(input_path: str | Path, output_dir: str | Path = "reports/figures") -> dict:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(input_path)
    feature_frame = frame[FEATURE_COLUMNS]

    sns.set_theme(context="notebook", style="whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for axis, column in zip(axes.flat, NUMERIC_COLUMNS):
        sns.histplot(data=frame, x=column, hue="target", multiple="layer", element="step", ax=axis)
        axis.set_title(f"Distribution: {column}")
    for axis in axes.flat[len(NUMERIC_COLUMNS):]:
        axis.axis("off")
    fig.suptitle("UCI Cleveland Heart Disease - Numeric Feature Distributions", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "numeric_histograms.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    counts = frame["target"].value_counts().reindex([0, 1], fill_value=0)
    ax.bar(["Absence (0)", "Presence (1)"], counts.values)
    ax.set_title("Binary Target Balance")
    ax.set_xlabel("Target class")
    ax.set_ylabel("Number of records")
    for index, value in enumerate(counts.values):
        ax.text(index, value, str(int(value)), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_dir / "class_balance.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    numeric_corr = frame[NUMERIC_COLUMNS].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sns.heatmap(numeric_corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    ax.set_title("Correlation Heatmap - Continuous Features Only")
    fig.tight_layout()
    fig.savefig(output_dir / "numeric_correlation_heatmap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    missing = feature_frame.isna().sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    missing.plot(kind="bar", ax=ax)
    ax.set_title("Missing Values by Feature")
    ax.set_ylabel("Missing cells")
    ax.set_xlabel("Feature")
    fig.tight_layout()
    fig.savefig(output_dir / "missing_values.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "row_count": int(len(frame)),
        "feature_count": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "target_counts": {str(k): int(v) for k, v in frame["target"].value_counts().sort_index().items()},
        "target_percentages": {
            str(k): round(float(v / len(frame) * 100), 2)
            for k, v in frame["target"].value_counts().sort_index().items()
        },
        "missing_values_by_feature": {k: int(v) for k, v in missing.items()},
        "total_missing_feature_cells": int(missing.sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "plots": [
            "numeric_histograms.png", "class_balance.png",
            "numeric_correlation_heatmap.png", "missing_values.png",
        ],
    }
    (output_dir.parent / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/processed/heart_disease_clean.csv")
    parser.add_argument("--output", default="reports/figures")
    args = parser.parse_args()
    run_eda(args.input, args.output)

if __name__ == "__main__":
    main()
