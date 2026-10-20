"""Run cross-validated model tuning, holdout evaluation, and MLflow tracking."""

from __future__ import annotations

import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split  # noqa: E402

from heart_mlops.data import FEATURE_COLUMNS  # noqa: E402
from heart_mlops.modeling import build_pipeline, parameter_grids  # noqa: E402

try:
    import mlflow
    import mlflow.sklearn
except ImportError:  # useful for unit tests in minimal/offline environments
    mlflow = None

RANDOM_STATE = 42

def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)

def evaluate_predictions(y_true, y_pred, y_probability) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_probability)),
    }

def _save_evaluation_plots(name, y_true, y_pred, y_probability, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = name.lower().replace(" ", "_")
    paths = []

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Predicted 0", "Predicted 1"])
    ax.set_yticks([0, 1], labels=["Actual 0", "Actual 1"])
    ax.set_title(f"{name} - Confusion Matrix")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(int(matrix[row, col])), ha="center", va="center")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    conf_path = output_dir / f"{safe_name}_confusion_matrix.png"
    fig.savefig(conf_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths.append(conf_path)

    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_probability)
    auc_value = roc_auc_score(y_true, y_probability)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(false_positive_rate, true_positive_rate, label=f"ROC-AUC = {auc_value:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Chance baseline")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"{name} - ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    roc_path = output_dir / f"{safe_name}_roc_curve.png"
    fig.savefig(roc_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths.append(roc_path)
    return paths

def _save_feature_importance(model, name, output_dir: Path) -> Path | None:
    classifier = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessor"]
    if hasattr(classifier, "feature_importances_"):
        values = np.asarray(classifier.feature_importances_)
    elif hasattr(classifier, "coef_"):
        values = np.abs(np.asarray(classifier.coef_[0]))
    else:
        return None
    names = preprocessor.get_feature_names_out()
    ranked = pd.DataFrame({"feature": names, "importance": values}).nlargest(15, "importance")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ranked.sort_values("importance").plot.barh(x="feature", y="importance", legend=False, ax=ax)
    ax.set_title(f"{name} - Top Transformed Features")
    ax.set_xlabel("Importance (absolute coefficient or tree importance)")
    ax.set_ylabel("Transformed feature")
    fig.tight_layout()
    path = output_dir / f"{name.lower().replace(' ', '_')}_feature_importance.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path

def _setup_mlflow(mlflow_dir: Path) -> None:
    if mlflow is None:
        print("WARNING: MLflow is not installed; local metrics/models will still be written.")
        return
    override = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(override or mlflow_dir.resolve().as_uri())
    mlflow.set_experiment("heart-disease-classification")

def run_training(
    input_path: str | Path = "data/processed/heart_disease_clean.csv",
    artifacts_dir: str | Path = "artifacts",
    reports_dir: str | Path = "reports",
    mlflow_dir: str | Path = "mlruns",
    model_path: str | Path = "models/heart_disease_model.joblib",
) -> dict:
    input_path = Path(input_path)
    artifacts_dir = Path(artifacts_dir)
    reports_dir = Path(reports_dir)
    figures_dir = reports_dir / "figures"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(input_path)
    missing = set(FEATURE_COLUMNS + ["target"]).difference(frame.columns)
    if missing:
        raise ValueError(f"Cleaned dataset is missing columns: {sorted(missing)}")
    X = frame[FEATURE_COLUMNS]
    y = frame["target"].astype(int)
    if y.nunique() != 2:
        raise ValueError("Training requires both target classes 0 and 1.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"roc_auc": "roc_auc", "accuracy": "accuracy", "precision": "precision", "recall": "recall"}
    grids = parameter_grids()
    all_results = {}
    selection_rows = []
    _setup_mlflow(Path(mlflow_dir))

    for model_name, grid in grids.items():
        run_context = (
            mlflow.start_run(run_name=model_name)
            if mlflow is not None
            else nullcontext()
        )
        with run_context:
            search = GridSearchCV(
                estimator=build_pipeline(model_name),
                param_grid=grid,
                scoring=scoring,
                refit="roc_auc",
                cv=cv,
                n_jobs=-1,
                return_train_score=False,
                error_score="raise",
            )
            search.fit(X_train, y_train)
            if mlflow is not None:
                # Every candidate is a nested run with its own metrics, JSON artifact and CV plot.
                cv_results = search.cv_results_
                for candidate_index, candidate_params in enumerate(cv_results["params"]):
                    with mlflow.start_run(
                        run_name=f"{model_name}_candidate_{candidate_index + 1:02d}", nested=True
                    ):
                        mlflow.set_tag("search_candidate", "true")
                        mlflow.log_params(
                            {key: str(value) for key, value in candidate_params.items()}
                        )
                        candidate_metrics = {
                            f"mean_cv_{metric}": float(
                                cv_results[f"mean_test_{metric}"][candidate_index]
                            )
                            for metric in scoring
                        }
                        candidate_metrics.update({
                            f"std_cv_{metric}": float(
                                cv_results[f"std_test_{metric}"][candidate_index]
                            )
                            for metric in scoring
                        })
                        fold_auc = [
                            float(cv_results[f"split{fold}_test_roc_auc"][candidate_index])
                            for fold in range(cv.get_n_splits())
                        ]
                        candidate_plot_dir = artifacts_dir / "candidate_plots"
                        candidate_plot_dir.mkdir(parents=True, exist_ok=True)
                        candidate_plot = candidate_plot_dir / (
                            f"{model_name.lower()}_candidate_{candidate_index + 1:02d}_cv_auc.png"
                        )
                        fig, ax = plt.subplots(figsize=(4.5, 3.2))
                        ax.bar([f"Fold {fold + 1}" for fold in range(len(fold_auc))], fold_auc)
                        ax.set_ylim(0, 1)
                        ax.set_ylabel("ROC-AUC")
                        ax.set_title(f"{model_name} candidate {candidate_index + 1}: CV folds")
                        fig.tight_layout()
                        fig.savefig(candidate_plot, dpi=120, bbox_inches="tight")
                        plt.close(fig)
                        mlflow.log_metrics(candidate_metrics)
                        mlflow.log_dict(
                            {
                                "candidate_index": candidate_index + 1,
                                "params": candidate_params,
                                "fold_roc_auc": fold_auc,
                                "metrics": candidate_metrics,
                            },
                            "candidate_summary.json",
                        )
                        mlflow.log_artifact(str(candidate_plot), artifact_path="plots")
            fitted_model = search.best_estimator_
            y_pred = fitted_model.predict(X_test)
            y_probability = fitted_model.predict_proba(X_test)[:, 1]
            holdout_metrics = evaluate_predictions(y_test, y_pred, y_probability)
            cv_summary = {
                "cv_roc_auc_mean": float(search.best_score_),
                "cv_roc_auc_std": float(search.cv_results_["std_test_roc_auc"][search.best_index_]),
                "cv_accuracy_mean": float(search.cv_results_["mean_test_accuracy"][search.best_index_]),
                "cv_precision_mean": float(search.cv_results_["mean_test_precision"][search.best_index_]),
                "cv_recall_mean": float(search.cv_results_["mean_test_recall"][search.best_index_]),
            }
            candidate_path = artifacts_dir / f"{model_name.lower()}_cv_results.csv"
            pd.DataFrame(search.cv_results_).to_csv(candidate_path, index=False)
            plot_paths = _save_evaluation_plots(
                model_name, y_test, y_pred, y_probability, figures_dir
            )
            importance_path = _save_feature_importance(fitted_model, model_name, figures_dir)
            if importance_path:
                plot_paths.append(importance_path)
            report = classification_report(
                y_test, y_pred, labels=[0, 1], target_names=["absence", "presence"],
                output_dict=True, zero_division=0,
            )
            report_path = artifacts_dir / f"{model_name.lower()}_classification_report.json"
            report_path.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
            result = {
                "model_name": model_name,
                "best_params": search.best_params_,
                "best_cv_roc_auc": float(search.best_score_),
                "cv_summary": cv_summary,
                "holdout_metrics": holdout_metrics,
                "classification_report": report,
            }
            all_results[model_name] = result

            if mlflow is not None:
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("cv_strategy", "5-fold StratifiedKFold; shuffle=True; random_state=42")
                mlflow.log_param("holdout_test_size", 0.20)
                for key, value in search.best_params_.items():
                    mlflow.log_param(f"best_{key}", str(value))
                mlflow.log_metrics({**cv_summary, **{f"test_{k}": v for k, v in holdout_metrics.items()}})
                mlflow.log_dict(result, f"reports/{model_name.lower()}_metrics.json")
                mlflow.log_artifact(str(candidate_path), artifact_path="cv_results")
                mlflow.log_artifact(str(report_path), artifact_path="evaluation")
                for plot_path in plot_paths:
                    mlflow.log_artifact(str(plot_path), artifact_path="plots")
                mlflow.sklearn.log_model(fitted_model, artifact_path="model")

            selection_rows.append({
                "model_name": model_name,
                "cv_roc_auc_mean": cv_summary["cv_roc_auc_mean"],
                "cv_roc_auc_std": cv_summary["cv_roc_auc_std"],
                **{f"test_{key}": value for key, value in holdout_metrics.items()},
                "best_params": json.dumps(search.best_params_, default=_json_default, sort_keys=True),
            })

    selection = pd.DataFrame(selection_rows).sort_values("cv_roc_auc_mean", ascending=False)
    selection_path = reports_dir / "model_selection.csv"
    selection.to_csv(selection_path, index=False)
    best_name = str(selection.iloc[0]["model_name"])
    best_result = all_results[best_name]
    # Refit on the training split only; the holdout set remains an honest final evaluation.
    best_search_model = build_pipeline(best_name)
    best_search_params = best_result["best_params"]
    best_search_model.set_params(**best_search_params)
    best_search_model.fit(X_train, y_train)
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_search_model, model_path)

    summary = {
        "dataset_path": str(input_path),
        "row_count": int(len(frame)),
        "feature_count": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "target_mapping": {"0": "absence", "1": "presence"},
        "split": {"test_size": 0.20, "random_state": RANDOM_STATE, "stratified": True},
        "cross_validation": {"folds": 5, "shuffle": True, "random_state": RANDOM_STATE, "selection_metric": "roc_auc"},
        "models": all_results,
        "model_selection": selection.to_dict(orient="records"),
        "selected_model": best_name,
        "selected_model_path": str(model_path),
        "limitations": [
            "Small historical dataset; results are not representative of modern clinical practice.",
            "This educational model is not a medical device and must not be used for diagnosis or treatment.",
        ],
    }
    (reports_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, default=_json_default), encoding="utf-8"
    )
    metadata = {
        "model_path": str(model_path),
        "model_name": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "target_mapping": {"0": "absence", "1": "presence"},
        "random_state": RANDOM_STATE,
    }
    (artifacts_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps({"selected_model": best_name, "model_path": str(model_path), "selection": summary["model_selection"]}, indent=2))
    return summary

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/processed/heart_disease_clean.csv")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--mlflow-dir", default="mlruns")
    parser.add_argument("--model-path", default="models/heart_disease_model.joblib")
    args = parser.parse_args()
    run_training(
        args.input, args.artifacts_dir, args.reports_dir, args.mlflow_dir, args.model_path
    )

if __name__ == "__main__":
    main()
