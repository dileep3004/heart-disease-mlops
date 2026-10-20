"""Generate the 10-page assignment report; run after training to insert measured results."""

from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "MLOps_Assignment_01_Report.docx"
METRICS_PATH = ROOT / "reports" / "metrics.json"
ARCH_IMAGE = ROOT / "docs" / "architecture.png"

def shade_cell(cell, fill: str):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)

def set_cell_text(cell, text: str, bold=False, color=None, size=8.5):
    cell.text = ""
    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.space_after = Pt(1)
    run = paragraph.add_run(str(text))
    run.bold = bold
    run.font.size = Pt(size)
    if color:
        run.font.color.rgb = RGBColor(*color)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER

def add_table(doc, headers, rows, widths=None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Light Shading Accent 1"
    for index, header in enumerate(headers):
        set_cell_text(table.rows[0].cells[index], header, bold=True, color=(255, 255, 255), size=8.5)
        shade_cell(table.rows[0].cells[index], "1F4E78")
    for row in rows:
        cells = table.add_row().cells
        for index, value in enumerate(row):
            set_cell_text(cells[index], value, size=8)
    if widths:
        for row in table.rows:
            for idx, width in enumerate(widths):
                row.cells[idx].width = Inches(width)
    doc.add_paragraph().paragraph_format.space_after = Pt(1)
    return table

def add_bullets(doc, items):
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(2)
        p.add_run(item)

def add_body(doc, text):
    p = doc.add_paragraph(text)
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.line_spacing = 1.05
    return p

def add_page_title(doc, text, subtitle=None):
    doc.add_heading(text, level=1)
    if subtitle:
        p = doc.add_paragraph(subtitle)
        p.paragraph_format.space_after = Pt(7)
        for run in p.runs:
            run.italic = True
            run.font.color.rgb = RGBColor(89, 89, 89)

def add_page_break(doc):
    doc.add_page_break()

def build():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else None
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(0.62)
    section.bottom_margin = Inches(0.58)
    section.left_margin = Inches(0.72)
    section.right_margin = Inches(0.72)
    section.header_distance = Inches(0.28)
    section.footer_distance = Inches(0.28)
    styles = doc.styles
    styles["Normal"].font.name = "Aptos"
    styles["Normal"].font.size = Pt(9.2)
    styles["Normal"].paragraph_format.space_after = Pt(4)
    for style_name, size in [("Title", 27), ("Heading 1", 17), ("Heading 2", 11.5), ("Heading 3", 10)]:
        styles[style_name].font.name = "Aptos Display" if style_name != "Normal" else "Aptos"
        styles[style_name].font.size = Pt(size)
        styles[style_name].font.color.rgb = RGBColor(31, 78, 121)
    header = section.header.paragraphs[0]
    header.text = "MLOps Assignment 01  |  UCI Heart Disease Classifier"
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    for run in header.runs:
        run.font.size = Pt(8)
        run.font.color.rgb = RGBColor(100, 100, 100)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = footer.add_run("Educational prototype - not a diagnostic tool  |  Page ")
    r.font.size = Pt(8)
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)

    # Page 1: title and summary
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(65)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("MACHINE LEARNING OPERATIONS")
    run.bold = True
    run.font.size = Pt(13)
    run.font.color.rgb = RGBColor(31, 78, 121)
    p = doc.add_paragraph(style="Title")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run("Assignment 01\nEnd-to-End Heart Disease ML System")
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run("Data acquisition | Experiments | CI/CD | Docker | API | Monitoring").italic = True
    doc.add_paragraph(" ")
    add_table(doc, ["Item", "Implementation"], [
        ["Dataset", "UCI Heart Disease - processed Cleveland subset"],
        ["Target", "Binary: original 0 -> absence; original 1-4 -> presence"],
        ["Models", "Logistic Regression and Random Forest"],
        ["Experiment tracking", "MLflow runs, params, metrics, CV tables, plots, model artifacts"],
        ["Serving / monitoring", "FastAPI + Docker + Prometheus + Grafana; Kubernetes manifests"],
    ], [1.5, 5.5])
    doc.add_heading("Executive summary", level=2)
    add_body(doc, "This project applies a reproducible MLOps workflow to a small historical tabular dataset. A downloader extracts the official UCI file, a cleaning step normalizes missing-value markers and derives the binary target, and an EDA script generates class-balance, feature-distribution, missingness, and correlation plots. Two scikit-learn pipelines are tuned using stratified cross-validation and evaluated on a held-out test set. MLflow tracks experiments, while FastAPI, Docker, CI, Kubernetes manifests, request logs, Prometheus metrics, and a provisioned Grafana dashboard cover delivery and operational visibility.")
    add_body(doc, "Important: all reported performance values must come from a successful run on the downloaded dataset. This report generator inserts values from reports/metrics.json when available; it does not manufacture metrics when the external dataset has not been downloaded.")
    add_page_break(doc)

    # Page 2: problem and assumptions
    add_page_title(doc, "1. Problem, Dataset, and Assumptions")
    add_body(doc, "The assignment asks for a binary classifier that consumes patient health features and predicts whether heart disease is present, then deploys the model as a cloud-ready monitored API. This is implemented as a coursework demonstration, not a clinically validated prediction system.")
    doc.add_heading("Dataset source and schema", level=2)
    add_body(doc, "The official UCI Heart Disease dataset page describes four source databases and 303 instances in the processed Cleveland subset. It also explains that the original goal field takes values 0 to 4 and that published Cleveland experiments commonly distinguish 0 (absence) from 1-4 (presence). The implementation uses the no-header processed Cleveland file from the UCI archive and declares 13 predictor columns plus the original goal field.")
    add_table(doc, ["Decision", "Reason / consequence"], [
        ["Cleveland subset", "Keeps to the classic 303-record processed file, suitable for this assignment's scope."],
        ["Binary target", "Maps original goal 0 to 0 and any positive goal to 1."],
        ["Missing features", "`?` is normalized to NaN; imputation is fit inside CV/training pipelines."],
        ["Train/test protocol", "Stratified 80/20 split, random_state=42; test set is not used for tuning."],
        ["Model selection", "5-fold StratifiedKFold; mean validation ROC-AUC is the selection metric."],
        ["Reproducibility", "A fixed random seed, parameter grids, requirements file, and data hash manifest."],
    ], [1.6, 5.4])
    doc.add_heading("Known limitations", level=2)
    add_bullets(doc, [
        "Historical, small dataset; the split metrics have high sampling uncertainty and are not evidence of clinical utility.",
        "Feature names and coded values need domain context; input validation enforces basic ranges but is not medical validation.",
        "The API intentionally returns an educational model estimate and a clear not-for-diagnosis disclaimer.",
    ])
    add_page_break(doc)

    # Page 3: architecture
    add_page_title(doc, "2. System Architecture", "Data plane, model lifecycle, and delivery path")
    add_body(doc, "The workflow separates offline model development from online inference. The raw archive is normalized to a stable tabular schema, and preprocessing stays attached to each estimator so the serving service can load one serializable object and apply the same transformations that were fit during training.")
    if ARCH_IMAGE.exists():
        doc.add_picture(str(ARCH_IMAGE), width=Inches(6.8))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_heading("Principal components", level=2)
    add_table(doc, ["Component", "Responsibility"], [
        ["Data/EDA", "Fetch, clean, record provenance, summarize missingness and distributions."],
        ["Training", "Grid search for two pipelines, CV and holdout metrics, plots and model selection."],
        ["MLflow", "Model-family runs with parameters, metrics, cross-validation tables, evaluation plots and model artifacts."],
        ["Serving", "FastAPI schema validation, prediction, probability, health/readiness routes and safe request logs."],
        ["Operations", "Docker image, Compose-based Prometheus/Grafana stack, Kubernetes Deployment/Service."],
        ["CI", "Ruff, pytest, data acquisition, EDA, training, and artifact upload on GitHub Actions."],
    ], [1.35, 5.65])
    add_page_break(doc)

    # Page 4: acquisition and EDA
    add_page_title(doc, "3. Data Acquisition, Cleaning, and EDA")
    add_body(doc, "Run `python -m heart_mlops.data --output data/processed/heart_disease_clean.csv`. The downloader retrieves the UCI ZIP archive, locates the processed Cleveland file, extracts the raw no-header CSV, writes a named cleaned dataset, and saves a JSON provenance manifest. The manifest records source URL, raw-file SHA-256, row count, feature count, missing feature cells, and class counts.")
    doc.add_heading("Cleaning rules", level=2)
    add_bullets(doc, [
        "Use an explicit 14-column schema rather than relying on column inference.",
        "Normalize `?` markers and invalid numeric strings to missing values.",
        "Drop rows with missing targets; derive `target = 1` when original `num > 0`, otherwise 0.",
        "Drop exact duplicate rows and persist features in a fixed order; preserve feature NaNs for train-fold imputation.",
    ])
    doc.add_heading("EDA outputs", level=2)
    add_table(doc, ["Plot", "Question answered"], [
        ["Numeric histograms by target", "How do the five continuous features distribute by class?"],
        ["Target balance", "How many records map to absence vs presence?"],
        ["Numeric correlation heatmap", "What pairwise linear relationships appear among continuous measurements?"],
        ["Missingness bars", "Which feature fields need imputation and how many cells are missing?"],
    ], [2.1, 4.9])
    add_body(doc, "Execute `python -m heart_mlops.eda --input data/processed/heart_disease_clean.csv --output reports/figures`. Numeric-only correlation is chosen to avoid presenting ordinal/coded categorical values as continuous measurements. Interpret the plot as exploratory association, not causation.")
    add_body(doc, "Observed values are generated in `reports/eda_summary.json`; do not copy general expected class ratios into the submission. Inspect the images after download and write two or three dataset-specific observations based on the actual output.")
    add_page_break(doc)

    # Page 5: feature engineering and modeling
    add_page_title(doc, "4. Feature Engineering and Model Development")
    add_body(doc, "The 13 fields are separated into five continuous/numeric measurements (age, trestbps, chol, thalach, oldpeak) and eight coded categorical fields (sex, cp, fbs, restecg, exang, slope, ca, thal). The data are not globally scaled or imputed before splitting. Instead, each model contains a ColumnTransformer fitted only on training folds.")
    add_table(doc, ["Step", "Logistic Regression", "Random Forest"], [
        ["Numeric missing values", "Median imputation", "Median imputation"],
        ["Categorical missing values", "Most-frequent imputation", "Most-frequent imputation"],
        ["Categorical encoding", "OneHotEncoder, unknown categories ignored", "OneHotEncoder, unknown categories ignored"],
        ["Numeric scaling", "StandardScaler", "No scaling"],
        ["Hyperparameter grid", "C in {0.1, 1, 10}; class_weight in {None, balanced}", "n_estimators in {100, 250}; max_depth in {None, 6}; min_samples_leaf in {1, 3}; class_weight in {None, balanced}"],
    ], [1.35, 2.8, 2.85])
    doc.add_heading("Protocol", level=2)
    add_bullets(doc, [
        "Use a stratified 80/20 train/test split with random_state=42.",
        "Tune using 5-fold StratifiedKFold with shuffle=True and random_state=42.",
        "Score candidate configurations with ROC-AUC, accuracy, precision, and recall; refit the best mean ROC-AUC configuration.",
        "Report holdout accuracy, precision, recall, F1, and ROC-AUC plus the confusion matrix, ROC curve, classification report, and transformed feature importance.",
        "Select the model family by cross-validation ROC-AUC, not by searching for the best-looking holdout score." ,
    ])
    add_body(doc, "ROC-AUC captures ranking quality across thresholds; precision describes the fraction of positive predictions that are correct; recall describes positive-class coverage. These metrics can conflict, so results should be discussed in context and not reduced to accuracy alone.")
    add_page_break(doc)

    # Page 6: results
    add_page_title(doc, "5. Experiment Results and Model Selection")
    if metrics:
        rows = []
        for model_name, result in metrics.get("models", {}).items():
            cv = result.get("cv_summary", {})
            holdout = result.get("holdout_metrics", {})
            rows.append([
                model_name,
                f"{cv.get('cv_roc_auc_mean', float('nan')):.3f} +/- {cv.get('cv_roc_auc_std', float('nan')):.3f}",
                f"{holdout.get('accuracy', float('nan')):.3f}",
                f"{holdout.get('precision', float('nan')):.3f}",
                f"{holdout.get('recall', float('nan')):.3f}",
                f"{holdout.get('roc_auc', float('nan')):.3f}",
            ])
        add_table(doc, ["Model", "CV ROC-AUC mean +/- SD", "Test acc.", "Test prec.", "Test recall", "Test ROC-AUC"], rows, [1.2, 1.45, 0.85, 0.85, 0.9, 1.0])
        add_body(doc, f"Selected family: {metrics.get('selected_model', 'not recorded')}. The selection is based on the highest 5-fold mean validation ROC-AUC. The complete measured report, best parameters, classification reports, and data split metadata are in `reports/metrics.json`; per-configuration CV results are in the `artifacts/` directory and MLflow.")
    else:
        add_body(doc, "Results pending execution. The data archive could not be downloaded in the report-generation environment, so no scores are guessed or presented as if measured. After the UCI download succeeds, run the training command in the README and execute `python scripts/generate_report.py` again. The regenerated table will be filled from `reports/metrics.json`.")
        add_table(doc, ["Model", "CV ROC-AUC", "Test accuracy", "Test precision", "Test recall", "Test ROC-AUC"], [
            ["Logistic Regression", "Run pipeline", "Run pipeline", "Run pipeline", "Run pipeline", "Run pipeline"],
            ["Random Forest", "Run pipeline", "Run pipeline", "Run pipeline", "Run pipeline", "Run pipeline"],
        ], [1.25, 1.05, 1.05, 1.05, 1.05, 1.05])
    doc.add_heading("How to interpret the result", level=2)
    add_bullets(doc, [
        "Compare cross-validation mean and standard deviation to assess both average quality and fold sensitivity.",
        "Compare holdout precision and recall to show the practical trade-off for classifying the positive class.",
        "Use the confusion matrix to identify false positives and false negatives; do not infer medical harm or benefit from this small dataset alone.",
        "Use feature-importance plots as model diagnostics, not as causal explanations.",
    ])
    add_page_break(doc)

    # Page 7: MLflow and reproducibility
    add_page_title(doc, "6. Experiment Tracking and Reproducibility")
    add_body(doc, "The training script creates an MLflow experiment named `heart-disease-classification`, with one parent run per model family and a nested child run for every GridSearchCV parameter configuration. Child runs log the candidate hyperparameters and the mean/standard deviation for validation ROC-AUC, accuracy, precision, and recall. Each parent run includes selected hyperparameters, split/CV settings, final holdout metrics, a fitted scikit-learn pipeline, JSON evaluation summary, classification report, plot artifacts, and the complete CV candidate-results table.")
    add_table(doc, ["Artifact", "Purpose"], [
        ["`models/heart_disease_model.joblib`", "Serializable preprocessing + classifier pipeline used by the API."],
        ["`artifacts/*_cv_results.csv`", "Per-configuration mean/std train-fold metrics and hyperparameters."],
        ["`artifacts/*_classification_report.json`", "Precision, recall, F1 and support by target class."],
        ["`artifacts/model_metadata.json`", "Selected model family, feature order and target mapping."],
        ["`reports/metrics.json`", "Consolidated model metrics, selection protocol and limitations."],
        ["`data/processed/*.manifest.json`", "Dataset provenance, source and SHA-256 digest."],
    ], [2.55, 4.45])
    doc.add_heading("Reproduce the run", level=2)
    add_body(doc, "Create a clean Python 3.11 virtual environment, install `requirements.txt`, install the project with `pip install -e .`, download the data, then run the EDA and training module commands in the README. `random_state=42` is used for splitting, CV shuffling, and estimator seeds where applicable. Exact metric values can still vary across library versions or numerical backends, so the installed environment and final output files should be retained.")
    add_body(doc, "Run `mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000` to inspect the local tracking store. The default store is local for the demonstration; a shared remote tracking server and artifact store would be required for multi-developer production use.")
    add_page_break(doc)

    # Page 8: CI/CD and Docker
    add_page_title(doc, "7. Automated Testing, CI/CD, and Containerization")
    doc.add_heading("Testing and linting", level=2)
    add_body(doc, "Ruff checks style and common defects. Pytest covers target binarization, missing-value preservation, pipeline fitting and probability output, API health and Prometheus metrics, prediction payload validation, and successful inference using a small deterministic test model. The test suite does not rely on external dataset access, allowing quick deterministic feedback on a laptop or CI runner.")
    doc.add_heading("GitHub Actions flow", level=2)
    add_table(doc, ["Stage", "Behavior / failure condition"], [
        ["Checkout + Python 3.11", "Starts with a clean runner and pip install from requirements."],
        ["Ruff", "Workflow fails if selected lint rules report errors."],
        ["Pytest", "Workflow fails if a unit or API test fails."],
        ["Dataset + EDA", "Workflow downloads official data then creates plots/summary."],
        ["Train + tracking", "Tuning, held-out evaluation and MLflow artifacts run after tests."],
        ["Upload artifacts", "Stores reports, model/evaluation artifacts and processed CSV for 14 days."],
    ], [1.75, 5.25])
    doc.add_heading("Docker packaging", level=2)
    add_body(doc, "The Dockerfile uses Python 3.11 slim and a separate `requirements-api.txt` to avoid installing MLflow and developer tooling in the serving image. It copies the service source and the trained joblib artifact, sets `MODEL_PATH`, exposes port 8000, and includes a health check. Train the model first; the Docker build deliberately fails if the expected artifact does not exist.")
    add_body(doc, "Run `docker compose up --build -d`; verify the API docs at `http://localhost:8000/docs`, then check `docker compose ps` and logs. Never put credentials, private data, or a production patient record in source control or image layers.")
    add_page_break(doc)

    # Page 9: deployment and monitoring
    add_page_title(doc, "8. Deployment, Logging, and Monitoring")
    doc.add_heading("API contract", level=2)
    add_table(doc, ["Endpoint", "Purpose"], [
        ["`GET /health`", "Process liveness; does not imply a trained model is present."],
        ["`GET /ready`", "Readiness check for the model artifact."],
        ["`POST /predict`", "Validates all 13 fields and returns class, class-1 probability and confidence."],
        ["`GET /metrics`", "Prometheus counters and request latency histogram."],
    ], [1.55, 5.45])
    doc.add_heading("Operational metrics", level=2)
    add_bullets(doc, [
        "Request counter labelled by method, path, and HTTP status code.",
        "Latency histogram supporting p95 latency calculation.",
        "Prediction counter split by predicted class, without recording patient features.",
        "Request logs contain method/path/status/elapsed milliseconds, not the input body.",
    ])
    add_body(doc, "Docker Compose starts Prometheus and Grafana and provisions a dashboard for request rate, p95 latency, prediction counts and HTTP 5xx rates. Generate sample traffic from Swagger or the supplied client, wait at least two scrape intervals, and verify Prometheus's target status is UP. The dashboard is operational visibility, not data/model drift monitoring; no ground-truth feedback loop exists in this static demo.")
    doc.add_heading("Kubernetes", level=2)
    add_body(doc, "`k8s/deployment.yaml` defines two API replicas, resource requests/limits, readiness and liveness probes, scrape annotations, and a LoadBalancer Service. For Minikube, load the image into the cluster and use `kubectl port-forward svc/heart-disease-api 8000:80`. A real external LoadBalancer depends on support in the target environment; a manifest alone does not prove public cloud deployment.")
    doc.add_heading("Production gaps", level=2)
    add_body(doc, "Before any non-demo deployment, add HTTPS/TLS, authentication and authorization, rate limiting, network restrictions, secret management, audit controls, privacy review, model/version registry, alerting, monitoring of input drift and observed outcomes, and governance approvals. This health-related dataset is not clinically valid for decision support.")
    add_page_break(doc)

    # Page 10: final execution guide/rubric/references
    add_page_title(doc, "9. Demo Runbook, Rubric Mapping, and References")
    add_body(doc, "A complete demonstration should be recorded from a clean environment and should show the outputs rather than only the source code. Execute the following commands from the repository root in the order shown.")
    add_table(doc, ["Step", "Command / evidence"], [
        ["1. Setup", "`pip install -r requirements.txt` then `pip install -e .`"],
        ["2. Acquire data", "`python -m heart_mlops.data --output data/processed/heart_disease_clean.csv`"],
        ["3. EDA", "`python -m heart_mlops.eda --input data/processed/heart_disease_clean.csv --output reports/figures`"],
        ["4. Train", "`python -m heart_mlops.train --input data/processed/heart_disease_clean.csv --artifacts-dir artifacts --reports-dir reports --mlflow-dir mlruns`"],
        ["5. Report", "`python scripts/generate_report.py` (reads measured metrics when present)"],
        ["6. Quality gates", "`ruff check src tests scripts` and `pytest`"],
        ["7. API", "`uvicorn heart_mlops.api:app --reload --port 8000`; test `/docs`, `/predict`, `/metrics`"],
        ["8. Monitor", "`docker compose up --build -d`; capture Prometheus and Grafana screenshots"],
        ["9. Deploy/CI", "Push to GitHub to run Actions; optionally apply the Minikube manifest and capture `kubectl` output."],
    ], [1.35, 5.65])
    doc.add_heading("Assignment deliverable map", level=2)
    add_body(doc, "EDA and data download are implemented in `data.py` and `eda.py`; feature engineering, two models and CV tuning are in `modeling.py` and `train.py`; MLflow logs experiments; joblib + requirements provide a reusable model and environment; pytest/Ruff/GitHub Actions cover automated quality checks; Docker and Kubernetes files implement packaging/deployment; request logs and Prometheus/Grafana cover monitoring; this report plus architecture diagram cover documentation. Record your actual repository URL and public API URL only after creating/validating them.")
    doc.add_heading("References", level=2)
    add_body(doc, "[1] UCI Machine Learning Repository, Heart Disease, https://archive.ics.uci.edu/dataset/45/heart+disease (dataset description, class definition, missing values and CC BY 4.0 attribution).")
    add_body(doc, "[2] MLflow tracking/autologging, https://mlflow.org/docs/latest/ml/tracking/autolog/ . [3] FastAPI testing, https://fastapi.tiangolo.com/tutorial/testing/ .")
    add_body(doc, "[4] Prometheus Python client, https://prometheus.github.io/client_python/ . [5] Kubernetes external LoadBalancer services, https://kubernetes.io/docs/tasks/access-application-cluster/create-external-load-balancer/ .")
    add_body(doc, "Final evidence checklist: add your GitHub repository URL, replace metrics only with values from the real run, save EDA/MLflow/Swagger/Docker/Grafana/GitHub Actions/Kubernetes screenshots as relevant, and include a short screen recording of the end-to-end flow.")

    doc.save(OUTPUT)
    print(f"Wrote {OUTPUT}")
    if metrics:
        print("Included measured model metrics from reports/metrics.json")
    else:
        print("No reports/metrics.json found: results page clearly marked pending; rerun after training.")

if __name__ == "__main__":
    build()
