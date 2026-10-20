# Heart Disease MLOps Assignment 01

End-to-end Python implementation for the assignment brief: UCI data acquisition, EDA,
preprocessing, two classifiers, cross-validation and parameter tuning, MLflow tracking,
reproducible model packaging, automated tests, GitHub Actions CI, Docker, Kubernetes manifests,
API logging, and Prometheus/Grafana monitoring.

> **Academic/health note:** Use this as a learning scaffold, understand each step, and personalize
> your own modelling choices and findings before submission. This historical dataset/model is
> for coursework demonstration only and must not be used to diagnose or treat a person.

## Explicit assumptions

1. Dataset: use the official UCI Heart Disease archive's processed Cleveland subset because the
   assignment names the 14-column heart dataset and the Cleveland subset is the conventional
   303-row experimental dataset. UCI describes 13 input features and an original `num` target
   from 0 to 4.
2. Binary target: `num == 0 -> target 0 (absence)` and `num > 0 -> target 1 (presence)`.
3. Split: 80% training / 20% holdout test, stratified, `random_state=42`. The holdout is not used
   for tuning; 5-fold stratified CV chooses hyperparameters by mean ROC-AUC.
4. Baselines: Logistic Regression and Random Forest. Numeric columns are median-imputed;
   categorical codes are most-frequent-imputed and one-hot encoded. Logistic Regression also
   scales numeric columns. All preprocessing is inside `Pipeline`/`ColumnTransformer` to avoid
   fitting imputers/encoders/scalers on the holdout data.
5. Local artifact paths are relative to the repository root. Python 3.11 is the recommended
   runtime for broad package compatibility.
6. Cloud deployment cannot be completed without your cloud/Kubernetes account, container
   registry (if remote), and repository credentials. This project provides manifests and local
   execution steps; do not claim it is publicly deployed until you verify a real endpoint.

## Project structure

```text
src/heart_mlops/data.py       download, cleaning, dataset provenance
src/heart_mlops/eda.py        histograms, class balance, correlations, missingness
src/heart_mlops/modeling.py   reusable preprocessing + estimator factories
src/heart_mlops/train.py      GridSearchCV, holdout metrics, plots, MLflow, model artifact
src/heart_mlops/api.py        FastAPI /predict, /health, /ready, /metrics
scripts/predict_sample.py     sample HTTP prediction request
tests/                        unit and API tests
.github/workflows/ci.yml      lint -> test -> data -> EDA -> train -> upload artifacts
Dockerfile, docker-compose.yml
monitoring/                   Prometheus config and provisioned Grafana dashboard
k8s/deployment.yaml           Deployment + LoadBalancer Service
docs/architecture.mmd         architecture diagram source
```

## 1. Environment setup (pip)

Use Python 3.11. From the repository root:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# macOS/Linux instead:
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

If activation is blocked in PowerShell, use a terminal where script activation is allowed or run
`.venv\\Scripts\\python.exe -m pip ...` and `.venv\\Scripts\\python.exe -m ...` explicitly.

## 2. Download and clean the UCI dataset

Internet access is required for this one-time download:

```bash
python -m heart_mlops.data --output data/processed/heart_disease_clean.csv
```

Expected outputs:

- `data/raw/processed.cleveland.data`
- `data/processed/heart_disease_clean.csv`
- `data/processed/heart_disease_clean.manifest.json`

The script validates target conversion and records SHA-256 provenance. A `?` in feature fields
becomes a missing value; the missing value is not filled using the full dataset. The training
pipeline imputes it after each cross-validation split is formed.

## 3. EDA

```bash
python -m heart_mlops.eda --input data/processed/heart_disease_clean.csv --output reports/figures
```

Open the output PNGs and `reports/eda_summary.json`:

- `numeric_histograms.png`
- `class_balance.png`
- `numeric_correlation_heatmap.png` (continuous features only)
- `missing_values.png`

Write your own observations after reviewing the generated plots. Avoid claiming correlations or
class imbalance in advance; they must be supported by the downloaded CSV and plots.

## 4. Train, tune, evaluate, and track with MLflow

```bash
python -m heart_mlops.train \
  --input data/processed/heart_disease_clean.csv \
  --artifacts-dir artifacts \
  --reports-dir reports \
  --mlflow-dir mlruns
```

This evaluates both candidate models using the same stratified training/test split and tunes
them using stratified 5-fold CV. The scoring dictionary includes ROC-AUC, accuracy, precision,
and recall. The best CV ROC-AUC selects the final pipeline; the holdout score is reporting-only.

Outputs include `models/heart_disease_model.joblib`, `reports/metrics.json`,
`reports/model_selection.csv`, `artifacts/*_cv_results.csv`, classification reports, and model
evaluation/feature-importance plots. MLflow logs one parent run per model family plus a nested run
for every parameter configuration. Parent runs contain selected params, CV/holdout metrics, result
tables, evaluation plots and fitted models; candidate runs contain their parameters, fold metrics,
per-candidate summary JSON and CV plot.

View experiment tracking in another terminal:

```bash
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000`. If you set `MLFLOW_TRACKING_URI`, use that same URI for the UI.

## 5. Automated tests and lint

```bash
ruff check src tests scripts
pytest
```

Tests use synthetic fixtures for deterministic fast checks and therefore do not download the UCI
dataset. The pipeline test checks missing-value handling, probability shape, and binary labels;
API tests cover health/metrics, schema validation, and an injected test model.

## 6. Run the API locally

Ensure the training step created `models/heart_disease_model.joblib`, then run:

```bash
uvicorn heart_mlops.api:app --reload --host 127.0.0.1 --port 8000
```

Browse `http://127.0.0.1:8000/docs`. Verify health/readiness/metrics in a second terminal:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
curl http://127.0.0.1:8000/metrics
python scripts/predict_sample.py
```

Example JSON payload (all 13 fields are required):

```json
{
  "age": 57, "sex": 1, "cp": 2, "trestbps": 130, "chol": 236,
  "fbs": 0, "restecg": 0, "thalach": 174, "exang": 0,
  "oldpeak": 0.0, "slope": 2, "ca": 0, "thal": 3
}
```

`POST /predict` responds with the binary prediction, label, model-estimated probability of class
1, maximum class confidence, model version, and a medical-use disclaimer. It does not log patient
feature values. `/health` checks process health, `/ready` checks model-artifact availability, and
`/metrics` exposes Prometheus counters and latency histograms.

## 7. Docker + Prometheus + Grafana demo

Train the model before building the image:

```bash
docker compose up --build -d
docker compose ps
docker compose logs -f api
```

URLs:

- FastAPI docs: `http://localhost:8000/docs`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (local demo credentials are `admin` / `admin`; change them
  for any non-demo environment)

In Swagger UI, call `POST /predict` a few times. In Prometheus, open **Status -> Targets** and
verify the API target is `UP`. In Grafana, open the provisioned **Heart Disease API Monitoring**
dashboard. Allow 15-30 seconds for Prometheus to scrape metrics. The charts show request rate,
p95 latency, prediction counts, and 5xx error rate.

Stop services with `docker compose down`. This image contains the serialized model generated by
your training run; rebuilding after retraining packages the new model.

## 8. Local Kubernetes (optional deployment evidence)

Requirements: Docker, `kubectl`, and Minikube or a configured cluster.

```bash
docker build -t heart-disease-api:latest .
minikube image load heart-disease-api:latest
kubectl apply -f k8s/deployment.yaml
kubectl rollout status deployment/heart-disease-api
kubectl get pods,svc
kubectl port-forward svc/heart-disease-api 8000:80
```

Visit `http://127.0.0.1:8000/docs` and test `/predict`. With Minikube, `minikube tunnel` may be
needed to provide a LoadBalancer IP; check the service status rather than assuming it is public.
A cloud public endpoint requires suitable cloud infrastructure and may incur charges.

## 9. CI/CD

`.github/workflows/ci.yml` runs on pushes and pull requests: install, Ruff lint, pytest, official
UCI acquisition, EDA, model tuning/tracking, and upload of reports/model artifacts. It fails on
lint/test/training errors and uploads outputs for 14 days. GitHub Actions must be enabled in the

## 10. Evidence, report, and video

Actual Running Screenshots can be seen here in `screenshots/`: EDA, MLflow runs, Swagger prediction,
Docker containers, Prometheus target, Grafana dashboard, successful GitHub Action, and Kubernetes
resources if you use Kubernetes.


Video: https://drive.google.com/file/d/1iz4v1glSiNL3mhgisnIm99jh02N5JP9l/view?usp=sharing


## API/schema note

The service is a classroom prototype, not a clinical tool. It has no authentication, rate limiting,
TLS termination, protected model registry, drift detection, or clinical validation. Do not expose it
to the public internet with real personal health data. Before any production use, add security,
privacy, data governance, model validation, fairness, monitoring, and human clinical review.

## References

1. UCI Machine Learning Repository, Heart Disease: https://archive.ics.uci.edu/dataset/45/heart+disease
2. MLflow tracking and sklearn autologging: https://mlflow.org/docs/latest/ml/tracking/autolog/
3. FastAPI testing guide: https://fastapi.tiangolo.com/tutorial/testing/
4. Prometheus Python client: https://prometheus.github.io/client_python/
5. Kubernetes LoadBalancer Service: https://kubernetes.io/docs/tasks/access-application-cluster/create-external-load-balancer/
