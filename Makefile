.PHONY: install data eda train test lint api monitor-down

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

data:
	python -m heart_mlops.data --output data/processed/heart_disease_clean.csv

eda:
	python -m heart_mlops.eda --input data/processed/heart_disease_clean.csv --output reports/figures

train:
	python -m heart_mlops.train --input data/processed/heart_disease_clean.csv --artifacts-dir artifacts --reports-dir reports --mlflow-dir mlruns

test:
	pytest

lint:
	ruff check src tests scripts

api:
	uvicorn heart_mlops.api:app --reload --host 127.0.0.1 --port 8000

monitor-down:
	docker compose down
