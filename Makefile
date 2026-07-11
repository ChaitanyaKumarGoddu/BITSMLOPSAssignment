# Convenience targets. On Windows run these via `make` (Git Bash / WSL) or
# copy the underlying commands. PY points at the venv interpreter.
PY ?= python

.PHONY: help install data eda train test lint format mlflow-ui serve docker-build docker-run compose-up clean

help:
	@echo "Targets:"
	@echo "  install       Install dev + runtime dependencies"
	@echo "  data          Download + preprocess the dataset"
	@echo "  eda           Generate EDA figures"
	@echo "  train         Train models (MLflow tracked) and save best model"
	@echo "  test          Run pytest with coverage"
	@echo "  lint          Run flake8"
	@echo "  format        Apply black + isort"
	@echo "  mlflow-ui     Launch MLflow UI at http://localhost:5000"
	@echo "  serve         Run the API locally at http://localhost:8000"
	@echo "  docker-build  Build the API Docker image"
	@echo "  docker-run    Run the API container on port 8000"
	@echo "  compose-up    Start API + Prometheus + Grafana"

install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements-dev.txt

data:
	$(PY) -m src.data.download
	$(PY) -m src.data.preprocess

eda:
	$(PY) -m src.data.eda

train:
	$(PY) -m src.models.train

test:
	$(PY) -m pytest

lint:
	$(PY) -m flake8 src tests

format:
	$(PY) -m black src tests
	$(PY) -m isort src tests

mlflow-ui:
	$(PY) -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

serve:
	$(PY) -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t heart-disease-api:latest .

docker-run:
	docker run --rm -p 8000:8000 heart-disease-api:latest

compose-up:
	docker compose up --build

clean:
	rm -rf .pytest_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +
