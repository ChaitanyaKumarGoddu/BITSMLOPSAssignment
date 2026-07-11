# Heart Disease Risk Prediction — End-to-End MLOps Pipeline

Predicts the risk of heart disease from patient clinical data and serves the model as a **cloud-ready, monitored API**. Built for the AIMLCZG523 MLOps Assignment 01.

This repository covers the full MLOps lifecycle: data acquisition → EDA → feature engineering → multi-model training with tuning → **MLflow experiment tracking** → reproducible packaging → **Pytest + GitHub Actions CI/CD** → **Dockerised FastAPI service** → **Kubernetes deployment** → **Prometheus + Grafana monitoring**.

---

## 1. Architecture

```
                         ┌────────────────────────────────────────────┐
                         │                Development                  │
                         │  download → preprocess → EDA → train (CV +  │
                         │  GridSearchCV) → MLflow (params/metrics/     │
                         │  artifacts) → model.pkl (sklearn Pipeline)   │
                         └───────────────┬────────────────────────────┘
                                         │  model.pkl + metadata
                                         ▼
   Git push ──► GitHub Actions ──►  lint ▸ pytest ▸ train ▸ docker build ▸ smoke test
                                         │
                                         ▼
                         ┌────────────────────────────────────────────┐
                         │            Docker image (FastAPI)           │
                         │   /predict  /health  /metrics  /docs        │
                         └───────────────┬────────────────────────────┘
                                         │  kubectl apply / helm install
                                         ▼
   ┌──────────────┐   scrape /metrics   ┌──────────────┐   dashboards   ┌──────────────┐
   │  Kubernetes  │ ──────────────────► │  Prometheus  │ ─────────────► │   Grafana    │
   │ Deployment×2 │                     └──────────────┘                └──────────────┘
   │  + LoadBal.  │
   └──────────────┘
```

A rendered diagram is in [`reports/figures/architecture.png`](reports/figures/) (regenerate with `python -m src.utils.architecture` if you edit it) and the report.

---

## 2. Project structure

```
heart-disease-mlops/
├── src/
│   ├── config.py                 # paths, feature schema, reproducibility knobs
│   ├── data/
│   │   ├── download.py           # Task 1: acquire dataset (ucimlrepo + URL fallback)
│   │   ├── preprocess.py         # Task 1: clean, binarise target, de-dup
│   │   └── eda.py                # Task 1: generate EDA figures
│   ├── features/pipeline.py      # Task 2/4: ColumnTransformer (impute+scale+encode)
│   ├── models/
│   │   ├── train.py              # Task 2/3/4: train, tune, MLflow, save best
│   │   └── evaluate.py           # metrics + ROC / confusion-matrix plots
│   └── api/
│       ├── main.py               # Task 6/8: FastAPI /predict + logging + metrics
│       └── schemas.py            # Pydantic request/response models
├── tests/                        # Task 5: pytest unit + API tests
├── notebooks/01_eda.ipynb        # Task 1: narrated EDA
├── .github/workflows/ci.yml      # Task 5: lint → test → train → docker build
├── Dockerfile                    # Task 6: multi-stage image
├── docker-compose.yml            # Task 8: API + Prometheus + Grafana
├── k8s/                          # Task 7: deployment.yaml + service.yaml
├── helm/heart-disease-api/       # Task 7: Helm chart (optional path)
├── monitoring/                   # Task 8: prometheus.yml + Grafana provisioning
├── reports/                      # Task 9: report + figures + screenshots
├── requirements.txt              # runtime deps (pinned, wheels for py3.11–3.14)
├── requirements-dev.txt          # + EDA / tests / lint / notebook
└── Makefile                      # convenience targets
```

---

## 3. Setup

Requires Python 3.11–3.14. From the repo root:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Git Bash / Linux / macOS:
source .venv/Scripts/activate      # or .venv/bin/activate on Linux/macOS

pip install --upgrade pip
pip install -r requirements-dev.txt
```

> The runtime image uses only `requirements.txt`; `requirements-dev.txt` adds EDA, tests, linting and Jupyter.

---

## 4. Reproduce the ML pipeline

```bash
python -m src.data.download      # -> data/raw/heart_disease_raw.csv (303 rows)
python -m src.data.preprocess    # -> data/processed/heart_disease_clean.csv
python -m src.data.eda           # -> reports/figures/eda_*.png
python -m src.models.train       # -> models/model.pkl + metadata + MLflow runs
```

Then browse the experiments:

```bash
python scripts/mlflow_ui.py --port 5000   # launcher with a Python 3.14 compat shim
# open http://localhost:5000
# (On Python <=3.13 the plain command also works:
#  mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000)
```

(Or just run `make data eda train`.)

---

## 5. Run the API locally

```bash
uvicorn src.api.main:app --reload --port 8000
```

- Swagger UI: <http://localhost:8000/docs>
- Health: <http://localhost:8000/health>
- Metrics: <http://localhost:8000/metrics>

Sample prediction:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

```json
{
  "prediction": 1,
  "label": "Heart disease",
  "confidence": 0.9635,
  "probability_disease": 0.9635,
  "model_name": "logistic_regression"
}
```

---

## 6. Docker (Task 6)

```bash
# Model must exist first (models/model.pkl) — run training once.
docker build -t heart-disease-api:latest .
docker run --rm -p 8000:8000 heart-disease-api:latest
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @sample_request.json
```

---

## 7. Monitoring stack (Task 8)

```bash
docker compose up --build
```

| Service    | URL                     | Notes                              |
|------------|-------------------------|------------------------------------|
| API        | http://localhost:8000   | FastAPI + `/metrics`               |
| Prometheus | http://localhost:9090   | scrapes the API every 10s          |
| Grafana    | http://localhost:3000   | admin/admin; dashboard auto-loaded |

The API logs every request (method, path, latency, status) and exposes a custom `heart_predictions_total{outcome=...}` counter plus standard HTTP latency/throughput metrics.

---

## 8. Kubernetes deployment (Task 7 — Docker Desktop)

Enable Kubernetes in **Docker Desktop → Settings → Kubernetes**, then:

```bash
docker build -t heart-disease-api:latest .     # image lives in the local daemon
kubectl apply -f k8s/
kubectl get pods,svc
# LoadBalancer is published on localhost with Docker Desktop:
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @sample_request.json
```

Helm alternative:

```bash
helm install heart ./helm/heart-disease-api
kubectl get pods,svc
```

Tear down: `kubectl delete -f k8s/` (or `helm uninstall heart`).

---

## 9. Tests, linting & CI/CD (Task 5)

```bash
pytest                 # unit + API tests with coverage
flake8 src tests       # lint
black --check src tests && isort --check-only src tests
```

`.github/workflows/ci.yml` runs on every push/PR: **lint → format check → pytest (with coverage artifact) → train model (uploads model + plots) → Docker build + container health smoke-test**. The pipeline fails on lint or test errors.

---

## 10. Task-to-file map (for grading)

| # | Task | Where |
|---|------|-------|
| 1 | Data + EDA | `src/data/`, `notebooks/01_eda.ipynb`, `reports/figures/eda_*.png` |
| 2 | Feature eng + models + tuning + CV | `src/features/pipeline.py`, `src/models/train.py` |
| 3 | MLflow tracking | `src/models/train.py`, `mlflow.db`, `mlartifacts/` |
| 4 | Packaging + reproducibility | `models/model.pkl`, `requirements*.txt`, sklearn Pipeline |
| 5 | CI/CD + tests | `tests/`, `.github/workflows/ci.yml` |
| 6 | Containerisation | `Dockerfile`, `/predict` endpoint |
| 7 | Deployment | `k8s/`, `helm/` |
| 8 | Monitoring | `src/api/main.py`, `docker-compose.yml`, `monitoring/` |
| 9 | Report | `reports/REPORT.md` |

See [`reports/REPORT.md`](reports/REPORT.md) for the full written report.
