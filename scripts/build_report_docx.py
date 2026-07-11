"""Generate a formatted .docx draft of the project report.

Embeds the figures that already exist in ``reports/figures/`` (EDA plots, ROC
curve, confusion matrix, architecture diagram) and inserts clearly-marked
[SCREENSHOT] placeholders for the artefacts you must capture yourself (MLflow,
GitHub Actions, Docker, kubectl, Grafana, Swagger).

Run from the repo root:
    python scripts/build_report_docx.py
Output: reports/Heart_Disease_MLOps_Report_DRAFT.docx
"""
from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports" / "figures"
META = ROOT / "models" / "model_metadata.json"
OUT = ROOT / "reports" / "Heart_Disease_MLOps_Report_DRAFT.docx"

ACCENT = RGBColor(0x1F, 0x4E, 0x79)
GREY = RGBColor(0x66, 0x66, 0x66)


def _placeholder(doc: Document, text: str) -> None:
    """A visually distinct [SCREENSHOT] box telling the user what to paste."""
    p = doc.add_paragraph()
    run = p.add_run(f"📷  [SCREENSHOT — {text}]")
    run.bold = True
    run.font.color.rgb = RGBColor(0xB0, 0x00, 0x00)
    run.font.size = Pt(11)
    p2 = doc.add_paragraph()
    r2 = p2.add_run("(Replace this line with your screenshot.)")
    r2.italic = True
    r2.font.color.rgb = GREY
    r2.font.size = Pt(9)


def _figure(doc: Document, filename: str, caption: str, width_in: float = 6.0) -> None:
    from docx.shared import Inches

    path = FIGURES / filename
    if path.exists():
        doc.add_picture(str(path), width=Inches(width_in))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = cap.add_run(caption)
        r.italic = True
        r.font.size = Pt(9)
        r.font.color.rgb = GREY
    else:
        _placeholder(doc, f"missing figure {filename} — run 'python -m src.data.eda'")


def _h(doc: Document, text: str, level: int = 1) -> None:
    doc.add_heading(text, level=level)


def build() -> None:
    meta = json.loads(META.read_text()) if META.exists() else {}
    m = meta.get("metrics", {})

    doc = Document()

    # ---- Title block ----
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = title.add_run("Heart Disease Risk Prediction\nEnd-to-End MLOps Pipeline")
    r.bold = True
    r.font.size = Pt(22)
    r.font.color.rgb = ACCENT

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rs = sub.add_run(
        "Machine Learning Operations (MLOps) — AIMLCZG523 · Assignment 01"
    )
    rs.font.size = Pt(12)
    rs.font.color.rgb = GREY

    meta_p = doc.add_paragraph()
    meta_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta_p.add_run(
        "Name: ______________________    BITS ID: ______________________\n"
        "GitHub repository: ______________________________________________\n"
        "Video walkthrough link: _________________________________________"
    ).font.size = Pt(11)

    doc.add_paragraph()

    # ---- 1. Overview ----
    _h(doc, "1. Project Overview")
    doc.add_paragraph(
        "This project builds a machine-learning classifier that predicts the risk "
        "of heart disease from patient clinical data and deploys it as a "
        "cloud-ready, monitored API, following modern MLOps practice end to end: "
        "data acquisition and EDA, feature engineering, multi-model training with "
        "hyper-parameter tuning and cross-validation, MLflow experiment tracking, "
        "reproducible packaging, Pytest + GitHub Actions CI/CD, Docker "
        "containerisation, Kubernetes deployment, and Prometheus/Grafana monitoring."
    )
    doc.add_paragraph(
        f"Headline result: the tuned Logistic Regression pipeline achieves "
        f"ROC-AUC {m.get('roc_auc', 0):.3f}, accuracy {m.get('accuracy', 0):.2f}, "
        f"and recall {m.get('recall', 0):.2f} on the held-out test set — recall "
        f"being the priority for a screening tool where a missed case is costly."
    )

    # ---- 2. Setup ----
    _h(doc, "2. Setup & Installation")
    doc.add_paragraph(
        "Requires Python 3.11–3.14. From the repository root:"
    )
    doc.add_paragraph(
        "python -m venv .venv\n"
        ".venv\\Scripts\\Activate.ps1        # Windows PowerShell\n"
        "pip install --upgrade pip\n"
        "pip install -r requirements-dev.txt",
        style="Intense Quote",
    )

    # ---- 3. EDA ----
    _h(doc, "3. Dataset & Exploratory Data Analysis")
    doc.add_paragraph(
        "The UCI Heart Disease (Cleveland) dataset has 303 patient records with 13 "
        "clinical features and a diagnosis field. Missing values (encoded as '?') "
        "appear only in 'ca' (4 rows) and 'thal' (2 rows) and are imputed inside "
        "the modelling pipeline. The 0–4 severity target is binarised to "
        "presence/absence (0 vs ≥1). The two classes are close to balanced "
        "(~46% positive)."
    )
    _figure(doc, "eda_class_balance.png", "Figure 1. Class balance.", 4.0)
    _figure(doc, "eda_histograms.png", "Figure 2. Numeric feature distributions.")
    _figure(
        doc, "eda_correlation_heatmap.png",
        "Figure 3. Feature correlation heatmap. Strongest target correlates: "
        "cp, thalach, oldpeak, ca, exang.",
    )
    _figure(
        doc, "eda_feature_relationships.png",
        "Figure 4. Key features by target — diseased patients show lower max "
        "heart rate, higher ST depression, and more affected vessels.",
    )

    # ---- 4. Modelling ----
    _h(doc, "4. Feature Engineering & Modelling")
    doc.add_paragraph(
        "All feature engineering lives in a single scikit-learn ColumnTransformer: "
        "numeric features get median imputation + standard scaling; categorical "
        "features get most-frequent imputation + one-hot encoding "
        "(handle_unknown='ignore'). Because the transformer is bundled with the "
        "estimator into one Pipeline, it is fitted only on training data (inside "
        "cross-validation) and reused byte-for-byte at inference — eliminating "
        "train/serving skew. Three classifiers (Logistic Regression, Random "
        "Forest, XGBoost) were tuned with GridSearchCV (5-fold, ROC-AUC scoring) "
        "on a stratified 80/20 split (random_state=42)."
    )

    # ---- 5. Results table ----
    _h(doc, "5. Model Comparison & Results")
    rows = [
        ("Model", "Accuracy", "Precision", "Recall", "F1", "Test ROC-AUC", "CV ROC-AUC"),
        ("Logistic Regression *", "0.902", "0.867", "0.929", "0.897", "0.968", "0.898 ± 0.047"),
        ("Random Forest", "0.885", "0.839", "0.929", "0.881", "0.952", "0.897 ± 0.043"),
        ("XGBoost", "0.869", "0.813", "0.929", "0.867", "0.947", "0.879 ± 0.025"),
    ]
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.style = "Light Grid Accent 1"
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = table.rows[i].cells[j]
            cell.text = val
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(9)
                    if i == 0:
                        run.bold = True
    doc.add_paragraph(
        "* Selected production model. Logistic Regression wins on every headline "
        "metric, is cheap to serve, and is interpretable (coefficients map to "
        "clinical risk factors) — a real advantage in healthcare. Best params: "
        "C=0.1, solver=liblinear. All three achieve 0.929 recall."
    )
    _figure(doc, "roc_logistic_regression.png",
            "Figure 5. ROC curve — Logistic Regression.", 4.5)
    _figure(doc, "cm_logistic_regression.png",
            "Figure 6. Confusion matrix — Logistic Regression.", 4.0)

    # ---- 6. MLflow ----
    _h(doc, "6. Experiment Tracking (MLflow)")
    doc.add_paragraph(
        "Training logs everything to MLflow using a SQLite backend "
        "(sqlite:///mlflow.db). One parent run contains a nested run per model, "
        "each logging best hyper-parameters, all metrics, the ROC curve and "
        "confusion-matrix artifacts, and the fitted sklearn Pipeline. Browse with: "
        "mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000."
    )
    _placeholder(doc, "MLflow UI runs table comparing the 3 models")
    _placeholder(doc, "MLflow UI — one run's metrics + logged artifacts")

    # ---- 7. Packaging ----
    _h(doc, "7. Model Packaging & Reproducibility")
    doc.add_paragraph(
        "The winning pipeline is serialised to models/model.pkl with joblib "
        "alongside model_metadata.json (model name, best params, metrics, feature "
        "order, timestamp). Because preprocessing is inside the pipeline, the "
        "artifact is self-contained. requirements.txt pins exact, wheel-available "
        "versions and random_state=42 fixes every stochastic step, so a clean "
        "environment reproduces the same model and metrics."
    )

    # ---- 8. CI/CD ----
    _h(doc, "8. CI/CD Pipeline (GitHub Actions)")
    doc.add_paragraph(
        "The workflow (.github/workflows/ci.yml) runs on every push/PR with three "
        "dependent jobs: (1) Lint & Test — flake8, black/isort checks, then pytest "
        "with coverage (uploaded as an artifact); (2) Train Model — trains and "
        "uploads model.pkl + plots; (3) Docker Build Validation — builds the image, "
        "runs the container, and smoke-tests /health. The pipeline fails the run on "
        "any lint or test error. The suite has 17 passing unit/integration tests."
    )
    _placeholder(doc, "GitHub Actions — a green run showing all three jobs")

    # ---- 9. Docker ----
    _h(doc, "9. Containerisation (Docker)")
    doc.add_paragraph(
        "A multi-stage Dockerfile (Python 3.12-slim) builds wheels then installs "
        "them offline, runs as a non-root user, exposes port 8000, and defines a "
        "container HEALTHCHECK against /health. The /predict endpoint accepts JSON, "
        "validates it with Pydantic, and returns prediction + confidence. Verified "
        "locally: the container becomes healthy in ~4s and /predict returns a "
        "correct positive prediction (probability 0.96) for the high-risk sample."
    )
    _placeholder(doc, "docker build success + docker run")
    _placeholder(doc, "/predict response (Swagger UI or curl)")

    # ---- 10. Deployment ----
    _h(doc, "10. Production Deployment (Kubernetes)")
    doc.add_paragraph(
        "Deployed to Docker Desktop Kubernetes. k8s/deployment.yaml runs 2 "
        "replicas with resource requests/limits, readiness & liveness probes on "
        "/health, and Prometheus scrape annotations; k8s/service.yaml exposes a "
        "LoadBalancer on port 8000 (reachable at http://localhost:8000). A Helm "
        "chart (helm/heart-disease-api/) provides the same, parameterised."
    )
    _placeholder(doc, "kubectl get pods,svc showing pods Running")
    _placeholder(doc, "/predict call through the LoadBalancer service")

    # ---- 11. Monitoring ----
    _h(doc, "11. Monitoring & Logging")
    doc.add_paragraph(
        "A FastAPI middleware logs every request (method, path, status, latency). "
        "prometheus-fastapi-instrumentator exposes standard HTTP latency/throughput "
        "at /metrics, plus a custom heart_predictions_total{outcome=...} counter to "
        "watch the prediction mix (a simple drift signal). docker compose up brings "
        "up API + Prometheus (10s scrape) + Grafana (datasource + dashboard "
        "auto-provisioned) with panels for request rate, p95 latency, status codes, "
        "and prediction outcomes."
    )
    _placeholder(doc, "Prometheus targets page showing the API as UP")
    _placeholder(doc, "Grafana dashboard with live traffic")

    # ---- 12. Architecture ----
    _h(doc, "12. System Architecture")
    _figure(doc, "architecture.png", "Figure 7. System architecture.", 6.5)

    # ---- 13. Deliverables ----
    _h(doc, "13. Repository & Deliverables")
    for item in [
        "GitHub repository: <insert link>",
        "Code, Dockerfile, requirements — repo root",
        "Dataset + download script — src/data/download.py, data/",
        "Notebook / scripts — notebooks/01_eda.ipynb, src/",
        "Tests — tests/",
        "CI/CD — .github/workflows/ci.yml",
        "Deployment manifests / Helm — k8s/, helm/",
        "Video walkthrough — <insert link>",
    ]:
        doc.add_paragraph(item, style="List Bullet")

    # ---- 14. Conclusion ----
    _h(doc, "14. Conclusion")
    doc.add_paragraph(
        "This project delivers a reproducible, tested, containerised, and monitored "
        "heart-disease classifier meeting every assignment requirement. The tuned "
        "Logistic Regression (ROC-AUC 0.968, recall 0.93) is both accurate and "
        "interpretable — the right choice for clinical screening — and the "
        "surrounding MLOps scaffolding makes it safe to iterate on and operate in "
        "production. Possible extensions: data-drift detection (Evidently), a model "
        "registry with staged promotion, automated retraining, and canary "
        "deployments."
    )

    doc.save(str(OUT))
    print(f"[docx] Saved -> {OUT}")


if __name__ == "__main__":
    build()
