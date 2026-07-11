# 🚀 RUNBOOK — Step-by-Step Guide to Complete & Submit

This is your operator's manual. Follow it top to bottom. Everything runs from **one path**:

```
c:\workspace\Assignment\heart-disease-mlops
```

> **Golden rule:** open your terminal (PowerShell) **in that folder** before running anything.
> In VS Code: right-click the `heart-disease-mlops` folder → *Open in Integrated Terminal*.

Legend: 🟢 = already done for you and verified · 🔵 = you run this · 📷 = capture a screenshot · 💾 = gets committed to git.

---

## 0. Prerequisites (one-time)

| Tool | Needed for | Check |
|------|-----------|-------|
| Python 3.11–3.14 | training, tests, API | `python --version` |
| Git | version control | `git --version` |
| Docker Desktop | container + K8s + monitoring | `docker --version` |
| Docker Desktop **Kubernetes** | Task 7 | Settings → Kubernetes → *Enable* → Apply |
| A GitHub account | pushing the repo | — |

The Python virtual environment is **already created** at `.venv\` with every dependency installed. If you ever need to recreate it:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements-dev.txt
```

**Activate it at the start of every session:**

```powershell
.venv\Scripts\Activate.ps1
```

Once activated, your prompt shows `(.venv)` and you can type `python` (instead of `.venv\Scripts\python.exe`). All commands below assume the venv is active.

---

## 1. What is already done (🟢 no action needed)

I ran and verified all of this — you do **not** need to re-run it, but you *can* to reproduce:

- ✅ Dataset downloaded & cleaned → `data/processed/heart_disease_clean.csv`
- ✅ EDA figures generated → `reports/figures/eda_*.png`
- ✅ Models trained (LR/RF/XGB) → **Logistic Regression won**: ROC-AUC 0.968, accuracy 0.90, recall 0.93 → `models/model.pkl`
- ✅ MLflow runs logged → `mlflow.db` + `mlartifacts/`
- ✅ 17 unit tests pass, lint/format clean
- ✅ Docker image built & container tested (`/predict` returns p=0.96)
- ✅ Git repo initialised with 2 commits
- ✅ Report draft generated → `reports/Heart_Disease_MLOps_Report_DRAFT.docx`

---

## 2. Reproduce the ML pipeline (🔵 optional — proves it runs on your machine)

Run these **in order**, from the project root, venv active:

```powershell
python -m src.data.download      # 303 rows -> data/raw/
python -m src.data.preprocess    # -> data/processed/
python -m src.data.eda           # -> reports/figures/eda_*.png
python -m src.models.train       # -> models/model.pkl + MLflow runs
```

Expected final line from training:
`[train] Winner: logistic_regression`

---

## 3. Experiment tracking — MLflow UI  📷

> **Use this launcher, not the raw `mlflow ui` command.** MLflow 3.14's web UI has a
> bug on Python 3.14 (it imports `importlib.abc.Traversable`, which 3.14 removed),
> so the plain command crashes with `ImportError: cannot import name 'Traversable'`.
> `scripts/mlflow_ui.py` applies a compatibility shim and then starts the UI normally.
> (This only affects the *viewer* — your logged runs, model, and API are unaffected.)

```powershell
python scripts/mlflow_ui.py --port 5000
```

Open <http://localhost:5000> → click the **`heart-disease-classification`** experiment.

📷 **Screenshot 1** — the **runs table** showing the 3 nested runs (logistic_regression, random_forest, xgboost) with their metrics side by side.
📷 **Screenshot 2** — click one run → the **Metrics** panel and the **Artifacts** panel (showing the logged ROC curve + confusion matrix).

Press `Ctrl+C` in the terminal to stop the UI when done.

---

## 4. Run the tests (🔵 shows the suite is green)  📷 optional

```powershell
pytest
```

Expected: `17 passed`. (📷 optional — a screenshot of the green test output is a nice extra.)

---

## 5. Run the API locally (🔵)  📷

```powershell
uvicorn src.api.main:app --reload --port 8000
```

Open <http://localhost:8000/docs> (Swagger UI).

📷 **Screenshot 3** — the **Swagger `/docs`** page.

Test a prediction — in a **second** terminal (venv active, same folder):

```powershell
curl.exe -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "@sample_request.json"
```

Expected: `{"prediction":1,"label":"Heart disease","confidence":0.9635,...}`

📷 **Screenshot 4** — the `/predict` request + response (either the curl output, or use *Try it out* in Swagger).

Stop the server with `Ctrl+C`.

> **Note (Windows):** use `curl.exe` (not `curl`) in PowerShell so it doesn't alias to `Invoke-WebRequest`. Or just use Swagger's *Try it out* button.

---

## 6. Docker — build & run the container (🔵)  📷

```powershell
docker build -t heart-disease-api:latest .
docker run --rm -p 8000:8000 heart-disease-api:latest
```

📷 **Screenshot 5** — the successful `docker build` (last lines showing the image was created) **and** the `docker run` startup logs.

In a second terminal, hit the container:

```powershell
curl.exe -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "@sample_request.json"
```

📷 **Screenshot 6** — the `/predict` response coming from the **container**.

Stop the container with `Ctrl+C`.

---

## 7. Kubernetes deployment — Docker Desktop (🔵)  📷

Make sure Kubernetes is enabled in Docker Desktop (green "Kubernetes running" indicator). The image from step 6 already lives in the local Docker daemon.

```powershell
kubectl apply -f k8s/
kubectl get pods,svc
```

Wait until both pods show `Running` (re-run `kubectl get pods` a few times).

📷 **Screenshot 7** — `kubectl get pods,svc` with **2 pods Running** and the `heart-disease-api-service` LoadBalancer.

Test through the service (Docker Desktop publishes LoadBalancer on localhost):

```powershell
curl.exe -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "@sample_request.json"
```

📷 **Screenshot 8** — `/predict` response through the Kubernetes service.

Tear down when finished: `kubectl delete -f k8s/`

> **Helm alternative** (optional, do *instead of* `kubectl apply`): `helm install heart ./helm/heart-disease-api` → `kubectl get pods,svc` → `helm uninstall heart`.

---

## 8. Monitoring — Prometheus + Grafana (🔵)  📷

> The compose stack maps the API to **host port 8001** (container stays 8000), so it
> runs happily **alongside** the Kubernetes deployment on 8000 — no need to tear
> anything down. Prometheus (host 9091) and Grafana (3000) don't conflict either.

```powershell
docker compose up --build
```

Wait ~30s, then generate some traffic (second terminal) so the graphs aren't empty
— note the port is **8001** here:

```powershell
1..20 | ForEach-Object { curl.exe -s -X POST http://localhost:8001/predict -H "Content-Type: application/json" -d "@sample_request.json" | Out-Null }
```

- **Prometheus** → <http://localhost:9091> → *Status → Targets*.
  📷 **Screenshot 9** — the `heart-disease-api` target showing **UP**.
- **Grafana** → <http://localhost:3000> (login `admin` / `admin`, skip password change) → *Dashboards* → **"Heart Disease API — Monitoring"**.
  📷 **Screenshot 10** — the dashboard with live request-rate / latency / prediction-outcome panels.

Stop with `Ctrl+C`, then `docker compose down`.

---

## 9. Push to GitHub (🔵)  📷

Create a **new empty repo** on GitHub (no README/license), then from the project root:

```powershell
git remote add origin https://github.com/<your-username>/heart-disease-mlops.git
git branch -M main
git push -u origin main
```

📷 **Screenshot 11** — the GitHub repo page showing your files.

### Watch CI/CD run  📷

Go to your repo → **Actions** tab. The `CI/CD Pipeline` workflow runs automatically on the push.

📷 **Screenshot 12** — a **green** workflow run showing the 3 jobs (Lint & Test → Train Model → Docker Build Validation).

> If Actions is disabled, enable it under repo *Settings → Actions → General*.

---

## 10. Finish the report (🔵)

1. Open `reports/Heart_Disease_MLOps_Report_DRAFT.docx` in Word.
2. Fill in the **title page**: your name, BITS ID, GitHub link, video link.
3. Replace each **📷 [SCREENSHOT — …]** placeholder with the matching screenshot from the table below.
4. The EDA plots, ROC curve, confusion matrix, and architecture diagram are **already embedded** — nothing to do there.
5. Save as PDF or DOCX for submission (the assignment allows either; ~10 pages).

> To regenerate the draft after retraining: `python scripts/build_report_docx.py`

---

## 11. Record the video (🔵)

A 3–5 min screen recording (Windows: `Win+G` Game Bar, or OBS/Zoom) walking through:
data → EDA → MLflow runs → tests → API `/predict` → Docker → `kubectl get pods` → Grafana.
Upload (Google Drive/YouTube-unlisted) and paste the link into the report + README.

---

## 📷 Screenshot → Report placeholder map

| # | Screenshot | Report section |
|---|-----------|----------------|
| 1 | MLflow runs table | §6 Experiment Tracking |
| 2 | MLflow run detail (metrics + artifacts) | §6 Experiment Tracking |
| 3 | Swagger `/docs` page | §9 Containerisation |
| 4 | Local `/predict` response | §9 Containerisation |
| 5 | `docker build` + `docker run` | §9 Containerisation |
| 6 | `/predict` from container | §9 Containerisation |
| 7 | `kubectl get pods,svc` (Running) | §10 Deployment |
| 8 | `/predict` via K8s service | §10 Deployment |
| 9 | Prometheus targets UP | §11 Monitoring |
| 10 | Grafana dashboard | §11 Monitoring |
| 11 | GitHub repo page | §13 Deliverables |
| 12 | GitHub Actions green run | §8 CI/CD |

(Save them in `reports/screenshots/` — see that folder's README for filenames.)

---

## 💾 What gets committed to git (already handled)

Already committed (✅):

```
src/, tests/, notebooks/, .github/, k8s/, helm/, monitoring/, scripts/
Dockerfile, docker-compose.yml, Makefile, requirements*.txt, pyproject.toml, .flake8
README.md, RUNBOOK.md, reports/REPORT.md, reports/figures/*.png
data/processed/heart_disease_clean.csv     (cleaned dataset — a deliverable)
models/model.pkl, models/model_metadata.json
```

**Intentionally NOT committed** (`.gitignore`): `.venv/`, `data/raw/`, `mlflow.db`, `mlartifacts/`, `__pycache__/`, and the generated `*.docx` draft (it's regenerable).

### If you want your finished report tracked too

After adding screenshots and exporting, drop the final file in the repo and commit it:

```powershell
git add reports/Heart_Disease_MLOps_Report.pdf     # or .docx
git commit -m "Add final report with screenshots"
git push
```

### If you change any code afterwards

```powershell
git add -A
git commit -m "Describe your change"
git push
```

---

## ⚡ Quick command reference (from project root, venv active)

| Goal | Command |
|------|---------|
| Activate venv | `.venv\Scripts\Activate.ps1` |
| Full pipeline | `python -m src.data.download; python -m src.data.preprocess; python -m src.data.eda; python -m src.models.train` |
| MLflow UI | `python scripts/mlflow_ui.py --port 5000` |
| Tests | `pytest` |
| Lint / format | `flake8 src tests` · `black src tests` · `isort src tests` |
| Serve API | `uvicorn src.api.main:app --reload --port 8000` |
| Docker build/run | `docker build -t heart-disease-api:latest .` · `docker run --rm -p 8000:8000 heart-disease-api:latest` |
| K8s | `kubectl apply -f k8s/` · `kubectl get pods,svc` · `kubectl delete -f k8s/` |
| Monitoring | `docker compose up --build` · `docker compose down` |
| Rebuild report draft | `python scripts/build_report_docx.py` |

---

## 🔧 Troubleshooting

| Symptom | Fix |
|---------|-----|
| `python` not found / wrong version | Activate the venv: `.venv\Scripts\Activate.ps1` |
| `curl` behaves oddly in PowerShell | Use `curl.exe`, or Swagger's *Try it out* |
| Port 8000 already in use | Stop the other API/container, or use `--port 8001` |
| K8s pods stuck `ErrImagePull` | Build the image locally first; `imagePullPolicy` is `IfNotPresent` — Docker Desktop uses the local image |
| K8s pods `Pending` forever | Kubernetes not enabled/running in Docker Desktop |
| Grafana panels empty | Generate traffic (step 8 loop); wait ~15s for scrape |
| `/predict` returns 503 | `models/model.pkl` missing — run `python -m src.models.train` |
| MLflow "file store maintenance mode" | Always pass `--backend-store-uri sqlite:///mlflow.db` |
| MLflow UI `ImportError: ... 'Traversable'` | Use `python scripts/mlflow_ui.py` (not raw `mlflow ui`) — see §3 |
