"""Launch the MLflow UI on Python 3.14 (works around an MLflow/py3.14 bug).

MLflow 3.14's UI server imports a name that Python 3.14 removed
(``importlib.abc.Traversable``). This launcher puts a ``sitecustomize`` shim on
PYTHONPATH so the patch loads in the server and every spawned worker, then
starts the UI exactly as ``mlflow ui`` would.

Run from the repo root:
    python scripts/mlflow_ui.py
    python scripts/mlflow_ui.py --port 5001      # any extra args pass through

Then open http://localhost:5000 (or your chosen port).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHIM_DIR = Path(__file__).resolve().parent / "_mlflow_shim"


def main() -> int:
    args = sys.argv[1:]
    # Sensible defaults; a user-supplied --port / --backend-store-uri overrides.
    if not any(a.startswith("--backend-store-uri") for a in args):
        args += ["--backend-store-uri", "sqlite:///mlflow.db"]
    if not any(a.startswith("--port") for a in args):
        args += ["--port", "5000"]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(SHIM_DIR) + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [sys.executable, "-m", "mlflow", "ui", *args]
    print(f"[mlflow-ui] {' '.join(cmd)}")
    print(f"[mlflow-ui] PYTHONPATH shim: {SHIM_DIR}")
    return subprocess.call(cmd, cwd=str(ROOT), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
