"""Compatibility shim auto-loaded by every Python interpreter on this PYTHONPATH.

Python 3.14 removed the deprecated ``importlib.abc.Traversable`` alias (it now
lives only in ``importlib.resources.abc``). MLflow 3.14's UI server still does
``from importlib.abc import Traversable``, which crashes the server — including
the worker subprocesses uvicorn spawns on Windows. Because ``sitecustomize`` is
imported automatically at interpreter startup, putting this directory on
PYTHONPATH patches the parent process AND every spawned child.

This only affects the local ``mlflow ui`` viewer; it does not change the model,
the training pipeline, or the API.
"""
import importlib.abc

if not hasattr(importlib.abc, "Traversable"):
    try:
        from importlib.resources.abc import Traversable

        importlib.abc.Traversable = Traversable
    except Exception:  # pragma: no cover - best-effort shim
        pass
