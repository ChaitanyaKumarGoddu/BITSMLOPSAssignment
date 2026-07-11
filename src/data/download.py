"""Task 1 — Data acquisition.

Downloads the UCI Heart Disease (Cleveland) dataset and writes a raw CSV with
proper column headers to ``data/raw/``.

Two acquisition strategies are attempted, in order:
  1. The official ``ucimlrepo`` package (``fetch_ucirepo(id=45)``).
  2. A direct HTTPS download of ``processed.cleveland.data`` from the UCI
     archive (fallback when the package or its API is unavailable).

Run from the repo root:
    python -m src.data.download
"""

from __future__ import annotations

import io
import sys
import urllib.request

import pandas as pd

from src.config import RAW_COLUMNS, RAW_CSV, ensure_dirs

UCI_DIRECT_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)


def _from_ucimlrepo() -> pd.DataFrame | None:
    """Fetch via the official UCI Python package. Returns None on any failure."""
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        print("[download] ucimlrepo not installed, skipping to direct download.")
        return None

    try:
        print("[download] Fetching dataset id=45 via ucimlrepo ...")
        ds = fetch_ucirepo(id=45)
        # Features + target arrive as separate frames; concatenate to one table.
        df = pd.concat([ds.data.features, ds.data.targets], axis=1)
        df.columns = RAW_COLUMNS
        return df
    except Exception as exc:  # network / API problems -> fall back
        print(f"[download] ucimlrepo failed ({exc!r}); falling back to direct URL.")
        return None


def _from_direct_url() -> pd.DataFrame:
    """Download the raw Cleveland file directly from the UCI archive."""
    print(f"[download] Downloading from {UCI_DIRECT_URL}")
    with urllib.request.urlopen(UCI_DIRECT_URL, timeout=60) as resp:
        text = resp.read().decode("utf-8")
    # The file is headerless, comma separated, and uses '?' for missing values.
    df = pd.read_csv(io.StringIO(text), header=None, names=RAW_COLUMNS)
    return df


def download() -> pd.DataFrame:
    """Acquire the dataset and persist it to ``RAW_CSV``. Returns the frame."""
    ensure_dirs()
    df = _from_ucimlrepo()
    if df is None:
        df = _from_direct_url()

    # Normalise the UCI '?' missing marker to a real NaN so pandas sees it.
    df = df.replace("?", pd.NA)
    df.to_csv(RAW_CSV, index=False)
    print(f"[download] Wrote {len(df)} rows x {df.shape[1]} cols -> {RAW_CSV}")
    return df


if __name__ == "__main__":
    try:
        download()
    except Exception as exc:  # pragma: no cover - surfaced to CI logs
        print(f"[download] FATAL: {exc!r}", file=sys.stderr)
        sys.exit(1)
