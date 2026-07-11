# Data

This project does **not** commit the raw dataset; it is downloaded reproducibly.

## Download

```bash
python -m src.data.download      # -> data/raw/heart_disease_raw.csv
python -m src.data.preprocess    # -> data/processed/heart_disease_clean.csv
```

## Source

**Heart Disease UCI (Cleveland)** — UCI Machine Learning Repository (dataset id 45).
<https://archive.ics.uci.edu/dataset/45/heart+disease>

- `download.py` fetches via the official `ucimlrepo` package, falling back to a
  direct download of `processed.cleveland.data`.
- 303 records, 13 clinical features, and a diagnosis field (`num`, 0–4) that is
  binarised to a 0/1 `target` (0 = no disease, ≥1 = disease).
- Missing values (`?`) appear only in `ca` and `thal`; they are imputed inside
  the modelling pipeline.

## Layout

```
data/
├── raw/         # downloaded, headered CSV (git-ignored — reproducible)
└── processed/   # cleaned, model-ready CSV (committed as a deliverable)
```
