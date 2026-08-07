# Datasphere — Child Education Risk Intelligence Platform

AI-powered platform to predict, explain, and prioritize student dropout risk.

Based on the **Tech Wizards** project workflow (Track 5: AI for Social Good).

## Project workflow

1. Data understanding and ingestion
2. Data validation and cleaning (Pandera)
3. Exploratory data analysis
4. Feature engineering
5. XGBoost model development (Low / Medium / High / Critical)
6. SHAP explainability
7. Intervention prioritization engine
8. BI layer
9. FastAPI + React dashboard
10. Prototype delivery

## Setup

```bash
bash .cursor/scripts/install.sh
```

## Datasets

Add your CSV files to `datasets/raw/` (for example `student_education_risk.csv`).

Expected scale from project spec: ~200K students, ~47 features, 4 risk levels.

## Notebooks

Primary worked notebook (EDA, cleaning, feature engineering, XGBoost, SHAP):

```bash
jupyter notebook notebooks/datasphere.ipynb
```

Scaffold pipeline notebook:

```bash
jupyter notebook notebooks/education_risk_pipeline.ipynb
```

## API

```bash
python3 -m uvicorn datasphere.app:app --host 0.0.0.0 --port 8000 --reload
```

Endpoints:

- `GET /health`
- `GET /api/datasets` — list CSV datasets found in `datasets/`
- `POST /api/model/train` — train XGBoost risk model from primary dataset

## Tests

```bash
python3 -m pytest
```
