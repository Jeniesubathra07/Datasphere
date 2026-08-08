# Datasphere — Child Education Risk Intelligence Platform

AI-assisted platform to predict, explain, and prioritize student dropout risk.

## Problem statement

Millions of students face dropout risk due to socioeconomic, academic, and environmental factors. Educators need data-driven tools to identify at-risk students early and allocate interventions effectively.

## Solution

A full-stack application that:
1. Validates and cleans student records
2. Engineers features from approved Stage 1 decisions
3. Predicts dropout risk (Low / Medium / High / Critical) using a trained ML pipeline
4. Returns class probabilities and model-based explanations
5. Presents results through a responsive React dashboard

## Architecture

```
React + Vite + Material UI  →  REST API  →  FastAPI  →  Serialized ML Pipeline
```

The API loads `models/education_risk_pipeline.joblib` **once at startup**. All predictions use the real Stage 2 artifact — no hardcoded responses.

## Dataset

- **Source:** `datasets/raw/child_education_risk_intelligence.csv`
- **Scale:** 200,000 students, 47 raw columns
- **Target:** `dropout_risk_level` (Low, Medium, High, Critical)

## ML pipeline (Stage 2)

| Item | Value |
|------|-------|
| Selected model | Logistic Regression (balanced class weights) |
| Test macro F1 | 0.646 |
| Test accuracy | 0.705 |
| Primary metric | Macro F1 |
| Split | Stratified 64/16/20 (seed=42) |

Excluded from model features: `student_id`, `dropout_probability_score`, `dropout_risk_level`, `case_notes`

See `docs/STAGE2_ML_PIPELINE.md` for full ML documentation.

## Installation

### Prerequisites

- Python 3.12+
- Node.js 18+
- Git

### Backend

```bash
git clone https://github.com/Jeniesubathra07/Datasphere.git
cd Datasphere
git checkout cursor/stage3-app-b7ba   # or main after merge

python -m pip install -e ".[dev]"
python -c "from datasphere.data.archives import extract_dataset_archives; extract_dataset_archives()"
python scripts/run_stage2_pipeline.py   # trains model if artifact missing
```

### Frontend

```bash
cd frontend
npm install
cp .env.example .env
```

## Running locally

### Terminal 1 — Backend (port 8080 recommended on Windows)

```bash
python scripts/run_stage2_pipeline.py   # creates models/education_risk_pipeline.joblib
python -m uvicorn datasphere.app:app --host 127.0.0.1 --port 8080 --reload
```

If port 8080 is busy, use another port (e.g. 8000) and set the same value in `frontend/.env`.

### Terminal 2 — Frontend (port 5173)

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

Open http://localhost:5173

The Vite dev server proxies API calls to `http://127.0.0.1:8080` by default. Set `VITE_API_BASE_URL` in `frontend/.env` if your backend uses a different port.

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check + model load status |
| POST | `/predict` | Risk prediction with probabilities |
| POST | `/predict/explain` | Prediction + feature importance explanation |
| GET | `/api/model-info` | Model version, metrics, feature schema |
| GET | `/api/model/schema` | Inference schema for frontend contract |
| GET | `/api/datasets` | List available datasets |
| POST | `/api/model/train` | Retrain pipeline (admin) |

### Prediction request (excerpt)

```json
{
  "record_generated_date": "2024-03-10",
  "enrollment_date": "2022-09-01",
  "academic_year": "2023-2024",
  "country": "Kenya",
  "age": 14,
  "gender": "Female",
  "attendance_rate_pct": 85,
  "average_test_score_pct": 65,
  "...": "43 approved fields total"
}
```

**Not accepted:** `dropout_risk_level`, `dropout_probability_score`, `student_id`, `case_notes`

### Prediction response

```json
{
  "prediction": {
    "risk_level": "Medium",
    "probabilities": { "Critical": 0.01, "High": 0.12, "Low": 0.45, "Medium": 0.42 }
  },
  "explanation": {
    "important_factors": [{ "feature": "...", "importance": 0.15 }],
    "summary": "The model predicts 'Medium' risk..."
  },
  "model": { "version": "2.0.0", "selected_model": "logistic_regression" }
}
```

## Testing

```bash
# Backend (31 tests)
python -m pytest tests/ -v

# Frontend (2 tests)
cd frontend && npm test
```

## Folder structure

```
datasphere/
  app.py                 # FastAPI app with model lifespan
  api/                   # Routes, schemas, services
  core/                  # Config, model loader
  ml/                    # Stage 2 ML pipeline
  data/                  # Dataset loading
frontend/
  src/                   # React + MUI UI
models/                  # Serialized pipeline artifacts
scripts/                 # run_stage2_pipeline.py
tests/                   # Backend tests
docs/                    # Stage 2 ML docs
```

## Limitations

- Predictions reflect statistical patterns, not certain outcomes
- Adjacent risk levels (Medium ↔ High) are inherently ambiguous
- Minority classes (Critical, High) have lower recall
- No authentication or rate limiting in this version
- Explanations are associative, not causal

## Future improvements

- Authentication and role-based access
- Batch prediction endpoint
- Model drift monitoring and retraining scheduler
- SHAP values in API responses
- Deployment (Docker, CI/CD)

## License

See LICENSE file.
