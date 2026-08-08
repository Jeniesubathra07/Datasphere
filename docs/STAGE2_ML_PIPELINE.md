# Stage 2 — Production ML Pipeline

## Overview

Reproducible, leakage-safe, validated student dropout risk prediction pipeline for the Child Education Risk Intelligence dataset (200,000 rows × 47 columns).

**Target:** `dropout_risk_level` (Low, Medium, High, Critical)  
**Primary selection metric:** Macro F1 (handles class imbalance; prioritizes minority risk classes)

## Architecture

```
RAW DATA (unchanged on disk)
    ↓ validate_raw_dataset()
VALIDATION
    ↓ clean_dataset() — copy only
CLEANING
    ↓ FeatureEngineer (inside sklearn Pipeline)
FEATURE ENGINEERING
    ↓ split_features_target()
TARGET SEPARATION
    ↓ create_splits() — stratified 64/16/20
TRAIN / VALIDATION / TEST
    ↓ ColumnTransformer (fit on train only)
PREPROCESSING
    ↓ Classifier
MODEL
```

## Excluded columns (leakage / identifiers)

| Column | Reason |
|--------|--------|
| `student_id` | Unique identifier — not predictive |
| `dropout_probability_score` | Target-derived leakage |
| `dropout_risk_level` | Target |
| `case_notes` | Unapproved free text (Stage 1) |

## Engineered features

| Feature | Source |
|---------|--------|
| `record_year`, `record_month` | `record_generated_date` |
| `enrollment_year`, `enrollment_month` | `enrollment_date` |
| `enrollment_duration_days` | date difference |
| `household_income_log` | `log1p(household_income_monthly_usd)` |

Raw date columns are dropped after engineering.

## Split strategy

**Stratified holdout:** 20% test (untouched), then 20% of remainder for validation → ~64% train / 16% val / 20% test.

Stratification preserves class proportions (Low 50%, Critical 7%). No temporal cutoff was mandated in Stage 1.

## Models benchmarked

1. **Baseline:** Stratified DummyClassifier
2. **Logistic Regression** (balanced class weights)
3. **Decision Tree** (max_depth=12)
4. **Random Forest** (100 trees)
5. **HistGradientBoosting** (200 iterations)

### Class imbalance

Compared `class_weight=balanced` vs none. **Selected: balanced** — improves recall on High/Critical without SMOTE (avoids synthetic sample distortion).

## Hyperparameter tuning

`RandomizedSearchCV` (3-fold CV, macro F1) on top 2 validation performers. Tuning uses **training data only** — never the held-out test set.

## Artifacts

| File | Contents |
|------|----------|
| `models/education_risk_pipeline.joblib` | FeatureEngineer + preprocessor + classifier |
| `models/education_risk_pipeline_metadata.json` | Schema, metrics, seed, model version |

## Usage

### Train pipeline

```bash
python scripts/run_stage2_pipeline.py
```

Options:
- `--sample-rows 5000` — faster development run
- `--no-tuning` — skip hyperparameter search
- `--no-class-weight` — disable balanced weights

### Inference (Python)

```python
from datasphere.ml.inference import predict_student_risk

result = predict_student_risk(student_dict)
# result: predicted_risk, class_probabilities, explanation
```

### Run tests

```bash
pytest tests/test_ml_pipeline.py -v
```

## Validation checklist

- [x] Target absent from X
- [x] Leakage columns excluded
- [x] student_id excluded
- [x] Preprocessing inside sklearn Pipeline (fit on train only)
- [x] Test set untouched during tuning
- [x] Baseline + multiple candidates compared
- [x] Pipeline serialized and reload-tested
- [x] Inference schema documented

## Stage 3 requirements (not built yet)

- FastAPI `POST /api/predict` endpoint
- React dashboard
- Model loading at API startup
- Batch scoring endpoint
- Auth / rate limiting

## Limitations

- Adjacent risk levels (Medium ↔ High) are inherently ambiguous
- Minority classes (Critical, High) have fewer training examples
- Explanations reflect statistical association, not causation
- No drift monitoring or automated retraining
