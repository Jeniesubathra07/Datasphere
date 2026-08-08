"""Inference input field definitions shared by API and ML layers."""

from __future__ import annotations

from datasphere.ml.config import EXCLUDED_FEATURE_COLUMNS, TARGET_COLUMN
from datasphere.ml.validation import REQUIRED_COLUMNS

# Fields accepted by POST /predict — excludes target, leakage, identifiers, and free text.
PREDICT_API_FIELDS: tuple[str, ...] = tuple(
    col
    for col in REQUIRED_COLUMNS
    if col not in EXCLUDED_FEATURE_COLUMNS and col != TARGET_COLUMN
)

CATEGORICAL_OPTIONS: dict[str, list[str]] = {
    "academic_year": ["2021-2022", "2022-2023", "2023-2024", "2024-2025"],
    "country": [
        "Bangladesh", "Ethiopia", "India", "Kenya", "Nepal",
        "Nigeria", "Pakistan", "Philippines",
    ],
    "state_province": [
        "Amhara", "Bagmati", "Balochistan", "Barisal", "Bicol", "Bihar", "Borno",
        "Coast", "Jharkhand", "Kaduna", "Kano", "Karnali", "Khulna",
        "Khyber Pakhtunkhwa", "Lagos", "Luzon", "Madhesh", "Mindanao", "Nairobi",
        "North Eastern", "Nyanza", "Odisha", "Oromia", "Punjab", "Rajasthan",
        "Rajshahi", "Rangpur", "SNNPR", "Sindh", "Sokoto", "Sudurpashchim",
        "Sylhet", "Tigray", "Turkana", "Uttar Pradesh", "Visayas", "West Bengal",
    ],
    "location_type": ["Rural", "Semi-Urban", "Urban"],
    "gender": ["Female", "Male", "Other"],
    "school_type": ["Government", "Government-Aided", "NGO/Community-Run", "Private"],
    "mode_of_transport": ["Bicycle", "No Transport", "Public Transport", "School Bus", "Walking"],
    "mother_education_level": [
        "Graduate", "Higher Secondary", "No Formal Education", "Primary", "Secondary",
    ],
    "father_education_level": [
        "Graduate", "Higher Secondary", "No Formal Education", "Primary", "Secondary",
    ],
    "report_source": [
        "Community Volunteer", "Government Survey", "Mobile Data Collection App",
        "NGO Field Worker", "School Record",
    ],
}

NUMERIC_BOUNDS: dict[str, tuple[float, float]] = {
    "age": (5, 20),
    "grade_level": (1, 12),
    "distance_to_school_km": (0, 50),
    "household_income_monthly_usd": (0, 5000),
    "family_size": (1, 15),
    "number_of_siblings": (0, 12),
    "attendance_rate_pct": (0, 100),
    "average_test_score_pct": (0, 100),
    "number_of_school_transfers": (0, 6),
    "disciplinary_incidents_count": (0, 6),
    "teacher_student_ratio": (1, 100),
    "school_infrastructure_score": (0, 100),
}
