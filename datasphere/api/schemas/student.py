"""Pydantic schemas for student prediction requests."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from datasphere.api.schema_fields import CATEGORICAL_OPTIONS, NUMERIC_BOUNDS, PREDICT_API_FIELDS


class StudentPredictionRequest(BaseModel):
    """Approved inference schema — no target, leakage, or identifier fields."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    record_generated_date: date
    enrollment_date: date
    academic_year: str
    country: str
    state_province: str
    location_type: str
    age: int = Field(ge=5, le=20)
    gender: str
    grade_level: int = Field(ge=1, le=12)
    school_type: str
    distance_to_school_km: float = Field(ge=0, le=50)
    mode_of_transport: str
    household_income_monthly_usd: float = Field(ge=0, le=5000)
    mother_education_level: str
    father_education_level: str
    family_size: float = Field(ge=1, le=15)
    number_of_siblings: int = Field(ge=0, le=12)
    is_orphan: bool
    single_parent_household: bool
    household_has_electricity: bool
    household_has_internet_access: bool
    owns_smartphone_or_computer: bool
    child_labor_involvement: bool
    attendance_rate_pct: float = Field(ge=0, le=100)
    average_test_score_pct: float = Field(ge=0, le=100)
    previous_year_pass: bool
    number_of_school_transfers: int = Field(ge=0, le=6)
    disciplinary_incidents_count: int = Field(ge=0, le=6)
    bullying_incidents_reported: bool
    health_issues_reported: bool
    special_needs_status: bool
    free_meal_program_enrolled: bool
    scholarship_received: bool
    extracurricular_participation: bool
    teacher_student_ratio: float = Field(ge=1, le=100)
    school_infrastructure_score: float = Field(ge=0, le=100)
    early_marriage_risk_flag: bool
    community_conflict_zone: bool
    seasonal_migration_family: bool
    ngo_intervention_present: bool
    literacy_program_enrolled: bool
    language_barrier_flag: bool
    report_source: str

    @field_validator(
        "academic_year", "country", "state_province", "location_type", "gender",
        "school_type", "mode_of_transport", "mother_education_level",
        "father_education_level", "report_source",
    )
    @classmethod
    def validate_categorical(cls, value: str, info) -> str:
        allowed = CATEGORICAL_OPTIONS.get(info.field_name)
        if allowed and value not in allowed:
            raise ValueError(f"{info.field_name} must be one of: {allowed}")
        return value

    def to_inference_dict(self) -> dict:
        data = self.model_dump()
        data["record_generated_date"] = self.record_generated_date.isoformat()
        data["enrollment_date"] = self.enrollment_date.isoformat()
        return data

    @classmethod
    def field_names(cls) -> list[str]:
        return list(PREDICT_API_FIELDS)


class ModelSchemaResponse(BaseModel):
    fields: list[str]
    categorical_options: dict[str, list[str]]
    numeric_bounds: dict[str, list[float]]
    excluded_fields: list[str]
    class_labels: list[str]
