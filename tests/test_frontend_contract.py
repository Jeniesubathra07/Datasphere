"""Verify frontend TypeScript field names match backend API schema."""

from __future__ import annotations

import re
from pathlib import Path

from datasphere.api.schemas.student import StudentPredictionRequest


def _extract_frontend_fields() -> set[str]:
    types_file = Path(__file__).resolve().parents[1] / "frontend" / "src" / "types" / "student.ts"
    content = types_file.read_text(encoding="utf-8")
    block = content.split("export interface StudentInput {", 1)[1].split("}", 1)[0]
    return set(re.findall(r"^\s+(\w+):", block, re.MULTILINE))


def test_frontend_types_match_backend_schema() -> None:
    api_fields = set(StudentPredictionRequest.model_fields.keys())
    frontend_fields = _extract_frontend_fields()
    assert frontend_fields == api_fields, (
        f"Frontend/backend mismatch: "
        f"frontend-only={frontend_fields - api_fields}, api-only={api_fields - frontend_fields}"
    )
