import type { ModelSchema, StudentInput } from '../types/student'

export function createDefaultStudent(schema?: ModelSchema): StudentInput {
  const opts = schema?.categorical_options ?? {}
  const pick = (key: string, fallback: string) => opts[key]?.[0] ?? fallback
  const today = new Date().toISOString().slice(0, 10)
  return {
    record_generated_date: today,
    enrollment_date: '2023-09-01',
    academic_year: pick('academic_year', '2023-2024'),
    country: pick('country', 'Kenya'),
    state_province: pick('state_province', 'Nairobi'),
    location_type: pick('location_type', 'Urban'),
    age: 14,
    gender: pick('gender', 'Female'),
    grade_level: 8,
    school_type: pick('school_type', 'Government'),
    distance_to_school_km: 2.5,
    mode_of_transport: pick('mode_of_transport', 'Walking'),
    household_income_monthly_usd: 250,
    mother_education_level: pick('mother_education_level', 'Secondary'),
    father_education_level: pick('father_education_level', 'Primary'),
    family_size: 5,
    number_of_siblings: 2,
    is_orphan: false,
    single_parent_household: false,
    household_has_electricity: true,
    household_has_internet_access: false,
    owns_smartphone_or_computer: false,
    child_labor_involvement: false,
    attendance_rate_pct: 85,
    average_test_score_pct: 65,
    previous_year_pass: true,
    number_of_school_transfers: 0,
    disciplinary_incidents_count: 0,
    bullying_incidents_reported: false,
    health_issues_reported: false,
    special_needs_status: false,
    free_meal_program_enrolled: true,
    scholarship_received: false,
    extracurricular_participation: true,
    teacher_student_ratio: 35,
    school_infrastructure_score: 60,
    early_marriage_risk_flag: false,
    community_conflict_zone: false,
    seasonal_migration_family: false,
    ngo_intervention_present: false,
    literacy_program_enrolled: false,
    language_barrier_flag: false,
    report_source: pick('report_source', 'School Record'),
  }
}

export function validateStudent(
  student: StudentInput,
  schema: ModelSchema,
): Partial<Record<keyof StudentInput, string>> {
  const errors: Partial<Record<keyof StudentInput, string>> = {}
  for (const [key, [min, max]] of Object.entries(schema.numeric_bounds)) {
    const field = key as keyof StudentInput
    const value = student[field]
    if (typeof value === 'number' && (value < min || value > max)) {
      errors[field] = `Must be between ${min} and ${max}`
    }
  }
  for (const [key, allowed] of Object.entries(schema.categorical_options)) {
    const field = key as keyof StudentInput
    const value = student[field]
    if (typeof value === 'string' && !allowed.includes(value)) {
      errors[field] = 'Invalid selection'
    }
  }
  return errors
}
