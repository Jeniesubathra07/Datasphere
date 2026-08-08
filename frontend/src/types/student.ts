export interface StudentInput {
  record_generated_date: string
  enrollment_date: string
  academic_year: string
  country: string
  state_province: string
  location_type: string
  age: number
  gender: string
  grade_level: number
  school_type: string
  distance_to_school_km: number
  mode_of_transport: string
  household_income_monthly_usd: number
  mother_education_level: string
  father_education_level: string
  family_size: number
  number_of_siblings: number
  is_orphan: boolean
  single_parent_household: boolean
  household_has_electricity: boolean
  household_has_internet_access: boolean
  owns_smartphone_or_computer: boolean
  child_labor_involvement: boolean
  attendance_rate_pct: number
  average_test_score_pct: number
  previous_year_pass: boolean
  number_of_school_transfers: number
  disciplinary_incidents_count: number
  bullying_incidents_reported: boolean
  health_issues_reported: boolean
  special_needs_status: boolean
  free_meal_program_enrolled: boolean
  scholarship_received: boolean
  extracurricular_participation: boolean
  teacher_student_ratio: number
  school_infrastructure_score: number
  early_marriage_risk_flag: boolean
  community_conflict_zone: boolean
  seasonal_migration_family: boolean
  ngo_intervention_present: boolean
  literacy_program_enrolled: boolean
  language_barrier_flag: boolean
  report_source: string
}

export interface ModelSchema {
  fields: string[]
  categorical_options: Record<string, string[]>
  numeric_bounds: Record<string, [number, number]>
  excluded_fields: string[]
  class_labels: string[]
}

export interface PredictionResponse {
  prediction: {
    risk_level: string
    probabilities: Record<string, number>
  }
  explanation?: {
    important_factors: { feature: string; importance: number }[]
    summary: string
  }
  model: {
    version: string
    selected_model: string
    primary_metric?: string
  }
}
