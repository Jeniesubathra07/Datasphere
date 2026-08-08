import { describe, expect, it } from 'vitest'
import { createDefaultStudent, validateStudent } from './validation'
import type { ModelSchema } from '../types/student'

const schema: ModelSchema = {
  fields: Object.keys(createDefaultStudent()) as string[],
  categorical_options: {
    gender: ['Female', 'Male', 'Other'],
    country: ['Kenya'],
    academic_year: ['2023-2024'],
    state_province: ['Nairobi'],
    location_type: ['Urban'],
    school_type: ['Government'],
    mode_of_transport: ['Walking'],
    mother_education_level: ['Secondary'],
    father_education_level: ['Primary'],
    report_source: ['School Record'],
  },
  numeric_bounds: {
    age: [5, 20],
    attendance_rate_pct: [0, 100],
    teacher_student_ratio: [1, 100],
  },
  excluded_fields: ['student_id'],
  class_labels: ['Low', 'Medium', 'High', 'Critical'],
}

describe('validation', () => {
  it('creates default student with required fields', () => {
    const student = createDefaultStudent(schema)
    expect(student.age).toBeGreaterThan(0)
    expect(student.gender).toBeTruthy()
  })

  it('flags out-of-range numeric values', () => {
    const student = { ...createDefaultStudent(schema), attendance_rate_pct: 150 }
    const errors = validateStudent(student, schema)
    expect(errors.attendance_rate_pct).toBeTruthy()
  })
})
