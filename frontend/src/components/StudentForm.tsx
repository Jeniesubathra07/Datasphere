import {
  Box,
  Button,
  FormControl,
  FormControlLabel,
  FormHelperText,
  Grid,
  InputLabel,
  MenuItem,
  Paper,
  Select,
  Switch,
  TextField,
  Typography,
} from '@mui/material'
import type { ModelSchema, StudentInput } from '../types/student'

interface Props {
  schema: ModelSchema
  value: StudentInput
  errors: Partial<Record<keyof StudentInput, string>>
  onChange: (next: StudentInput) => void
  onSubmit: () => void
  onReset: () => void
  loading: boolean
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <Paper sx={{ p: 3, mb: 3 }}>
      <Typography variant="h6" gutterBottom>{title}</Typography>
      <Grid container spacing={2}>{children}</Grid>
    </Paper>
  )
}

function Field({
  label,
  error,
  children,
}: {
  label: string
  error?: string
  children: React.ReactNode
}) {
  return (
    <Grid size={{ xs: 12, sm: 6, md: 4 }}>
      {children}
      {error ? <FormHelperText error>{error}</FormHelperText> : null}
      {!error ? <FormHelperText>{label}</FormHelperText> : null}
    </Grid>
  )
}

export default function StudentForm({ schema, value, errors, onChange, onSubmit, onReset, loading }: Props) {
  const opts = schema.categorical_options
  const set = <K extends keyof StudentInput>(key: K, val: StudentInput[K]) =>
    onChange({ ...value, [key]: val })

  const num = (key: keyof StudentInput) => (
    <TextField
      fullWidth
      type="number"
      label={key.replace(/_/g, ' ')}
      value={value[key] as number}
      onChange={(e) => set(key, Number(e.target.value) as StudentInput[typeof key])}
      error={Boolean(errors[key])}
      slotProps={{
        htmlInput: {
          min: schema.numeric_bounds[key]?.[0],
          max: schema.numeric_bounds[key]?.[1],
          step: key.includes('pct') || key.includes('score') ? 0.1 : 1,
        },
      }}
    />
  )

  const select = (key: keyof StudentInput) => (
    <FormControl fullWidth error={Boolean(errors[key])}>
      <InputLabel>{key.replace(/_/g, ' ')}</InputLabel>
      <Select
        label={key.replace(/_/g, ' ')}
        value={value[key] as string}
        onChange={(e) => set(key, e.target.value as StudentInput[typeof key])}
      >
        {(opts[key] || []).map((o) => (
          <MenuItem key={o} value={o}>{o}</MenuItem>
        ))}
      </Select>
    </FormControl>
  )

  const flag = (key: keyof StudentInput) => (
    <FormControlLabel
      control={
        <Switch
          checked={Boolean(value[key])}
          onChange={(e) => set(key, e.target.checked as StudentInput[typeof key])}
        />
      }
      label={key.replace(/_/g, ' ')}
    />
  )

  return (
    <Box component="form" onSubmit={(e) => { e.preventDefault(); onSubmit() }}>
      <Section title="1. Student Information">
        <Field label="Record date" error={errors.record_generated_date}>
          <TextField fullWidth type="date" value={value.record_generated_date}
            onChange={(e) => set('record_generated_date', e.target.value)}
            slotProps={{ inputLabel: { shrink: true } }} />
        </Field>
        <Field label="Enrollment date" error={errors.enrollment_date}>
          <TextField fullWidth type="date" value={value.enrollment_date}
            onChange={(e) => set('enrollment_date', e.target.value)}
            slotProps={{ inputLabel: { shrink: true } }} />
        </Field>
        <Field label="Academic year" error={errors.academic_year}>{select('academic_year')}</Field>
        <Field label="Country" error={errors.country}>{select('country')}</Field>
        <Field label="State/Province" error={errors.state_province}>{select('state_province')}</Field>
        <Field label="Location type" error={errors.location_type}>{select('location_type')}</Field>
        <Field label="Age" error={errors.age}>{num('age')}</Field>
        <Field label="Gender" error={errors.gender}>{select('gender')}</Field>
        <Field label="Grade level" error={errors.grade_level}>{num('grade_level')}</Field>
      </Section>

      <Section title="2. Academic Information">
        <Field label="School type" error={errors.school_type}>{select('school_type')}</Field>
        <Field label="Distance to school (km)" error={errors.distance_to_school_km}>{num('distance_to_school_km')}</Field>
        <Field label="Mode of transport" error={errors.mode_of_transport}>{select('mode_of_transport')}</Field>
        <Field label="School transfers" error={errors.number_of_school_transfers}>{num('number_of_school_transfers')}</Field>
        <Field label="Disciplinary incidents" error={errors.disciplinary_incidents_count}>{num('disciplinary_incidents_count')}</Field>
        <Field label="Report source" error={errors.report_source}>{select('report_source')}</Field>
        <Grid size={{ xs: 12, sm: 6 }}>{flag('previous_year_pass')}</Grid>
        <Grid size={{ xs: 12, sm: 6 }}>{flag('extracurricular_participation')}</Grid>
      </Section>

      <Section title="3. Attendance & Performance">
        <Field label="Attendance %" error={errors.attendance_rate_pct}>{num('attendance_rate_pct')}</Field>
        <Field label="Average test score %" error={errors.average_test_score_pct}>{num('average_test_score_pct')}</Field>
      </Section>

      <Section title="4. Family & Socioeconomic Factors">
        <Field label="Household income (USD/month)" error={errors.household_income_monthly_usd}>{num('household_income_monthly_usd')}</Field>
        <Field label="Mother education" error={errors.mother_education_level}>{select('mother_education_level')}</Field>
        <Field label="Father education" error={errors.father_education_level}>{select('father_education_level')}</Field>
        <Field label="Family size" error={errors.family_size}>{num('family_size')}</Field>
        <Field label="Number of siblings" error={errors.number_of_siblings}>{num('number_of_siblings')}</Field>
        <Grid size={{ xs: 12, sm: 6 }}>{flag('is_orphan')}</Grid>
        <Grid size={{ xs: 12, sm: 6 }}>{flag('single_parent_household')}</Grid>
        <Grid size={{ xs: 12, sm: 6 }}>{flag('household_has_electricity')}</Grid>
        <Grid size={{ xs: 12, sm: 6 }}>{flag('household_has_internet_access')}</Grid>
        <Grid size={{ xs: 12, sm: 6 }}>{flag('owns_smartphone_or_computer')}</Grid>
        <Grid size={{ xs: 12, sm: 6 }}>{flag('child_labor_involvement')}</Grid>
        <Grid size={{ xs: 12, sm: 6 }}>{flag('seasonal_migration_family')}</Grid>
      </Section>

      <Section title="5. School Environment">
        <Field label="Teacher-student ratio" error={errors.teacher_student_ratio}>{num('teacher_student_ratio')}</Field>
        <Field label="Infrastructure score" error={errors.school_infrastructure_score}>{num('school_infrastructure_score')}</Field>
      </Section>

      <Section title="6. Risk Factors">
        <Grid size={{ xs: 12, sm: 6 }}>{flag('bullying_incidents_reported')}</Grid>
        <Grid size={{ xs: 12, sm: 6 }}>{flag('health_issues_reported')}</Grid>
        <Grid size={{ xs: 12, sm: 6 }}>{flag('special_needs_status')}</Grid>
        <Grid size={{ xs: 12, sm: 6 }}>{flag('early_marriage_risk_flag')}</Grid>
        <Grid size={{ xs: 12, sm: 6 }}>{flag('community_conflict_zone')}</Grid>
        <Grid size={{ xs: 12, sm: 6 }}>{flag('language_barrier_flag')}</Grid>
      </Section>

      <Section title="7. Support Factors">
        <Grid size={{ xs: 12, sm: 6 }}>{flag('free_meal_program_enrolled')}</Grid>
        <Grid size={{ xs: 12, sm: 6 }}>{flag('scholarship_received')}</Grid>
        <Grid size={{ xs: 12, sm: 6 }}>{flag('ngo_intervention_present')}</Grid>
        <Grid size={{ xs: 12, sm: 6 }}>{flag('literacy_program_enrolled')}</Grid>
      </Section>

      <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
        <Button type="submit" variant="contained" size="large" disabled={loading}>
          {loading ? 'Assessing risk…' : 'Assess dropout risk'}
        </Button>
        <Button type="button" variant="outlined" onClick={onReset} disabled={loading}>
          Reset form
        </Button>
      </Box>
    </Box>
  )
}
