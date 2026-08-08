import { useEffect, useState } from 'react'
import {
  Alert,
  Box,
  CircularProgress,
  Container,
  Typography,
} from '@mui/material'
import StudentForm from '../components/StudentForm'
import PredictionResult from '../components/PredictionResult'
import { checkHealth, fetchModelSchema, formatApiError, predictRisk } from '../api/client'
import type { ModelSchema, PredictionResponse, StudentInput } from '../types/student'
import { createDefaultStudent, validateStudent } from '../utils/validation'

export default function AssessmentPage() {
  const [schema, setSchema] = useState<ModelSchema | null>(null)
  const [student, setStudent] = useState<StudentInput | null>(null)
  const [errors, setErrors] = useState<Partial<Record<keyof StudentInput, string>>>({})
  const [result, setResult] = useState<PredictionResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [bootError, setBootError] = useState<string | null>(null)
  const [apiError, setApiError] = useState<string | null>(null)

  useEffect(() => {
    async function boot() {
      try {
        const health = await checkHealth()
        if (!health.model_loaded) {
          const detail = health.model_load_error
          setBootError(
            detail
              ? `ML model failed to load: ${detail}`
              : 'ML model is not loaded. Run: python scripts\\run_stage2_pipeline.py',
          )
          return
        }
        const loadedSchema = await fetchModelSchema()
        setSchema(loadedSchema)
        setStudent(createDefaultStudent(loadedSchema))
      } catch (err) {
        setBootError(formatApiError(err))
      }
    }
    boot()
  }, [])

  const handleSubmit = async () => {
    if (!schema || !student) return
    const clientErrors = validateStudent(student, schema)
    setErrors(clientErrors)
    if (Object.keys(clientErrors).length > 0) return

    setLoading(true)
    setApiError(null)
    try {
      const response = await predictRisk(student, true)
      setResult(response)
    } catch (err) {
      setApiError(formatApiError(err))
      setResult(null)
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    if (schema) setStudent(createDefaultStudent(schema))
    setErrors({})
    setResult(null)
    setApiError(null)
  }

  if (bootError) {
    return (
      <Container maxWidth="md" sx={{ py: 6 }}>
        <Alert severity="error">{bootError}</Alert>
      </Container>
    )
  }

  if (!schema || !student) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', py: 10 }}>
        <CircularProgress />
      </Box>
    )
  }

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Typography variant="h4" gutterBottom>
        AI-assisted student dropout risk assessment
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        Enter a student profile to receive a model-based risk assessment with probability estimates.
        This tool supports decision-making; it does not determine a student&apos;s future with certainty.
      </Typography>

      {apiError ? <Alert severity="error" sx={{ mb: 2 }}>{apiError}</Alert> : null}

      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', lg: result ? '1fr 1fr' : '1fr' }, gap: 3 }}>
        <StudentForm
          schema={schema}
          value={student}
          errors={errors}
          onChange={setStudent}
          onSubmit={handleSubmit}
          onReset={handleReset}
          loading={loading}
        />
        {result ? (
          <PredictionResult result={result} inputSummary={student} />
        ) : null}
      </Box>
    </Container>
  )
}
