import {
  Box,
  Chip,
  LinearProgress,
  Paper,
  Stack,
  Typography,
} from '@mui/material'
import type { PredictionResponse, StudentInput } from '../types/student'
import { riskColors } from '../theme'

interface Props {
  result: PredictionResponse
  inputSummary: StudentInput
}

export default function PredictionResult({ result, inputSummary }: Props) {
  const { prediction, explanation, model } = result
  const color = riskColors[prediction.risk_level] || '#1565c0'
  const sortedProbs = Object.entries(prediction.probabilities).sort((a, b) => b[1] - a[1])

  return (
    <Stack spacing={3}>
      <Paper sx={{ p: 3, borderLeft: `6px solid ${color}` }}>
        <Typography variant="overline" color="text.secondary">
          AI-assisted assessment — not a certain prediction of future outcomes
        </Typography>
        <Typography variant="h4" sx={{ color, mt: 1 }}>
          {prediction.risk_level} Risk
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
          Model: {model.selected_model} (v{model.version})
        </Typography>
      </Paper>

      <Paper sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>Class probability distribution</Typography>
        <Stack spacing={2}>
          {sortedProbs.map(([label, prob]) => (
            <Box key={label}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                <Typography variant="body2">{label}</Typography>
                <Typography variant="body2">{(prob * 100).toFixed(1)}%</Typography>
              </Box>
              <LinearProgress
                variant="determinate"
                value={prob * 100}
                sx={{
                  height: 10,
                  borderRadius: 5,
                  backgroundColor: '#e0e0e0',
                  '& .MuiLinearProgress-bar': { backgroundColor: riskColors[label] || '#90a4ae' },
                }}
              />
            </Box>
          ))}
        </Stack>
      </Paper>

      {explanation ? (
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>Factors associated with this prediction</Typography>
          <Typography variant="body2" sx={{ mb: 2 }}>{explanation.summary}</Typography>
          <Stack direction="row" sx={{ flexWrap: 'wrap', gap: 1 }}>
            {explanation.important_factors.slice(0, 8).map((f) => (
              <Chip
                key={f.feature}
                label={`${f.feature.replace(/__/g, ' ')} (${f.importance.toFixed(3)})`}
                variant="outlined"
                size="small"
              />
            ))}
          </Stack>
        </Paper>
      ) : null}

      <Paper sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>Input summary</Typography>
        <Typography variant="body2">Age {inputSummary.age} · Grade {inputSummary.grade_level} · {inputSummary.gender}</Typography>
        <Typography variant="body2">Attendance {inputSummary.attendance_rate_pct}% · Test score {inputSummary.average_test_score_pct}%</Typography>
        <Typography variant="body2">{inputSummary.country}, {inputSummary.state_province} ({inputSummary.location_type})</Typography>
      </Paper>
    </Stack>
  )
}
