import { CssBaseline, ThemeProvider } from '@mui/material'
import AssessmentPage from './pages/AssessmentPage'
import { theme } from './theme'

export default function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <AssessmentPage />
    </ThemeProvider>
  )
}
