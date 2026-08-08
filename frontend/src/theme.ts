import { createTheme } from '@mui/material/styles'

export const theme = createTheme({
  palette: {
    mode: 'light',
    primary: { main: '#1565c0' },
    secondary: { main: '#00838f' },
    background: { default: '#f4f7fb' },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h4: { fontWeight: 700 },
  },
  shape: { borderRadius: 12 },
})

export const riskColors: Record<string, string> = {
  Low: '#2e7d32',
  Medium: '#f9a825',
  High: '#ef6c00',
  Critical: '#c62828',
}
