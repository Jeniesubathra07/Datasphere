import axios from 'axios'
import type { ModelSchema, PredictionResponse, StudentInput } from '../types/student'

const DEFAULT_DEV_API_URL = 'http://127.0.0.1:8080'

function resolveApiBaseUrl(): string {
  const configured = import.meta.env.VITE_API_BASE_URL?.trim()
  if (configured) return configured
  // Bypass the Vite proxy in local dev — more reliable on Windows than http-proxy.
  if (import.meta.env.DEV) return DEFAULT_DEV_API_URL
  return ''
}

const api = axios.create({
  baseURL: resolveApiBaseUrl(),
  headers: { 'Content-Type': 'application/json' },
})

export async function fetchModelSchema(): Promise<ModelSchema> {
  const { data } = await api.get<ModelSchema>('/api/model/schema')
  return data
}

export async function predictRisk(student: StudentInput, explain = true): Promise<PredictionResponse> {
  const endpoint = explain ? '/predict/explain' : '/predict'
  const { data } = await api.post<PredictionResponse>(endpoint, student)
  return data
}

export async function checkHealth(): Promise<{
  status: string
  model_loaded: boolean
  model_load_error?: string
}> {
  const { data } = await api.get('/health')
  return data
}

export function formatApiError(error: unknown): string {
  if (axios.isAxiosError(error)) {
    const detail = error.response?.data?.detail
    if (typeof detail === 'string') return detail
    if (Array.isArray(detail)) {
      return detail.map((d) => d.msg || JSON.stringify(d)).join('; ')
    }
    return error.message
  }
  return 'An unexpected error occurred.'
}
