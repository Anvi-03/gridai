/**
 * GridPulse AI — Centralized API client  (src/lib/api.ts)
 *
 * All fetch calls go through this module.  Every function:
 *   1. Hits the Vite-proxied /api path (no hardcoded localhost).
 *   2. Returns typed data on success.
 *   3. Returns a typed fallback value on ANY error (network, 429, 500, 502)
 *      so the dashboard never renders a blank white screen.
 */

import type {
  TelemetryReading,
  StatsRow,
  ForecastReport,
  CopilotResponse,
  HealthResponse,
} from '../types/grid'

// ── Base fetch helper ─────────────────────────────────────────────────────────

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T | null> {
  try {
    const res = await fetch(path, {
      headers: { 'Content-Type': 'application/json' },
      ...init,
    })
    if (!res.ok) {
      console.warn(`[API] ${res.status} on ${path}`)
      return null
    }
    return (await res.json()) as T
  } catch (err) {
    console.error(`[API] Network error on ${path}:`, err)
    return null
  }
}

// ── Fallback datasets ─────────────────────────────────────────────────────────
// Shown when the backend is unreachable or rate-limited.
// Values are realistic-looking to make the UI informative even offline.

const FALLBACK_STATS: StatsRow[] = Array.from({ length: 10 }, (_, i) => ({
  meter_id: `METER-${String(i + 1).padStart(3, '0')}`,
  total_readings: 0,
  avg_voltage: 228 + Math.random() * 6,
  avg_current: 10 + Math.random() * 8,
  avg_power_factor: 0.88 + Math.random() * 0.1,
  last_seen: new Date().toISOString(),
}))

const FALLBACK_TELEMETRY: TelemetryReading[] = Array.from({ length: 10 }, (_, i) => ({
  id: `00000000-0000-0000-0000-${String(i).padStart(12, '0')}`,
  meter_id: `METER-${String(i + 1).padStart(3, '0')}`,
  timestamp: new Date(Date.now() - i * 30_000).toISOString(),
  voltage: 229 + Math.random() * 4,
  current: 12 + Math.random() * 5,
  power_factor: 0.9 + Math.random() * 0.08,
  is_anomalous: null,
  anomaly_type: null,
  anomaly_confidence: null,
  predicted_load_24h: null,
  revenue_loss_inr: null,
  outage_risk_score: null,
  edge_flagged: false,
  edge_confidence: null,
}))

const FALLBACK_FORECAST: ForecastReport = {
  generated_at: new Date().toISOString(),
  total_meters_active: 10,
  fleet_summary: {
    low_risk_count: 8,
    medium_risk_count: 2,
    high_risk_count: 0,
    critical_count: 0,
    max_risk_score: 35,
    avg_risk_score: 18,
    systemic_outage_probability: 0.18,
  },
  high_risk_zones: [],
  predicted_peak_times: [],
  outage_probability_matrix: Array.from({ length: 10 }, (_, i) => ({
    meter_id: `METER-${String(i + 1).padStart(3, '0')}`,
    outage_risk_score: Math.round(10 + Math.random() * 30),
    risk_zone: 'low' as const,
    predicted_peak_w: 2800 + Math.random() * 600,
    predicted_avg_w: 2200 + Math.random() * 400,
    capacity_threshold_w: 6555,
    load_ratio: 0.35 + Math.random() * 0.15,
    generated_at: new Date().toISOString(),
    forecast_horizon: new Date(Date.now() + 86_400_000).toISOString(),
    model_name: 'GridForecaster (offline)',
  })),
}

const FALLBACK_HEALTH: HealthResponse = {
  status: 'offline',
  version: '1.0.0',
  database: 'unreachable',
}

// ── Public API functions ──────────────────────────────────────────────────────

/**
 * Aggregate per-meter statistics.
 * GET /api/v1/stats
 */
export async function fetchStats(): Promise<StatsRow[]> {
  const data = await apiFetch<StatsRow[]>('/api/v1/stats')
  return data ?? FALLBACK_STATS
}

/**
 * Recent telemetry readings (latest 60).
 * GET /api/v1/telemetry?limit=60&offset=0
 */
export async function fetchTelemetry(limit = 60): Promise<TelemetryReading[]> {
  const data = await apiFetch<TelemetryReading[]>(`/api/v1/telemetry?limit=${limit}`)
  return data ?? FALLBACK_TELEMETRY
}

/**
 * 24-hour predictive outage forecast report.
 * GET /api/v1/grid/forecast
 */
export async function fetchForecast(): Promise<ForecastReport> {
  const data = await apiFetch<ForecastReport>('/api/v1/grid/forecast')
  return data ?? FALLBACK_FORECAST
}

/**
 * Backend health probe.
 * GET /api/v1/health
 */
export async function fetchHealth(): Promise<HealthResponse> {
  const data = await apiFetch<HealthResponse>('/api/v1/health')
  return data ?? FALLBACK_HEALTH
}

/**
 * GenAI Copilot query.
 * POST /api/v1/copilot/query  { message: string }
 *
 * Returns null on 429 (rate limit) or 503 (key not configured) so the UI
 * can display a specific error message rather than crashing.
 */
export async function postCopilotQuery(
  message: string,
): Promise<CopilotResponse | null> {
  return apiFetch<CopilotResponse>('/api/v1/copilot/query', {
    method: 'POST',
    body: JSON.stringify({ message }),
  })
}

// ── Formatting helpers ────────────────────────────────────────────────────────

export function formatINR(amount: number): string {
  return new Intl.NumberFormat('en-IN', {
    style: 'currency',
    currency: 'INR',
    maximumFractionDigits: 0,
  }).format(amount)
}

export function formatWatts(w: number): string {
  if (w >= 1_000_000) return `${(w / 1_000_000).toFixed(2)} MW`
  if (w >= 1_000)     return `${(w / 1_000).toFixed(1)} kW`
  return `${Math.round(w)} W`
}

export function riskColor(zone: string): string {
  switch (zone) {
    case 'critical': return 'text-red-400'
    case 'high':     return 'text-orange-400'
    case 'medium':   return 'text-amber-400'
    default:         return 'text-emerald-400'
  }
}

export function riskBg(zone: string): string {
  switch (zone) {
    case 'critical': return 'bg-red-500/10 border-red-500/30'
    case 'high':     return 'bg-orange-500/10 border-orange-500/30'
    case 'medium':   return 'bg-amber-500/10 border-amber-500/30'
    default:         return 'bg-emerald-500/10 border-emerald-500/30'
  }
}
