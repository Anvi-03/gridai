// GridPulse AI — TypeScript type contracts
// Exact mirrors of backend Pydantic response models.

// ── Telemetry ─────────────────────────────────────────────────────────────────

export interface TelemetryReading {
  id: string;
  meter_id: string;
  timestamp: string;
  voltage: number;
  current: number;
  power_factor: number;
  // ML analytics — null until the background analytics pipeline completes
  is_anomalous: boolean | null;
  anomaly_type: string | null;
  anomaly_confidence: number | null;
  predicted_load_24h: number | null;
  // Economic impact — null for healthy readings
  revenue_loss_inr: number | null;
  outage_risk_score: number | null;
  // Edge AI metadata (Feature 6)
  edge_flagged: boolean;
  edge_confidence: number | null;
}

// ── Per-meter aggregate stats (GET /api/v1/stats) ────────────────────────────

export interface StatsRow {
  meter_id: string;
  total_readings: number;
  avg_voltage: number;
  avg_current: number;
  avg_power_factor: number;
  last_seen: string;
}

// ── Forecast (GET /api/v1/grid/forecast) ─────────────────────────────────────

export type RiskZone = 'low' | 'medium' | 'high' | 'critical';

export interface MeterForecastItem {
  meter_id: string;
  outage_risk_score: number;
  risk_zone: RiskZone;
  predicted_peak_w: number;
  predicted_avg_w: number;
  capacity_threshold_w: number;
  load_ratio: number;
  generated_at: string;
  forecast_horizon: string;
  model_name: string;
}

export interface PeakTimeItem {
  meter_id: string;
  predicted_peak_w: number;
  forecast_horizon: string;
  risk_zone: string;
}

export interface FleetSummary {
  low_risk_count: number;
  medium_risk_count: number;
  high_risk_count: number;
  critical_count: number;
  max_risk_score: number;
  avg_risk_score: number;
  systemic_outage_probability: number;
}

export interface ForecastReport {
  generated_at: string;
  total_meters_active: number;
  fleet_summary: FleetSummary;
  high_risk_zones: MeterForecastItem[];
  predicted_peak_times: PeakTimeItem[];
  outage_probability_matrix: MeterForecastItem[];
}

// ── GenAI Copilot (POST /api/v1/copilot/query) ───────────────────────────────

export interface CopilotResponse {
  answer: string;
  model: string;
  context_chars: number;
  input_tokens: number | null;
  output_tokens: number | null;
}

// ── Health (GET /api/v1/health) ───────────────────────────────────────────────

export interface HealthResponse {
  status: string;
  version: string;
  database: string;
}

// ── Merged view per meter (derived in UI) ────────────────────────────────────

export interface MeterNode {
  meter_id: string;
  // Stats aggregates
  avg_voltage: number;
  avg_current: number;
  avg_power_factor: number;
  total_readings: number;
  last_seen: string;
  // Latest reading signals
  is_anomalous: boolean;
  anomaly_type: string | null;
  anomaly_confidence: number | null;
  edge_flagged: boolean;
  edge_confidence: number | null;
  revenue_loss_inr: number | null;
  latest_voltage: number | null;
  latest_current: number | null;
  // Forecast
  outage_risk_score: number;
  risk_zone: RiskZone;
  predicted_avg_w: number;
  predicted_peak_w: number;
}

// ── Chart data shape ──────────────────────────────────────────────────────────

export interface ChartPoint {
  time: string;
  actual: number | null;
  forecast: number | null;
  risk: number | null;
}
