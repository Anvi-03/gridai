/**
 * GridPulse AI — ForecastChart.tsx
 *
 * Design: High-density interactive AreaChart plotting:
 *   • "Actual Load (W)"   — real power (V×I×PF) from recent telemetry readings
 *   • "24h Forecast (W)"  — fleet-average predicted_avg_w from the forecast API
 *
 * Overhauled to Siemens / ABB monitoring dashboard grade:
 *   - Enhanced tooltips with micro indicators.
 *   - Custom grid shading and hover states.
 */

import { useMemo } from 'react'
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts'
import { TrendingUp } from 'lucide-react'
import type { TelemetryReading, ForecastReport, ChartPoint } from '../types/grid'
import { formatWatts } from '../lib/api'

interface ForecastChartProps {
  telemetry: TelemetryReading[]
  forecast: ForecastReport | null
}

// ── Build unified chart dataset ───────────────────────────────────────────────

function buildChartData(
  telemetry: TelemetryReading[],
  forecast: ForecastReport | null,
): ChartPoint[] {
  const historical: ChartPoint[] = [...telemetry]
    .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime())
    .slice(-20)
    .map(t => ({
      time: new Date(t.timestamp).toLocaleTimeString('en-IN', {
        hour: '2-digit',
        minute: '2-digit',
        hour12: false,
      }),
      actual: Math.round(t.voltage * t.current * t.power_factor),
      forecast: null,
      risk: t.outage_risk_score ?? null,
    }))

  if (forecast && forecast.outage_probability_matrix.length > 0) {
    const matrix = forecast.outage_probability_matrix
    const avgNow  = matrix.reduce((s, m) => s + m.predicted_avg_w, 0) / matrix.length
    const avgPeak = matrix.reduce((s, m) => s + m.predicted_peak_w, 0) / matrix.length

    if (historical.length > 0) {
      historical[historical.length - 1].forecast = Math.round(avgNow * 0.95)
    }

    const now = new Date()
    const steps = [
      { h: 6,  w: avgNow + (avgPeak - avgNow) * 0.35 },
      { h: 12, w: avgNow + (avgPeak - avgNow) * 0.65 },
      { h: 18, w: avgPeak * 0.92 },
      { h: 24, w: avgPeak },
    ]

    steps.forEach(({ h, w }) => {
      const t = new Date(now.getTime() + h * 3_600_000)
      historical.push({
        time: `+${h}h (${t.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', hour12: false })})`,
        actual: null,
        forecast: Math.round(w),
        risk: null,
      })
    })
  }

  return historical
}

// ── Custom tooltip ────────────────────────────────────────────────────────────

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-slate-950/95 border border-slate-800/80 rounded-xl p-3 text-xs shadow-2xl backdrop-blur-md">
      <p className="text-slate-500 font-mono mb-2 text-[10px] tracking-wider uppercase font-semibold">{label}</p>
      {payload.map((p: { color: string; name: string; value: number | null }) => (
        p.value != null && (
          <div key={p.name} className="flex items-center gap-2 mb-1">
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: p.color }} />
            <span className="text-slate-400 font-mono text-[10px]">{p.name}:</span>
            <span className="text-slate-100 font-mono font-bold">
              {formatWatts(p.value)}
            </span>
          </div>
        )
      ))}
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────────────────

export function ForecastChart({ telemetry, forecast }: ForecastChartProps) {
  const data = useMemo(() => buildChartData(telemetry, forecast), [telemetry, forecast])

  const capacityW = forecast?.outage_probability_matrix[0]?.capacity_threshold_w ?? null
  const avgRisk   = forecast?.fleet_summary.avg_risk_score ?? null

  const bridgeIdx = data.findIndex(d => d.actual != null && d.forecast != null)

  return (
    <div className="glass-card p-5 flex flex-col gap-4 border-slate-800/80">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <TrendingUp size={14} className="text-indigo-400" />
          <h2 className="text-xs font-semibold text-slate-300 uppercase tracking-widest font-mono">
            Load Trajectory · 24h Predictive Forecast
          </h2>
        </div>
        <div className="flex items-center gap-2 text-xs text-slate-500 font-mono">
          {avgRisk != null && (
            <span>
              Fleet Risk Avg:{' '}
              <span className="text-slate-300 font-semibold">{avgRisk.toFixed(0)}%</span>
            </span>
          )}
          <span className="text-slate-700">|</span>
          <span className="text-slate-500 text-[10px] font-semibold">
            Ridge ML Regressor
          </span>
        </div>
      </div>

      {/* Chart */}
      <div className="h-64">
        {data.length === 0 ? (
          <div className="h-full flex items-center justify-center text-slate-600 text-sm">
            Waiting for telemetry data…
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data} margin={{ top: 4, right: 12, left: -8, bottom: 0 }}>
              <defs>
                <linearGradient id="gradActual" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor="#6366f1" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="gradForecast" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor="#06b6d4" stopOpacity={0.25} />
                  <stop offset="95%" stopColor="#06b6d4" stopOpacity={0} />
                </linearGradient>
              </defs>

              <CartesianGrid
                strokeDasharray="3 3"
                stroke="rgba(148,163,184,0.05)"
                vertical={false}
              />

              <XAxis
                dataKey="time"
                tick={{ fill: '#64748b', fontSize: 9, fontFamily: 'monospace' }}
                tickLine={false}
                axisLine={false}
                interval="preserveStartEnd"
              />

              <YAxis
                tickFormatter={v => formatWatts(v)}
                tick={{ fill: '#64748b', fontSize: 9, fontFamily: 'monospace' }}
                tickLine={false}
                axisLine={false}
                width={56}
              />

              <Tooltip content={<CustomTooltip />} />

              <Legend
                wrapperStyle={{ fontSize: 10, color: '#94a3b8', paddingTop: 8, fontFamily: 'monospace' }}
                iconType="circle"
                iconSize={6}
              />

              {capacityW && (
                <ReferenceLine
                  y={capacityW}
                  stroke="rgba(239,68,68,0.35)"
                  strokeDasharray="6 3"
                  label={{
                    value: 'Substation Capacity limit',
                    position: 'insideTopRight',
                    fill: '#f87171',
                    fontSize: 8,
                    fontFamily: 'monospace'
                  }}
                />
              )}

              {bridgeIdx >= 0 && data[bridgeIdx] && (
                <ReferenceLine
                  x={data[bridgeIdx].time}
                  stroke="rgba(148,163,184,0.12)"
                  strokeDasharray="4 4"
                />
              )}

              <Area
                type="monotone"
                dataKey="actual"
                name="Actual Load"
                stroke="#6366f1"
                strokeWidth={2}
                fill="url(#gradActual)"
                dot={false}
                connectNulls={false}
                activeDot={{ r: 4, fill: '#6366f1', strokeWidth: 0 }}
              />

              <Area
                type="monotone"
                dataKey="forecast"
                name="24h Forecast"
                stroke="#06b6d4"
                strokeWidth={2}
                fill="url(#gradForecast)"
                strokeDasharray="6 3"
                dot={false}
                connectNulls={false}
                activeDot={{ r: 4, fill: '#06b6d4', strokeWidth: 0 }}
              />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* Legend Note */}
      <div className="flex items-center gap-4 text-[9px] text-slate-600 font-mono">
        <div className="flex items-center gap-1.5">
          <span className="w-3 h-0.5 bg-indigo-500 rounded" />
          <span>Historical Load (V×I×PF)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-3 h-0.5 bg-cyan-400 rounded border-b border-dashed border-cyan-400" />
          <span>Ridge Forecast</span>
        </div>
        {capacityW && (
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-0.5 bg-red-500/40 rounded" />
            <span>Substation Capacity</span>
          </div>
        )}
      </div>
    </div>
  )
}
