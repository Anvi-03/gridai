/**
 * GridPulse AI — MetricCards.tsx
 *
 * Four glassmorphism KPI cards shown at the top of the dashboard:
 *   1. Active Meters          — total meters reporting
 *   2. Anomaly Alerts         — flashing red badge when > 0
 *   3. Total Revenue Loss (₹) — sum of recent anomaly economic impact
 *   4. System Outage Risk     — fleet max risk score, color-coded
 */

import { Activity, AlertTriangle, TrendingDown, ShieldAlert } from 'lucide-react'
import type { TelemetryReading, StatsRow, ForecastReport } from '../types/grid'
import { formatINR } from '../lib/api'
import { clsx } from 'clsx'

interface MetricCardsProps {
  stats: StatsRow[]
  telemetry: TelemetryReading[]
  forecast: ForecastReport | null
  isLoading: boolean
}

interface CardProps {
  title: string
  value: string | number
  subtitle?: string
  icon: React.ReactNode
  accentClass: string
  glowClass?: string
  flash?: boolean
  loading?: boolean
}

function StatCard({ title, value, subtitle, icon, accentClass, glowClass, flash, loading }: CardProps) {
  return (
    <div
      className={clsx(
        'glass-card p-5 flex flex-col gap-3 relative overflow-hidden transition-all duration-300',
        'hover:border-slate-600/60 hover:-translate-y-0.5',
        flash && 'animate-pulse-red',
        glowClass,
      )}
    >
      {/* Background glow orb */}
      <div
        className={clsx(
          'absolute -top-6 -right-6 w-24 h-24 rounded-full blur-2xl opacity-20 pointer-events-none',
          accentClass,
        )}
      />

      <div className="flex items-center justify-between">
        <span className="text-slate-400 text-xs font-medium uppercase tracking-wider">
          {title}
        </span>
        <div className={clsx('p-2 rounded-lg', accentClass.replace('bg-', 'bg-').concat('/20'))}>
          {icon}
        </div>
      </div>

      {loading ? (
        <div className="skeleton h-9 w-3/4" />
      ) : (
        <div className="flex items-end gap-2">
          <span className="text-3xl font-bold text-slate-100 tracking-tight leading-none">
            {value}
          </span>
          {flash && (
            <span className="mb-0.5 flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-red-500 animate-blink" />
              <span className="text-red-400 text-xs font-medium">LIVE</span>
            </span>
          )}
        </div>
      )}

      {subtitle && (
        <p className="text-slate-500 text-xs leading-snug">{subtitle}</p>
      )}
    </div>
  )
}

export function MetricCards({ stats, telemetry, forecast, isLoading }: MetricCardsProps) {
  // ── Derived metrics ──────────────────────────────────────────────────────────
  const activeMeterCount = stats.length

  const anomalyCount = telemetry.filter(t => t.is_anomalous === true).length
  const edgeFlaggedCount = telemetry.filter(t => t.edge_flagged).length

  const totalRevenueLoss = telemetry.reduce(
    (sum, t) => sum + (t.revenue_loss_inr ?? 0),
    0,
  )

  const maxRiskScore = forecast?.fleet_summary.max_risk_score ?? 0
  const systemic = forecast?.fleet_summary.systemic_outage_probability ?? 0
  const riskLabel =
    maxRiskScore >= 70 ? 'CRITICAL'
    : maxRiskScore >= 50 ? 'HIGH'
    : maxRiskScore >= 30 ? 'MEDIUM'
    : 'LOW'

  const riskAccent =
    maxRiskScore >= 70 ? 'bg-red-500 text-red-400'
    : maxRiskScore >= 50 ? 'bg-orange-500 text-orange-400'
    : maxRiskScore >= 30 ? 'bg-amber-500 text-amber-400'
    : 'bg-emerald-500 text-emerald-400'

  return (
    <div className="grid grid-cols-2 xl:grid-cols-4 gap-4">
      {/* Card 1 — Active Meters */}
      <StatCard
        title="Active Meters"
        value={activeMeterCount}
        subtitle={`${edgeFlaggedCount} edge-screened this cycle`}
        icon={<Activity size={16} className="text-indigo-400" />}
        accentClass="bg-indigo-500"
        loading={isLoading && activeMeterCount === 0}
      />

      {/* Card 2 — Anomaly Alerts */}
      <StatCard
        title="Active Anomaly Alerts"
        value={anomalyCount}
        subtitle={
          anomalyCount > 0
            ? `${anomalyCount} meter${anomalyCount !== 1 ? 's' : ''} flagged by ML pipeline`
            : 'All meters nominal'
        }
        icon={<AlertTriangle size={16} className={anomalyCount > 0 ? 'text-red-400' : 'text-emerald-400'} />}
        accentClass={anomalyCount > 0 ? 'bg-red-500' : 'bg-emerald-500'}
        flash={anomalyCount > 0}
        loading={isLoading && anomalyCount === 0}
      />

      {/* Card 3 — Revenue Loss */}
      <StatCard
        title="Total Revenue Loss"
        value={formatINR(totalRevenueLoss)}
        subtitle="Cumulative impact from current anomaly events"
        icon={<TrendingDown size={16} className="text-amber-400" />}
        accentClass="bg-amber-500"
        loading={isLoading}
      />

      {/* Card 4 — System Outage Risk */}
      <StatCard
        title="System Outage Risk"
        value={`${maxRiskScore}/100`}
        subtitle={`${riskLabel} · Systemic P = ${(systemic * 100).toFixed(1)}%`}
        icon={<ShieldAlert size={16} className={riskAccent.split(' ')[1]} />}
        accentClass={riskAccent.split(' ')[0]}
        glowClass={maxRiskScore >= 70 ? 'animate-pulse-red' : maxRiskScore >= 50 ? 'animate-pulse-amber' : undefined}
        loading={isLoading && forecast === null}
      />
    </div>
  )
}
