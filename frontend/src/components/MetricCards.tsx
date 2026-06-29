/**
 * GridPulse AI — MetricCards.tsx
 *
 * Design: GE Digital / Tesla Energy Enterprise style executive dashboard overview.
 * Displays 8 KPI monitoring blocks across a dense, highly scannable header:
 *   1. Grid Health Score       - inverse of maximum outage risk
 *   2. Active Anomaly Alerts   - flashing alert counter
 *   3. Predicted Outages       - risk assessment counts
 *   4. Revenue Loss Today      - formatted in INR rupees
 *   5. Energy Efficiency       - derived fleet power factor
 *   6. Carbon Savings (tons)   - sustainability impact calculator
 *   7. Critical Assets count   - vulnerable segment triggers
 *   8. System Uptime           - grid availability rating
 */

import {
  Activity,
  AlertTriangle,
  TrendingDown,
  ShieldAlert,
  Flame,
  Zap,
  TrendingUp,
  Cpu,
} from 'lucide-react'
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
  borderColorClass?: string
  flash?: boolean
  loading?: boolean
}

function StatCard({
  title,
  value,
  subtitle,
  icon,
  accentClass,
  borderColorClass,
  flash,
  loading,
}: CardProps) {
  return (
    <div
      className={clsx(
        'glass-card p-4 flex flex-col gap-2 relative overflow-hidden transition-all duration-300 border-slate-800/80',
        'hover:border-slate-700/60 hover:-translate-y-0.5 shadow-md',
        flash && 'animate-pulse-red border-red-500/30 bg-red-950/5',
        borderColorClass
      )}
    >
      {/* Background radial highlight */}
      <div
        className={clsx(
          'absolute -top-10 -right-10 w-20 h-20 rounded-full blur-2xl opacity-10 pointer-events-none',
          accentClass
        )}
      />

      <div className="flex items-center justify-between">
        <span className="text-slate-500 text-[9px] font-semibold uppercase tracking-wider leading-none">
          {title}
        </span>
        <div className={clsx('p-1.5 rounded-lg text-xs leading-none bg-slate-800/50', accentClass.replace('bg-', 'text-'))}>
          {icon}
        </div>
      </div>

      {loading ? (
        <div className="skeleton h-6 w-3/4 my-1" />
      ) : (
        <div className="flex items-baseline gap-1 mt-1">
          <span className="text-lg font-bold text-slate-100 tracking-tight leading-none font-mono">
            {value}
          </span>
          {flash && (
            <span className="flex items-center gap-0.5 ml-1">
              <span className="w-1.5 h-1.5 rounded-full bg-red-500 animate-blink" />
              <span className="text-red-400 text-[8px] font-bold font-mono">CRIT</span>
            </span>
          )}
        </div>
      )}

      {subtitle && (
        <p className="text-slate-500 text-[9px] leading-tight font-mono font-medium truncate">{subtitle}</p>
      )}
    </div>
  )
}

export function MetricCards({ stats, telemetry, forecast, isLoading }: MetricCardsProps) {
  const activeMeterCount = stats.length

  // Derived Values
  const anomalyCount = telemetry.filter(t => t.is_anomalous === true).length
  const edgeFlaggedCount = telemetry.filter(t => t.edge_flagged).length

  const totalRevenueLoss = telemetry.reduce(
    (sum, t) => sum + (t.revenue_loss_inr ?? 0),
    0,
  )

  const maxRiskScore = forecast?.fleet_summary.max_risk_score ?? 0
  const systemic = forecast?.fleet_summary.systemic_outage_probability ?? 0
  const predictedOutageCount = forecast?.high_risk_zones.length ?? 0

  // 1. Grid Health Score
  const healthScore = Math.max(0, 100 - maxRiskScore)
  const healthStatus = healthScore >= 85 ? '🟢 Optimal' : healthScore >= 60 ? '🟡 Warning' : '🔴 Critical'
  const healthBorder = healthScore < 60 ? 'border-red-500/25 bg-red-950/5' : healthScore < 85 ? 'border-amber-500/20' : undefined

  // 5. Energy Efficiency (derived from stats average power factor)
  const avgPF = stats.length > 0
    ? stats.reduce((sum, s) => sum + s.avg_power_factor, 0) / stats.length
    : 0.92
  const efficiencyPercent = (avgPF * 100).toFixed(1)

  // 6. Carbon Savings (Sustainability indicator)
  const carbonSavedTons = (activeMeterCount * 0.12 * (avgPF / 0.95)).toFixed(2)

  // 7. Critical Assets
  const criticalAssets = predictedOutageCount + (anomalyCount > 0 ? 1 : 0)

  // 8. System Uptime
  const uptime = anomalyCount > 0 ? '99.92%' : '99.99%'

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-8 gap-3">
      {/* 1. Grid Health Score */}
      <StatCard
        title="Grid Health"
        value={`${healthScore}%`}
        subtitle={healthStatus}
        icon={<Activity size={12} />}
        accentClass={healthScore < 60 ? 'bg-red-500' : healthScore < 85 ? 'bg-amber-500' : 'bg-emerald-500'}
        borderColorClass={healthBorder}
        loading={isLoading && stats.length === 0}
      />

      {/* 2. Active Anomaly Alerts */}
      <StatCard
        title="Active Alerts"
        value={anomalyCount}
        subtitle={anomalyCount > 0 ? 'Alert queue active' : 'All systems nominal'}
        icon={<AlertTriangle size={12} />}
        accentClass={anomalyCount > 0 ? 'bg-red-500' : 'bg-emerald-500'}
        flash={anomalyCount > 0}
        loading={isLoading}
      />

      {/* 3. Predicted Outages */}
      <StatCard
        title="Predicted Outages"
        value={predictedOutageCount}
        subtitle={`${forecast?.fleet_summary.high_risk_count ?? 0} high-risk zones`}
        icon={<ShieldAlert size={12} />}
        accentClass="bg-orange-500"
        loading={isLoading && forecast === null}
      />

      {/* 4. Revenue Loss Today */}
      <StatCard
        title="Revenue Loss"
        value={formatINR(totalRevenueLoss)}
        subtitle="Current billing cycles"
        icon={<TrendingDown size={12} />}
        accentClass="bg-red-400"
        loading={isLoading}
      />

      {/* 5. Energy Efficiency */}
      <StatCard
        title="Grid Efficiency"
        value={`${efficiencyPercent}%`}
        subtitle={`PF: ${avgPF.toFixed(2)} Target: 0.95`}
        icon={<Zap size={12} />}
        accentClass="bg-indigo-500"
        loading={isLoading && stats.length === 0}
      />

      {/* 6. Carbon Savings */}
      <StatCard
        title="Carbon Offset"
        value={`${carbonSavedTons} t`}
        subtitle="Estimated CO2 avoided"
        icon={<TrendingUp size={12} />}
        accentClass="bg-emerald-400"
        loading={isLoading}
      />

      {/* 7. Critical Assets */}
      <StatCard
        title="Critical Assets"
        value={criticalAssets}
        subtitle={`${activeMeterCount} connected assets`}
        icon={<Cpu size={12} />}
        accentClass="bg-indigo-400"
        loading={isLoading}
      />

      {/* 8. System Uptime */}
      <StatCard
        title="System Uptime"
        value={uptime}
        subtitle="Core services active"
        icon={<Flame size={12} />}
        accentClass="bg-emerald-500"
        loading={isLoading}
      />
    </div>
  )
}
