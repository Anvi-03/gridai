/**
 * GridPulse AI — App.tsx
 *
 * Master dashboard layout with 2-second polling loop.
 * All child components receive typed props from centralised state.
 *
 * Polling architecture:
 *   - fetchStats()    → StatsRow[]       (lightweight, every tick)
 *   - fetchForecast() → ForecastReport   (heavier; every tick but cached by backend)
 *   - fetchTelemetry()→ TelemetryReading[] (last 60 records)
 *   - fetchHealth()   → HealthResponse   (once on mount, then every 10s)
 *
 * Error resilience:
 *   - All fetch helpers return typed fallback data on any error.
 *   - Connection status banner shows LIVE / DEGRADED / OFFLINE.
 */

import { useEffect, useState, useRef, useCallback } from 'react'
import { Activity, RefreshCw, Wifi, WifiOff, Zap } from 'lucide-react'
import { clsx } from 'clsx'
import type { StatsRow, TelemetryReading, ForecastReport, HealthResponse } from './types/grid'
import { fetchStats, fetchTelemetry, fetchForecast, fetchHealth } from './lib/api'
import { MetricCards }      from './components/MetricCards'
import { DigitalTwinTopology } from './components/DigitalTwinTopology'
import { ForecastChart }    from './components/ForecastChart'
import { GridCopilot }      from './components/GridCopilot'

// ── Connection status type ────────────────────────────────────────────────────

type ConnectionStatus = 'live' | 'degraded' | 'offline'

// ── Main App ──────────────────────────────────────────────────────────────────

export default function App() {
  const [stats, setStats]         = useState<StatsRow[]>([])
  const [telemetry, setTelemetry] = useState<TelemetryReading[]>([])
  const [forecast, setForecast]   = useState<ForecastReport | null>(null)
  const [health, setHealth]       = useState<HealthResponse | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [lastUpdated, setLastUpdated]     = useState<Date | null>(null)
  const [connStatus, setConnStatus]       = useState<ConnectionStatus>('live')
  const [consecutiveErrors, setConsecutiveErrors] = useState(0)
  const healthTickRef = useRef(0)

  // ── Core polling function ────────────────────────────────────────────────────

  const poll = useCallback(async () => {
    try {
      // Run all three fetches in parallel
      const [newStats, newForecast, newTelemetry] = await Promise.all([
        fetchStats(),
        fetchForecast(),
        fetchTelemetry(60),
      ])

      setStats(newStats)
      setForecast(newForecast)
      setTelemetry(newTelemetry)
      setLastUpdated(new Date())
      setConsecutiveErrors(0)
      setConnStatus('live')
    } catch {
      // Shouldn't reach here — api.ts absorbs all errors into fallbacks
      setConsecutiveErrors(c => {
        const next = c + 1
        setConnStatus(next >= 5 ? 'offline' : 'degraded')
        return next
      })
    } finally {
      if (isLoading) setIsLoading(false)
    }
  }, [isLoading])

  // ── Polling interval (2 s) ────────────────────────────────────────────────────

  useEffect(() => {
    poll() // immediate first call
    const interval = setInterval(poll, 2_000)
    return () => clearInterval(interval)
  }, [poll])

  // ── Health check (once on mount, then every 10 s) ────────────────────────────

  useEffect(() => {
    const checkHealth = async () => {
      const h = await fetchHealth()
      setHealth(h)
      // Derive connection status from health.status
      if (h.status === 'offline') setConnStatus('offline')
    }
    checkHealth()
    const interval = setInterval(() => {
      healthTickRef.current++
      checkHealth()
    }, 10_000)
    return () => clearInterval(interval)
  }, [])

  // ── Derived data for child components ─────────────────────────────────────────

  const forecastMatrix = forecast?.outage_probability_matrix ?? []
  const criticalCount  = forecast?.fleet_summary.critical_count ?? 0
  const highCount      = forecast?.fleet_summary.high_risk_count ?? 0

  // ── Render ───────────────────────────────────────────────────────────────────

  return (
    <div className="min-h-screen bg-transparent text-slate-100">

      {/* ── Top navigation bar ──────────────────────────────────────────────── */}
      <header className="sticky top-0 z-30 border-b border-slate-800/60 bg-slate-950/80 backdrop-blur-xl">
        <div className="max-w-screen-2xl mx-auto px-6 h-14 flex items-center justify-between">

          {/* Brand */}
          <div className="flex items-center gap-3">
            <div className="p-1.5 rounded-lg bg-indigo-600/20 border border-indigo-500/30">
              <Zap size={16} className="text-indigo-400" />
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-base font-bold text-slate-100 tracking-tight">GridPulse</span>
              <span className="text-base font-light text-indigo-400">AI</span>
              <span className="hidden sm:inline text-xs text-slate-600 font-mono ml-1">
                v{health?.version ?? '1.0.0'}
              </span>
            </div>

            {/* Critical alerts pill */}
            {criticalCount > 0 && (
              <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-red-500/15 border border-red-500/30 text-red-400 text-xs font-medium animate-pulse-red">
                <span className="w-1.5 h-1.5 rounded-full bg-red-500 animate-blink" />
                {criticalCount} CRITICAL · {highCount} HIGH
              </div>
            )}
          </div>

          {/* Right controls */}
          <div className="flex items-center gap-4">

            {/* Last updated */}
            {lastUpdated && (
              <span className="hidden md:flex items-center gap-1.5 text-xs text-slate-500 font-mono">
                <RefreshCw size={10} className={clsx(connStatus === 'live' ? 'text-emerald-400' : 'text-amber-400')} />
                {lastUpdated.toLocaleTimeString()}
              </span>
            )}

            {/* Connection status badge */}
            <div
              className={clsx(
                'flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border',
                connStatus === 'live'
                  ? 'bg-emerald-500/10 border-emerald-500/25 text-emerald-400'
                  : connStatus === 'degraded'
                  ? 'bg-amber-500/10 border-amber-500/25 text-amber-400'
                  : 'bg-red-500/10 border-red-500/25 text-red-400',
              )}
            >
              {connStatus === 'offline' ? <WifiOff size={11} /> : <Wifi size={11} />}
              <span className="uppercase tracking-wider">{connStatus}</span>
            </div>

            {/* DB status */}
            {health && (
              <span className={clsx(
                'hidden lg:inline text-xs font-mono px-2 py-1 rounded-lg border',
                health.database === 'healthy'
                  ? 'text-emerald-400 border-emerald-500/20 bg-emerald-500/5'
                  : 'text-red-400 border-red-500/20 bg-red-500/5',
              )}>
                DB: {health.database}
              </span>
            )}
          </div>
        </div>
      </header>

      {/* ── Main content ─────────────────────────────────────────────────────── */}
      <main className="max-w-screen-2xl mx-auto px-6 py-6 space-y-6">

        {/* Loading skeleton overlay — shown only on first load */}
        {isLoading && stats.length === 0 && (
          <div className="absolute inset-0 z-20 flex items-center justify-center bg-slate-950/70 backdrop-blur-sm">
            <div className="flex flex-col items-center gap-3">
              <Activity size={28} className="text-indigo-400 animate-spin" />
              <p className="text-slate-400 text-sm font-mono">Connecting to GridPulse backend…</p>
            </div>
          </div>
        )}

        {/* ── Section: KPI Metric Cards ─────────────────────────────────────── */}
        <section aria-label="Key performance indicators">
          <MetricCards
            stats={stats}
            telemetry={telemetry}
            forecast={forecast}
            isLoading={isLoading}
          />
        </section>

        {/* ── Section: Digital Twin Grid + Forecast Chart (side by side on xl) ── */}
        <section
          aria-label="Digital twin grid and forecast"
          className="grid grid-cols-1 xl:grid-cols-3 gap-6"
        >
          {/* Digital Twin Grid — takes 2/3 width */}
          <div className="xl:col-span-2 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Activity size={14} className="text-indigo-400" />
                <h2 className="text-sm font-semibold text-slate-300">
                  Digital Twin Grid
                </h2>
                <span className="text-slate-600 text-xs font-mono">
                  {stats.length} meters active
                </span>
              </div>
              <div className="flex items-center gap-3 text-xs text-slate-600">
                {/* Legend */}
                {[
                  { label: 'Normal',   color: 'bg-emerald-400' },
                  { label: 'Medium',   color: 'bg-amber-400'   },
                  { label: 'High',     color: 'bg-orange-400'  },
                  { label: 'Critical', color: 'bg-red-500'     },
                ].map(l => (
                  <div key={l.label} className="flex items-center gap-1">
                    <span className={clsx('w-2 h-2 rounded-full', l.color)} />
                    <span>{l.label}</span>
                  </div>
                ))}
              </div>
            </div>

            <DigitalTwinTopology
              stats={stats}
              telemetry={telemetry}
              forecastItems={forecastMatrix}
            />
          </div>

          {/* Fleet Risk Breakdown sidebar — 1/3 width */}
          <div className="xl:col-span-1 space-y-4">
            {/* Risk distribution card */}
            <div className="glass-card p-5">
              <h3 className="text-sm font-semibold text-slate-300 mb-4">Fleet Risk Distribution</h3>
              {forecast ? (
                <div className="space-y-3">
                  {[
                    { label: 'Low Risk',   count: forecast.fleet_summary.low_risk_count,    color: 'bg-emerald-500', textColor: 'text-emerald-400' },
                    { label: 'Medium',     count: forecast.fleet_summary.medium_risk_count,  color: 'bg-amber-500',   textColor: 'text-amber-400'   },
                    { label: 'High',       count: forecast.fleet_summary.high_risk_count,    color: 'bg-orange-500',  textColor: 'text-orange-400'  },
                    { label: 'Critical',   count: forecast.fleet_summary.critical_count,     color: 'bg-red-500',     textColor: 'text-red-400'     },
                  ].map(item => {
                    const total = forecast.total_meters_active || 1
                    const pct   = Math.round((item.count / total) * 100)
                    return (
                      <div key={item.label}>
                        <div className="flex items-center justify-between text-xs mb-1">
                          <span className="text-slate-400">{item.label}</span>
                          <span className={clsx('font-mono font-semibold', item.textColor)}>
                            {item.count} <span className="text-slate-600">({pct}%)</span>
                          </span>
                        </div>
                        <div className="h-1.5 rounded-full bg-slate-800 overflow-hidden">
                          <div
                            className={clsx('h-full rounded-full transition-all duration-700', item.color)}
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                      </div>
                    )
                  })}
                </div>
              ) : (
                <div className="space-y-3">
                  {[1, 2, 3, 4].map(i => <div key={i} className="skeleton h-8" />)}
                </div>
              )}
            </div>

            {/* High-risk zones */}
            {forecast && forecast.high_risk_zones.length > 0 && (
              <div className="glass-card p-5">
                <h3 className="text-sm font-semibold text-slate-300 mb-3">High-Risk Zones</h3>
                <div className="space-y-2">
                  {forecast.high_risk_zones.slice(0, 5).map(zone => (
                    <div
                      key={zone.meter_id}
                      className={clsx(
                        'flex items-center justify-between px-3 py-2 rounded-lg border text-xs',
                        zone.risk_zone === 'critical'
                          ? 'bg-red-500/8 border-red-500/25 text-red-300'
                          : 'bg-orange-500/8 border-orange-500/25 text-orange-300',
                      )}
                    >
                      <span className="font-mono">{zone.meter_id}</span>
                      <div className="flex items-center gap-2">
                        <span className="font-bold">{zone.outage_risk_score}/100</span>
                        <span className="uppercase text-[10px] opacity-60">{zone.risk_zone}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Systemic outage probability gauge */}
            {forecast && (
              <div className="glass-card p-5">
                <h3 className="text-sm font-semibold text-slate-300 mb-3">
                  Systemic Outage Probability
                </h3>
                <div className="flex items-end justify-between mb-2">
                  <span className={clsx(
                    'text-4xl font-bold font-mono',
                    forecast.fleet_summary.systemic_outage_probability >= 0.7 ? 'text-red-400'
                    : forecast.fleet_summary.systemic_outage_probability >= 0.4 ? 'text-amber-400'
                    : 'text-emerald-400',
                  )}>
                    {(forecast.fleet_summary.systemic_outage_probability * 100).toFixed(1)}
                    <span className="text-xl">%</span>
                  </span>
                  <span className="text-xs text-slate-500 font-mono pb-1">fleet-wide</span>
                </div>
                <div className="h-2.5 rounded-full bg-slate-800 overflow-hidden">
                  <div
                    className={clsx('h-full rounded-full transition-all duration-700', {
                      'bg-gradient-to-r from-red-600 to-red-400':     forecast.fleet_summary.systemic_outage_probability >= 0.7,
                      'bg-gradient-to-r from-amber-600 to-amber-400': forecast.fleet_summary.systemic_outage_probability >= 0.4,
                      'bg-gradient-to-r from-emerald-600 to-emerald-400': forecast.fleet_summary.systemic_outage_probability < 0.4,
                    })}
                    style={{ width: `${Math.min(100, forecast.fleet_summary.systemic_outage_probability * 100)}%` }}
                  />
                </div>
                <p className="text-slate-600 text-xs mt-2 font-mono">
                  Updated {new Date(forecast.generated_at).toLocaleTimeString()}
                </p>
              </div>
            )}
          </div>
        </section>

        {/* ── Section: Forecast Chart ──────────────────────────────────────────── */}
        <section aria-label="Predictive load forecast">
          <ForecastChart telemetry={telemetry} forecast={forecast} />
        </section>

        {/* ── Footer ──────────────────────────────────────────────────────────── */}
        <footer className="text-center text-xs text-slate-700 font-mono pb-4">
          GridPulse AI · Digital Twin Dashboard · Feature 7
          <span className="mx-2 text-slate-800">·</span>
          Backend: localhost:8000
          <span className="mx-2 text-slate-800">·</span>
          Polling every 2s
        </footer>
      </main>

      {/* ── Floating GenAI Copilot ───────────────────────────────────────────── */}
      <GridCopilot />
    </div>
  )
}
