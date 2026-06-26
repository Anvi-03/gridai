/**
 * GridPulse AI — DigitalTwinGrid.tsx
 *
 * 2D grid/card matrix representing all active smart meters.
 * Nodes dynamically color and pulse based on risk_zone / is_anomalous state.
 * Clicking a node opens a right-side slide-over drawer with full diagnostics.
 */

import { useState, useMemo } from 'react'
import {
  X,
  Zap,
  AlertTriangle,
  CheckCircle,
  Activity,
  Thermometer,
  ChevronRight,
} from 'lucide-react'
import { clsx } from 'clsx'
import type { TelemetryReading, StatsRow, MeterForecastItem, MeterNode } from '../types/grid'
import { formatINR, formatWatts, riskColor, riskBg } from '../lib/api'

interface DigitalTwinGridProps {
  stats: StatsRow[]
  telemetry: TelemetryReading[]
  forecastItems: MeterForecastItem[]
}

// ── Build merged MeterNode list ───────────────────────────────────────────────

function buildMeterNodes(
  stats: StatsRow[],
  telemetry: TelemetryReading[],
  forecastItems: MeterForecastItem[],
): MeterNode[] {
  // Build lookup maps once
  const latestByMeter = new Map<string, TelemetryReading>()
  for (const t of [...telemetry].reverse()) {
    latestByMeter.set(t.meter_id, t)
  }

  const forecastByMeter = new Map<string, MeterForecastItem>()
  for (const f of forecastItems) {
    forecastByMeter.set(f.meter_id, f)
  }

  return stats.map(s => {
    const t = latestByMeter.get(s.meter_id)
    const f = forecastByMeter.get(s.meter_id)

    return {
      meter_id:          s.meter_id,
      avg_voltage:       s.avg_voltage,
      avg_current:       s.avg_current,
      avg_power_factor:  s.avg_power_factor,
      total_readings:    s.total_readings,
      last_seen:         s.last_seen,
      is_anomalous:      t?.is_anomalous === true,
      anomaly_type:      t?.anomaly_type ?? null,
      anomaly_confidence: t?.anomaly_confidence ?? null,
      edge_flagged:      t?.edge_flagged ?? false,
      edge_confidence:   t?.edge_confidence ?? null,
      revenue_loss_inr:  t?.revenue_loss_inr ?? null,
      latest_voltage:    t?.voltage ?? null,
      latest_current:    t?.current ?? null,
      outage_risk_score: f?.outage_risk_score ?? 0,
      risk_zone:         f?.risk_zone ?? 'low',
      predicted_avg_w:   f?.predicted_avg_w ?? 0,
      predicted_peak_w:  f?.predicted_peak_w ?? 0,
    }
  })
}

// ── Node card styles by risk / anomaly state ──────────────────────────────────

function nodeCardClass(node: MeterNode): string {
  if (node.is_anomalous || node.risk_zone === 'critical') {
    return clsx(
      'border-red-500/50 bg-red-500/8 animate-pulse-red',
      'hover:border-red-400/70 hover:bg-red-500/12',
    )
  }
  if (node.risk_zone === 'high') {
    return clsx(
      'border-orange-500/40 bg-orange-500/6 animate-pulse-amber',
      'hover:border-orange-400/60',
    )
  }
  if (node.risk_zone === 'medium') {
    return clsx(
      'border-amber-500/35 bg-amber-500/5',
      'hover:border-amber-400/55',
    )
  }
  return clsx(
    'border-slate-700/40 bg-slate-800/30',
    'hover:border-emerald-500/40 hover:bg-emerald-500/5',
  )
}

// ── Slide-over drawer ─────────────────────────────────────────────────────────

interface DrawerProps {
  node: MeterNode
  onClose: () => void
}

function MeterDrawer({ node, onClose }: DrawerProps) {
  const realPowerW = node.latest_voltage && node.latest_current
    ? node.latest_voltage * node.latest_current * node.avg_power_factor
    : node.avg_voltage * node.avg_current * node.avg_power_factor

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 z-40 bg-black/40 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Drawer panel */}
      <div className="fixed inset-y-0 right-0 z-50 w-full max-w-sm animate-slide-in-right overflow-y-auto">
        <div className="min-h-full bg-slate-900/95 backdrop-blur-2xl border-l border-slate-700/50 p-6 flex flex-col gap-6">

          {/* Header */}
          <div className="flex items-start justify-between">
            <div>
              <h2 className="text-lg font-bold text-slate-100 font-mono">{node.meter_id}</h2>
              <div className="flex items-center gap-2 mt-1">
                <span className={clsx('badge', `badge-${node.risk_zone}`)}>
                  {node.risk_zone.toUpperCase()}
                </span>
                {node.edge_flagged && (
                  <span className="flex items-center gap-1 text-indigo-400 text-xs font-medium">
                    <Zap size={11} /> EDGE SCREENED
                  </span>
                )}
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-2 rounded-lg text-slate-400 hover:text-slate-100 hover:bg-slate-700/50 transition-colors"
            >
              <X size={18} />
            </button>
          </div>

          {/* Status banner */}
          {node.is_anomalous ? (
            <div className="flex items-center gap-3 p-3 rounded-xl bg-red-500/10 border border-red-500/30">
              <AlertTriangle size={18} className="text-red-400 shrink-0" />
              <div>
                <p className="text-red-300 text-sm font-semibold">Anomaly Detected</p>
                <p className="text-slate-400 text-xs mt-0.5 capitalize">
                  {node.anomaly_type?.replace(/_/g, ' ') ?? 'Unknown type'} ·{' '}
                  {node.anomaly_confidence != null
                    ? `${(node.anomaly_confidence * 100).toFixed(0)}% confidence`
                    : ''}
                </p>
              </div>
            </div>
          ) : (
            <div className="flex items-center gap-3 p-3 rounded-xl bg-emerald-500/8 border border-emerald-500/25">
              <CheckCircle size={18} className="text-emerald-400 shrink-0" />
              <p className="text-emerald-300 text-sm font-medium">Operating Normally</p>
            </div>
          )}

          {/* Electrical readings */}
          <section>
            <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">
              Electrical Readings
            </h3>
            <div className="grid grid-cols-2 gap-3">
              {[
                {
                  label: 'Voltage',
                  value: `${(node.latest_voltage ?? node.avg_voltage).toFixed(1)} V`,
                  sub: `Avg ${node.avg_voltage.toFixed(1)} V`,
                  ok: Math.abs((node.latest_voltage ?? node.avg_voltage) - 230) < 10,
                },
                {
                  label: 'Current',
                  value: `${(node.latest_current ?? node.avg_current).toFixed(1)} A`,
                  sub: `Avg ${node.avg_current.toFixed(1)} A`,
                  ok: (node.latest_current ?? node.avg_current) < 40,
                },
                {
                  label: 'Power Factor',
                  value: node.avg_power_factor.toFixed(3),
                  sub: node.avg_power_factor >= 0.8 ? 'Healthy' : 'Below threshold',
                  ok: node.avg_power_factor >= 0.8,
                },
                {
                  label: 'Real Power',
                  value: formatWatts(realPowerW),
                  sub: 'V × I × PF',
                  ok: true,
                },
              ].map(item => (
                <div key={item.label} className="glass-card p-3">
                  <p className="text-slate-500 text-xs mb-1">{item.label}</p>
                  <p className={clsx('text-base font-bold font-mono', item.ok ? 'text-slate-100' : 'text-amber-400')}>
                    {item.value}
                  </p>
                  <p className="text-slate-600 text-xs mt-0.5">{item.sub}</p>
                </div>
              ))}
            </div>
          </section>

          {/* Edge AI metadata */}
          {node.edge_flagged && (
            <section>
              <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">
                Edge AI Pre-Screening
              </h3>
              <div className="glass-card p-4 flex items-center gap-4">
                <Zap size={20} className="text-indigo-400 shrink-0" />
                <div>
                  <p className="text-slate-200 text-sm font-medium">Flagged before cloud transmission</p>
                  <p className="text-slate-400 text-xs mt-0.5">
                    Z-score confidence:{' '}
                    <span className="text-indigo-300 font-mono">
                      {node.edge_confidence != null
                        ? (node.edge_confidence * 100).toFixed(1) + '%'
                        : 'N/A'}
                    </span>
                  </p>
                </div>
              </div>
            </section>
          )}

          {/* Forecast risk */}
          <section>
            <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">
              24-Hour Forecast
            </h3>
            <div className="space-y-3">
              {/* Risk score bar */}
              <div>
                <div className="flex items-center justify-between text-xs text-slate-400 mb-1.5">
                  <span>Outage Risk</span>
                  <span className={clsx('font-mono font-semibold', riskColor(node.risk_zone))}>
                    {node.outage_risk_score}/100
                  </span>
                </div>
                <div className="h-2 rounded-full bg-slate-800 overflow-hidden">
                  <div
                    className={clsx('h-full rounded-full transition-all duration-700', {
                      'bg-red-500':     node.risk_zone === 'critical',
                      'bg-orange-500':  node.risk_zone === 'high',
                      'bg-amber-500':   node.risk_zone === 'medium',
                      'bg-emerald-500': node.risk_zone === 'low',
                    })}
                    style={{ width: `${node.outage_risk_score}%` }}
                  />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div className="glass-card p-3">
                  <p className="text-slate-500 text-xs">Predicted Avg</p>
                  <p className="text-slate-100 font-bold font-mono text-sm mt-1">
                    {formatWatts(node.predicted_avg_w)}
                  </p>
                </div>
                <div className="glass-card p-3">
                  <p className="text-slate-500 text-xs">Predicted Peak</p>
                  <p className="text-slate-100 font-bold font-mono text-sm mt-1">
                    {formatWatts(node.predicted_peak_w)}
                  </p>
                </div>
              </div>
            </div>
          </section>

          {/* Economic impact */}
          {node.revenue_loss_inr != null && node.revenue_loss_inr > 0 && (
            <section>
              <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">
                Economic Impact
              </h3>
              <div className="glass-card p-4 border border-amber-500/25">
                <p className="text-amber-300 font-bold text-xl font-mono">
                  {formatINR(node.revenue_loss_inr)}
                </p>
                <p className="text-slate-500 text-xs mt-1">
                  Estimated revenue loss from this anomaly event
                </p>
              </div>
            </section>
          )}

          {/* Metadata footer */}
          <div className="mt-auto pt-4 border-t border-slate-800 flex items-center justify-between text-xs text-slate-600">
            <span>{node.total_readings.toLocaleString()} total readings</span>
            <span>Last seen {new Date(node.last_seen).toLocaleTimeString()}</span>
          </div>
        </div>
      </div>
    </>
  )
}

// ── Main grid component ───────────────────────────────────────────────────────

export function DigitalTwinGrid({ stats, telemetry, forecastItems }: DigitalTwinGridProps) {
  const [selectedMeter, setSelectedMeter] = useState<MeterNode | null>(null)

  const nodes = useMemo(
    () => buildMeterNodes(stats, telemetry, forecastItems),
    [stats, telemetry, forecastItems],
  )

  if (nodes.length === 0) {
    return (
      <div className="glass-card p-12 flex flex-col items-center justify-center gap-3 text-slate-500">
        <Activity size={32} className="opacity-30" />
        <p className="text-sm">No meters reporting yet. Start the simulator to see data.</p>
      </div>
    )
  }

  return (
    <>
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-3">
        {nodes.map(node => (
          <button
            key={node.meter_id}
            onClick={() => setSelectedMeter(node)}
            className={clsx(
              'relative rounded-xl border p-3 text-left transition-all duration-300',
              'focus:outline-none focus:ring-2 focus:ring-indigo-500/40',
              nodeCardClass(node),
            )}
          >
            {/* Edge-flagged badge */}
            {node.edge_flagged && (
              <span className="absolute top-2 right-2 text-indigo-400" title="Edge AI pre-screened">
                <Zap size={11} />
              </span>
            )}

            {/* Status dot */}
            <span
              className={clsx(
                'absolute top-2 left-2 w-2 h-2 rounded-full',
                node.is_anomalous
                  ? 'bg-red-500 animate-blink'
                  : node.risk_zone === 'critical'
                  ? 'bg-red-500 animate-blink'
                  : node.risk_zone === 'high'
                  ? 'bg-orange-400'
                  : node.risk_zone === 'medium'
                  ? 'bg-amber-400'
                  : 'bg-emerald-400',
              )}
            />

            {/* Meter ID */}
            <p className="text-slate-300 font-mono text-xs font-medium mt-3 mb-2 leading-none truncate">
              {node.meter_id}
            </p>

            {/* Voltage */}
            <p className="text-slate-100 font-bold text-base leading-none">
              {(node.latest_voltage ?? node.avg_voltage).toFixed(0)}
              <span className="text-slate-500 font-normal text-xs ml-0.5">V</span>
            </p>

            {/* Risk score */}
            <div className="mt-2 flex items-center justify-between">
              <span className={clsx('text-xs font-mono font-semibold', riskColor(node.risk_zone))}>
                {node.outage_risk_score}
              </span>
              <ChevronRight size={10} className="text-slate-700" />
            </div>

            {/* Mini risk bar */}
            <div className="mt-1.5 h-0.5 rounded-full bg-slate-800 overflow-hidden">
              <div
                className={clsx('h-full rounded-full', {
                  'bg-red-500':     node.risk_zone === 'critical' || node.is_anomalous,
                  'bg-orange-400':  node.risk_zone === 'high',
                  'bg-amber-400':   node.risk_zone === 'medium',
                  'bg-emerald-400': node.risk_zone === 'low' && !node.is_anomalous,
                })}
                style={{ width: `${Math.max(4, node.outage_risk_score)}%` }}
              />
            </div>
          </button>
        ))}
      </div>

      {/* Slide-over drawer */}
      {selectedMeter && (
        <MeterDrawer node={selectedMeter} onClose={() => setSelectedMeter(null)} />
      )}
    </>
  )
}
