/**
 * GridPulse AI — DigitalTwinTopology.tsx
 *
 * Visualise smart meter assets as a hierarchical topological network graph
 * built on top of `reactflow`.
 *
 * Levels:
 *   1. Root Node: Substation Delta-01
 *   2. Intermediate Nodes: Transformer Alpha & Transformer Beta
 *   3. Leaf Nodes: Smart meters from active stats array
 *
 * Support simulation overrides to test layout resilience under simulated scenarios.
 */

import { useState, useMemo } from 'react'
import ReactFlow, { Background, Controls } from 'reactflow'
import 'reactflow/dist/style.css'
import {
  X,
  Zap,
  AlertTriangle,
  CheckCircle,
  Activity,
} from 'lucide-react'
import { clsx } from 'clsx'
import type { TelemetryReading, StatsRow, MeterForecastItem, MeterNode, SimulationResponse } from '../types/grid'
import { formatINR, formatWatts, riskColor } from '../lib/api'

interface DigitalTwinTopologyProps {
  stats: StatsRow[]
  telemetry: TelemetryReading[]
  forecastItems: MeterForecastItem[]
  onSelectNode?: (nodeId: string) => void
  simulationOverride?: SimulationResponse | null
}

// ── Merge stats, telemetry, and forecast data ───────────────────────────────

function buildMeterNodes(
  stats: StatsRow[],
  telemetry: TelemetryReading[],
  forecastItems: MeterForecastItem[],
): MeterNode[] {
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

// ── Lateral inspection panel drawer ──────────────────────────────────────────

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
                  <span className="flex items-center gap-1 text-indigo-400 text-xs font-medium font-mono">
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
                  ok: Math.abs((node.latest_voltage ?? node.avg_voltage) - 230) < 15,
                },
                {
                  label: 'Current',
                  value: `${(node.latest_current ?? node.avg_current).toFixed(1)} A`,
                  sub: `Avg ${node.avg_current.toFixed(1)} A`,
                  ok: (node.latest_current ?? node.avg_current) < 60,
                },
                {
                  label: 'Power Factor',
                  value: node.avg_power_factor.toFixed(3),
                  sub: node.avg_power_factor >= 0.75 ? 'Healthy' : 'Below threshold',
                  ok: node.avg_power_factor >= 0.75,
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

// ── Main topology graph component ─────────────────────────────────────────────

export function DigitalTwinTopology({
  stats,
  telemetry,
  forecastItems,
  onSelectNode,
  simulationOverride,
}: DigitalTwinTopologyProps) {
  const [selectedMeter, setSelectedMeter] = useState<MeterNode | null>(null)

  const meterNodesData = useMemo(
    () => buildMeterNodes(stats, telemetry, forecastItems),
    [stats, telemetry, forecastItems],
  )

  // ── Construct React Flow Nodes and Edges ─────────────────────────────────────
  const { flowNodes, flowEdges } = useMemo(() => {
    if (meterNodesData.length === 0) {
      return { flowNodes: [], flowEdges: [] }
    }

    const nodes: any[] = []
    const edges: any[] = []

    // 1. Root Level: Substation Node
    nodes.push({
      id: 'substation',
      type: 'input',
      position: { x: 420, y: 20 },
      style: { background: 'none', border: 'none', padding: 0 },
      data: {
        label: (
          <div className="flex items-center gap-3 px-4 py-3 rounded-2xl border border-indigo-500/60 bg-indigo-950/80 text-slate-100 shadow-2xl backdrop-blur-md">
            <div className="p-2 rounded-xl bg-indigo-500/20 text-indigo-400">
              <Zap size={20} className="animate-pulse" />
            </div>
            <div className="text-left">
              <span className="text-[10px] text-indigo-400 uppercase tracking-widest font-semibold block leading-none">Substation</span>
              <span className="text-sm font-bold font-mono">DELTA-01</span>
            </div>
          </div>
        )
      }
    })

    // 2. Intermediate Level: Transformer Nodes
    nodes.push({
      id: 'transformer-alpha',
      type: 'default',
      position: { x: 220, y: 140 },
      style: { background: 'none', border: 'none', padding: 0 },
      data: {
        label: (
          <div className="flex items-center gap-3 px-4 py-3 rounded-2xl border border-emerald-500 bg-slate-900/90 text-slate-100 shadow-2xl backdrop-blur-md">
            <div className="p-2 rounded-xl bg-emerald-500/25 text-emerald-400">
              <Activity size={16} />
            </div>
            <div className="text-left">
              <span className="text-[10px] text-emerald-400 uppercase tracking-widest font-semibold block leading-none">Transformer</span>
              <span className="text-sm font-bold font-mono">ALPHA</span>
            </div>
          </div>
        )
      }
    })

    nodes.push({
      id: 'transformer-beta',
      type: 'default',
      position: { x: 620, y: 140 },
      style: { background: 'none', border: 'none', padding: 0 },
      data: {
        label: (
          <div className="flex items-center gap-3 px-4 py-3 rounded-2xl border border-emerald-500 bg-slate-900/90 text-slate-100 shadow-2xl backdrop-blur-md">
            <div className="p-2 rounded-xl bg-emerald-500/25 text-emerald-400">
              <Activity size={16} />
            </div>
            <div className="text-left">
              <span className="text-[10px] text-emerald-400 uppercase tracking-widest font-semibold block leading-none">Transformer</span>
              <span className="text-sm font-bold font-mono">BETA</span>
            </div>
          </div>
        )
      }
    })

    // Substation to Transformer Edges
    const animateSubAlpha = simulationOverride?.target_transformer_id === 'transformer-alpha'
    const animateSubBeta  = simulationOverride?.target_transformer_id === 'transformer-beta'

    edges.push({
      id: 'e-sub-alpha',
      source: 'substation',
      target: 'transformer-alpha',
      animated: animateSubAlpha,
      style: animateSubAlpha
        ? { stroke: '#ef4444', strokeWidth: 3 }
        : { stroke: '#334155', strokeWidth: 2 },
    })

    edges.push({
      id: 'e-sub-beta',
      source: 'substation',
      target: 'transformer-beta',
      animated: animateSubBeta,
      style: animateSubBeta
        ? { stroke: '#ef4444', strokeWidth: 3 }
        : { stroke: '#334155', strokeWidth: 2 },
    })

    // 3. Leaf Level: Smart Meters
    const alphaMeters = meterNodesData.filter((_, idx) => idx % 2 === 0)
    const betaMeters  = meterNodesData.filter((_, idx) => idx % 2 !== 0)

    // Layout helper for alpha meters
    const alphaWidth = 360
    const alphaStep = alphaMeters.length > 1 ? alphaWidth / (alphaMeters.length - 1) : alphaWidth
    alphaMeters.forEach((node, idx) => {
      const isMeterAffected = simulationOverride?.affected_meter_ids.includes(node.meter_id) ?? false
      const isCritical = simulationOverride 
        ? isMeterAffected 
        : (node.is_anomalous || node.edge_flagged)

      const displayVoltage = isMeterAffected && simulationOverride
        ? simulationOverride.simulated_telemetry.voltage
        : (node.latest_voltage ?? node.avg_voltage)

      const displayRisk = isMeterAffected && simulationOverride
        ? simulationOverride.failure_probability
        : node.outage_risk_score

      const displayRiskZone = isMeterAffected && simulationOverride
        ? (simulationOverride.failure_probability >= 70 ? 'critical'
           : simulationOverride.failure_probability >= 50 ? 'high'
           : simulationOverride.failure_probability >= 30 ? 'medium' : 'low')
        : node.risk_zone

      const x = 20 + idx * alphaStep
      const y = 300 + (idx % 2) * 50 // subtle staggering prevents overlap

      nodes.push({
        id: node.meter_id,
        type: 'output',
        position: { x, y },
        style: { background: 'none', border: 'none', padding: 0 },
        data: {
          label: (
            <div className={clsx(
              "flex flex-col p-3 rounded-2xl border backdrop-blur-md shadow-xl transition-all duration-300 relative min-w-[140px]",
              isCritical
                ? "border-red-500 bg-red-950/70 text-red-200 animate-pulse-red"
                : "border-slate-700 bg-slate-900/90 text-slate-200 hover:border-emerald-500/60 hover:bg-emerald-950/10"
            )}>
              {isCritical && (
                <span className="absolute -top-3 left-1/2 -translate-x-1/2 px-2 py-0.5 rounded-full bg-red-500 text-white text-[9px] font-bold tracking-wider uppercase animate-blink whitespace-nowrap shadow-lg">
                  {simulationOverride ? "⚡ Active Scenario" : (node.edge_flagged ? "⚡ Edge Flagged" : "❌ Theft Alert")}
                </span>
              )}
              <div className="flex items-center justify-between mb-1.5 font-mono">
                <span className="text-[10px] text-slate-400 font-semibold">{node.meter_id}</span>
                <span className={clsx("w-1.5 h-1.5 rounded-full",
                  isCritical || displayRiskZone === 'critical'
                    ? "bg-red-500 animate-blink"
                    : displayRiskZone === 'high'
                    ? "bg-orange-500"
                    : displayRiskZone === 'medium'
                    ? "bg-amber-400"
                    : "bg-emerald-400"
                )} />
              </div>
              <div className="text-base font-bold font-mono text-slate-100">
                {displayVoltage.toFixed(0)}
                <span className="text-slate-500 text-xs font-normal ml-0.5">V</span>
              </div>
              <div className="flex items-center justify-between mt-2 pt-1 border-t border-slate-800 text-[10px]">
                <span className="text-slate-500">Risk Score</span>
                <span className={clsx("font-semibold font-mono",
                  displayRiskZone === 'critical' ? "text-red-400"
                  : displayRiskZone === 'high' ? "text-orange-400"
                  : displayRiskZone === 'medium' ? "text-amber-400"
                  : "text-emerald-400"
                )}>{displayRisk}</span>
              </div>
            </div>
          )
        }
      })

      edges.push({
        id: `e-alpha-${node.meter_id}`,
        source: 'transformer-alpha',
        target: node.meter_id,
        animated: isCritical,
        style: isCritical
          ? { stroke: '#ef4444', strokeWidth: 3 }
          : { stroke: '#334155', strokeWidth: 1.5 },
      })
    })

    // Layout helper for beta meters
    const betaWidth = 360
    const betaStep = betaMeters.length > 1 ? betaWidth / (betaMeters.length - 1) : betaWidth
    betaMeters.forEach((node, idx) => {
      const isMeterAffected = simulationOverride?.affected_meter_ids.includes(node.meter_id) ?? false
      const isCritical = simulationOverride 
        ? isMeterAffected 
        : (node.is_anomalous || node.edge_flagged)

      const displayVoltage = isMeterAffected && simulationOverride
        ? simulationOverride.simulated_telemetry.voltage
        : (node.latest_voltage ?? node.avg_voltage)

      const displayRisk = isMeterAffected && simulationOverride
        ? simulationOverride.failure_probability
        : node.outage_risk_score

      const displayRiskZone = isMeterAffected && simulationOverride
        ? (simulationOverride.failure_probability >= 70 ? 'critical'
           : simulationOverride.failure_probability >= 50 ? 'high'
           : simulationOverride.failure_probability >= 30 ? 'medium' : 'low')
        : node.risk_zone

      const x = 460 + idx * betaStep
      const y = 300 + (idx % 2) * 50

      nodes.push({
        id: node.meter_id,
        type: 'output',
        position: { x, y },
        style: { background: 'none', border: 'none', padding: 0 },
        data: {
          label: (
            <div className={clsx(
              "flex flex-col p-3 rounded-2xl border backdrop-blur-md shadow-xl transition-all duration-300 relative min-w-[140px]",
              isCritical
                ? "border-red-500 bg-red-950/70 text-red-200 animate-pulse-red"
                : "border-slate-700 bg-slate-900/90 text-slate-200 hover:border-emerald-500/60 hover:bg-emerald-950/10"
            )}>
              {isCritical && (
                <span className="absolute -top-3 left-1/2 -translate-x-1/2 px-2 py-0.5 rounded-full bg-red-500 text-white text-[9px] font-bold tracking-wider uppercase animate-blink whitespace-nowrap shadow-lg">
                  {simulationOverride ? "⚡ Active Scenario" : (node.edge_flagged ? "⚡ Edge Flagged" : "❌ Theft Alert")}
                </span>
              )}
              <div className="flex items-center justify-between mb-1.5 font-mono">
                <span className="text-[10px] text-slate-400 font-semibold">{node.meter_id}</span>
                <span className={clsx("w-1.5 h-1.5 rounded-full",
                  isCritical || displayRiskZone === 'critical'
                    ? "bg-red-500 animate-blink"
                    : displayRiskZone === 'high'
                    ? "bg-orange-500"
                    : displayRiskZone === 'medium'
                    ? "bg-amber-400"
                    : "bg-emerald-400"
                )} />
              </div>
              <div className="text-base font-bold font-mono text-slate-100">
                {displayVoltage.toFixed(0)}
                <span className="text-slate-500 text-xs font-normal ml-0.5">V</span>
              </div>
              <div className="flex items-center justify-between mt-2 pt-1 border-t border-slate-800 text-[10px]">
                <span className="text-slate-500">Risk Score</span>
                <span className={clsx("font-semibold font-mono",
                  displayRiskZone === 'critical' ? "text-red-400"
                  : displayRiskZone === 'high' ? "text-orange-400"
                  : displayRiskZone === 'medium' ? "text-amber-400"
                  : "text-emerald-400"
                )}>{displayRisk}</span>
              </div>
            </div>
          )
        }
      })

      edges.push({
        id: `e-beta-${node.meter_id}`,
        source: 'transformer-beta',
        target: node.meter_id,
        animated: isCritical,
        style: isCritical
          ? { stroke: '#ef4444', strokeWidth: 3 }
          : { stroke: '#334155', strokeWidth: 1.5 },
      })
    })

    return { flowNodes: nodes, flowEdges: edges }
  }, [meterNodesData, simulationOverride])

  // ── Node click interaction handler ──────────────────────────────────────────
  const handleNodeClick = (_event: React.MouseEvent, node: any) => {
    if (onSelectNode) {
      onSelectNode(node.id)
    }

    const clickedMeter = meterNodesData.find(m => m.meter_id === node.id)
    if (clickedMeter) {
      const isMeterAffected = simulationOverride?.affected_meter_ids.includes(clickedMeter.meter_id) ?? false
      if (isMeterAffected && simulationOverride) {
        setSelectedMeter({
          ...clickedMeter,
          latest_voltage: simulationOverride.simulated_telemetry.voltage,
          latest_current: simulationOverride.simulated_telemetry.current,
          avg_power_factor: simulationOverride.simulated_telemetry.power_factor,
          outage_risk_score: simulationOverride.failure_probability,
          risk_zone: (simulationOverride.failure_probability >= 70 ? 'critical'
                      : simulationOverride.failure_probability >= 50 ? 'high'
                      : simulationOverride.failure_probability >= 30 ? 'medium' : 'low'),
          is_anomalous: true,
          anomaly_type: simulationOverride.scenario,
          revenue_loss_inr: 0,
        })
      } else {
        setSelectedMeter(clickedMeter)
      }
    }
  }

  if (meterNodesData.length === 0) {
    return (
      <div className="glass-card p-12 flex flex-col items-center justify-center gap-3 text-slate-500">
        <Activity size={32} className="opacity-30" />
        <p className="text-sm font-mono">No meters reporting yet. Start the simulator to see data.</p>
      </div>
    )
  }

  return (
    <>
      <div className="glass-card p-4 relative overflow-hidden bg-slate-900/20" style={{ height: '550px' }}>
        <ReactFlow
          nodes={flowNodes}
          edges={flowEdges}
          onNodeClick={handleNodeClick}
          fitView
          nodesDraggable={true}
          nodesConnectable={false}
          zoomOnScroll={false}
          panOnDrag={true}
        >
          <Background color="#6366f1" gap={16} size={1} style={{ opacity: 0.06 }} />
          <Controls className="bg-slate-900 border border-slate-700 text-slate-100" />
        </ReactFlow>
      </div>

      {/* Slide-over inspector panel */}
      {selectedMeter && (
        <MeterDrawer node={selectedMeter} onClose={() => setSelectedMeter(null)} />
      )}
    </>
  )
}
