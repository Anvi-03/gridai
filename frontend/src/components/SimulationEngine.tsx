/**
 * GridPulse AI — SimulationEngine.tsx
 *
 * A modular glassmorphism control panel with buttons to trigger grid scenarios.
 * Triggers: POST /api/v1/simulation/trigger
 * Calls onSimulationTriggered with the resulting SimulationResponse on success.
 */

import { useState } from 'react'
import { Play, RotateCcw } from 'lucide-react'
import { postTriggerSimulation } from '../lib/api'
import type { SimulationResponse } from '../types/grid'
import { clsx } from 'clsx'

interface SimulationEngineProps {
  onSimulationTriggered: (data: SimulationResponse) => void
  onResetSimulation: () => void
  activeScenario: string | null
  onUnauthorized?: () => void
}

const SCENARIOS = [
  { key: 'heatwave',            label: 'Heatwave',            emoji: '🔥' },
  { key: 'transformer_failure', label: 'Transformer Failure', emoji: '⚡' },
  { key: 'ev_surge',            label: 'EV Surge',            emoji: '🚗' },
  { key: 'heavy_rain',          label: 'Heavy Rain',          emoji: '🌧️' },
  { key: 'industrial_peak',     label: 'Industrial Peak',     emoji: '🏭' },
]

export function SimulationEngine({
  onSimulationTriggered,
  onResetSimulation,
  activeScenario,
  onUnauthorized,
}: SimulationEngineProps) {
  const [loadingScenario, setLoadingScenario] = useState<string | null>(null)

  const handleTrigger = async (scenario: string) => {
    setLoadingScenario(scenario)
    try {
      const { data, unauthorized } = await postTriggerSimulation(scenario)
      if (unauthorized) {
        onUnauthorized?.()
        return
      }
      if (data) {
        onSimulationTriggered(data)
      }
    } catch (error) {
      console.error('Failed to trigger simulation:', error)
    } finally {
      setLoadingScenario(null)
    }
  }

  return (
    <div className="glass-card p-5 flex flex-col gap-4 relative overflow-hidden">
      {/* Glow orb */}
      <div className="absolute -top-10 -left-10 w-28 h-28 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none" />

      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold text-slate-200">Scenario Simulation</h3>
          <p className="text-slate-500 text-xs mt-0.5 font-mono">
            Deform telemetry dynamically to test layout resilience
          </p>
        </div>

        {activeScenario && (
          <button
            onClick={onResetSimulation}
            className="flex items-center gap-1 text-xs px-2.5 py-1.5 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 hover:bg-red-500/20 transition-all font-mono"
          >
            <RotateCcw size={12} />
            <span>Reset [X]</span>
          </button>
        )}
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-5 gap-3">
        {SCENARIOS.map(sc => {
          const isActive = activeScenario === sc.key
          const isLoading = loadingScenario === sc.key

          return (
            <button
              key={sc.key}
              onClick={() => handleTrigger(sc.key)}
              disabled={loadingScenario !== null}
              className={clsx(
                'flex flex-col items-center justify-center p-3 rounded-xl border text-center transition-all duration-300 relative group font-mono text-xs',
                isActive
                  ? 'border-indigo-500 bg-indigo-500/15 text-indigo-300 shadow-lg'
                  : 'border-slate-800 bg-slate-950/40 text-slate-400 hover:border-slate-700 hover:bg-slate-900/50 hover:text-slate-200',
                loadingScenario !== null && !isLoading && 'opacity-50 cursor-not-allowed'
              )}
            >
              {/* Animated play overlay on hover */}
              {!isActive && !isLoading && (
                <span className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity text-indigo-400">
                  <Play size={10} fill="currentColor" />
                </span>
              )}

              <span className="text-2xl mb-1.5" role="img" aria-label={sc.label}>
                {sc.emoji}
              </span>
              <span className="font-semibold leading-tight">{sc.label}</span>

              {isLoading && (
                <span className="absolute inset-0 flex items-center justify-center bg-slate-950/80 rounded-xl">
                  <span className="w-4 h-4 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin" />
                </span>
              )}
            </button>
          )
        })}
      </div>
    </div>
  )
}
