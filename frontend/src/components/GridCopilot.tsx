/**
 * GridPulse AI — GridCopilot.tsx
 *
 * Floating terminal-style chat panel in the bottom-right corner.
 * Overhauled to present rich, visual, card-based diagnostics with action
 * triggers, copy helper, severity highlights, and formatting boundaries.
 */

import { useState, useRef, useEffect, useCallback } from 'react'
import { Terminal, Send, X, Minimize2, Maximize2, Bot, User, AlertCircle, Copy, Check } from 'lucide-react'
import { clsx } from 'clsx'
import { postCopilotQuery } from '../lib/api'

interface Message {
  id: string
  role: 'user' | 'assistant' | 'error'
  text: string
  timestamp: Date
  model?: string
  tokens?: number | null
}

const SUGGESTED_QUERIES = [
  'Which meter has the highest outage risk right now?',
  'Summarise all active anomalies and their economic impact.',
  'What is the fleet-level systemic outage probability?',
  'Which meters are edge-screened and what types of anomalies were detected?',
]

let msgCounter = 0
function nextId() { return `msg-${++msgCounter}` }

// ── Rich diagnosis report formatter ──────────────────────────────────────────

function RichMessage({ text, onCopy }: { text: string; onCopy: () => void }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    onCopy()
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  if (!text.includes('### 🔍 Diagnosis Report')) {
    return <p className="whitespace-pre-wrap break-words">{text}</p>
  }

  // Extract sections dynamically
  const rootCauseMatch = text.match(/\*\*Root Cause\*\*\s*([\s\S]*?)(?=\*\*Estimated Loss\*\*|\*\*Recommendation\*\*|$)/)
  const estimatedLossMatch = text.match(/\*\*Estimated Loss\*\*\s*([\s\S]*?)(?=\*\*Recommendation\*\*|$)/)
  const recommendationMatch = text.match(/\*\*Recommendation\*\*\s*([\s\S]*?)$/)

  const rootCauses = rootCauseMatch
    ? rootCauseMatch[1]
        .split('\n')
        .map(l => l.replace(/^\s*\*\s*/, '').trim())
        .filter(Boolean)
    : []

  const estimatedLosses = estimatedLossMatch
    ? estimatedLossMatch[1]
        .split('\n')
        .map(l => l.replace(/^\s*\*\s*/, '').trim())
        .filter(Boolean)
    : []

  const recommendations = recommendationMatch
    ? recommendationMatch[1]
        .split('\n')
        .map(l => l.replace(/^\s*\*\s*/, '').trim())
        .filter(Boolean)
    : []

  return (
    <div className="space-y-3 font-sans text-xs text-left text-slate-200">
      <div className="flex items-center justify-between border-b border-red-500/20 pb-2 mb-1">
        <div className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-red-500 animate-blink" />
          <span className="font-bold text-red-400 tracking-wider font-mono text-[10px]">DIAGNOSIS ENGINE</span>
        </div>
        <button
          onClick={handleCopy}
          className="text-slate-500 hover:text-slate-300 transition-colors p-1"
          title="Copy raw markdown"
        >
          {copied ? <Check size={11} className="text-emerald-400" /> : <Copy size={11} />}
        </button>
      </div>

      {rootCauses.length > 0 && (
        <div className="space-y-1.5 bg-slate-900/60 p-2.5 rounded-xl border border-slate-800/80">
          <span className="font-semibold text-slate-500 block text-[9px] uppercase font-mono tracking-widest">Root Cause</span>
          <ul className="space-y-1 list-none pl-0">
            {rootCauses.map((rc, i) => {
              const isProb = rc.toLowerCase().includes('probability')
              return (
                <li key={i} className="flex items-start gap-1.5 leading-snug">
                  <span className={clsx("select-none mt-0.5", isProb ? "text-red-400 font-bold" : "text-slate-500")}>
                    {isProb ? "⚠" : "•"}
                  </span>
                  <span className={clsx(isProb && "text-slate-100 font-semibold")}>{rc}</span>
                </li>
              )
            })}
          </ul>
        </div>
      )}

      {estimatedLosses.length > 0 && (
        <div className="space-y-1.5 bg-slate-900/60 p-2.5 rounded-xl border border-slate-800/80">
          <span className="font-semibold text-slate-500 block text-[9px] uppercase font-mono tracking-widest">Estimated Loss</span>
          <div className="text-red-400 font-bold font-mono text-sm leading-none py-0.5">
            {estimatedLosses.join(', ').replace('**Financial Impact:**', '')}
          </div>
        </div>
      )}

      {recommendations.length > 0 && (
        <div className="space-y-1.5 bg-slate-900/60 p-2.5 rounded-xl border border-slate-800/80">
          <span className="font-semibold text-slate-500 block text-[9px] uppercase font-mono tracking-widest">Action Interventions</span>
          <ul className="space-y-1 list-none pl-0">
            {recommendations.map((rec, i) => (
              <li key={i} className="flex items-start gap-1.5 leading-snug">
                <span className="text-emerald-400 select-none font-bold mt-0.5">✓</span>
                <span>{rec}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="flex gap-2 pt-1 font-mono">
        <button
          onClick={() => alert('Operational Alert Acknowledged')}
          className="px-2.5 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 transition-all text-[9px]"
        >
          Acknowledge
        </button>
        <button
          onClick={() => alert('Crew Despatched to Substation Location')}
          className="px-2.5 py-1.5 rounded bg-red-600/25 hover:bg-red-600/40 text-red-300 border border-red-500/20 transition-all text-[9px]"
        >
          Dispatch Crew
        </button>
      </div>
    </div>
  )
}

// ── Main Component ───────────────────────────────────────────────────────────

export function GridCopilot({ onUnauthorized }: { onUnauthorized?: () => void }) {
  const [isOpen, setIsOpen]     = useState(false)
  const [isExpanded, setIsExpanded] = useState(false)
  const [messages, setMessages] = useState<Message[]>([
    {
      id: nextId(),
      role: 'assistant',
      text: '⚡ GridPulse Copilot online.\n\nI have full visibility into your live telemetry, anomaly detections, economic impact estimates, and 24h predictive forecasts.\n\nAsk me anything about your grid.',
      timestamp: new Date(),
    },
  ])
  const [input, setInput]         = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)
  const inputRef  = useRef<HTMLInputElement>(null)

  // Auto-scroll to the latest message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Focus input when panel opens
  useEffect(() => {
    if (isOpen) setTimeout(() => inputRef.current?.focus(), 120)
  }, [isOpen])

  const sendMessage = useCallback(async (text: string) => {
    if (!text.trim() || isLoading) return

    const userMsg: Message = {
      id: nextId(),
      role: 'user',
      text: text.trim(),
      timestamp: new Date(),
    }
    setMessages(prev => [...prev, userMsg])
    setInput('')
    setIsLoading(true)

    const { data: result, unauthorized } = await postCopilotQuery(text.trim())
    setIsLoading(false)

    if (unauthorized) {
      onUnauthorized?.()
      setMessages(prev => [
        ...prev,
        {
          id: nextId(),
          role: 'error' as const,
          text: 'Session expired. Please log in again.',
          timestamp: new Date(),
        },
      ])
      return
    }

    if (result === null) {
      setMessages(prev => [
        ...prev,
        {
          id: nextId(),
          role: 'error' as const,
          text: 'Copilot is unavailable right now (rate limit or API key not configured). Please try again in a moment.',
          timestamp: new Date(),
        },
      ])
      return
    }

    setMessages(prev => [
      ...prev,
      {
        id: nextId(),
        role: 'assistant' as const,
        text: result.answer,
        timestamp: new Date(),
        model: result.model,
        tokens: (result.input_tokens ?? 0) + (result.output_tokens ?? 0),
      },
    ])
  }, [isLoading, onUnauthorized])

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage(input)
    }
  }

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
  }

  const panelWidth  = isExpanded ? 'w-[520px]'  : 'w-[380px]'
  const panelHeight = isExpanded ? 'h-[600px]'  : 'h-[440px]'

  return (
    <>
      {/* Floating toggle button */}
      {!isOpen && (
        <button
          onClick={() => setIsOpen(true)}
          className={clsx(
            'fixed bottom-6 right-6 z-40',
            'flex items-center gap-2 px-4 py-3 rounded-2xl',
            'bg-indigo-600 hover:bg-indigo-500 text-white',
            'shadow-2xl shadow-indigo-900/50',
            'transition-all duration-200 hover:scale-105 active:scale-95',
            'border border-indigo-500/50 animate-pulse-blue',
          )}
        >
          <Terminal size={16} />
          <span className="text-sm font-semibold">Grid Copilot</span>
        </button>
      )}

      {/* Chat panel */}
      {isOpen && (
        <div
          className={clsx(
            'fixed bottom-6 right-6 z-40',
            panelWidth,
            panelHeight,
            'flex flex-col rounded-2xl overflow-hidden',
            'border border-slate-700/60',
            'bg-slate-950/95 backdrop-blur-2xl',
            'shadow-2xl shadow-black/60',
            'animate-fade-in',
          )}
        >
          {/* ── Panel header ── */}
          <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800/80 bg-slate-900/50 shrink-0">
            <div className="flex items-center gap-2">
              <div className="relative">
                <Terminal size={15} className="text-indigo-400" />
                <span className="absolute -top-0.5 -right-0.5 w-1.5 h-1.5 bg-emerald-400 rounded-full" />
              </div>
              <span className="text-slate-200 text-sm font-semibold font-mono">GridPulse Copilot</span>
              <span className="text-slate-600 text-xs font-mono">v2.5</span>
            </div>
            <div className="flex items-center gap-1">
              <button
                onClick={() => setIsExpanded(e => !e)}
                className="p-1.5 rounded-lg text-slate-500 hover:text-slate-300 hover:bg-slate-800/60 transition-colors"
                title={isExpanded ? 'Minimize' : 'Expand'}
              >
                {isExpanded ? <Minimize2 size={13} /> : <Maximize2 size={13} />}
              </button>
              <button
                onClick={() => setIsOpen(false)}
                className="p-1.5 rounded-lg text-slate-500 hover:text-slate-300 hover:bg-slate-800/60 transition-colors"
                title="Close"
              >
                <X size={13} />
              </button>
            </div>
          </div>

          {/* ── Message thread ── */}
          <div className="flex-1 overflow-y-auto px-4 py-3 space-y-4 min-h-0">
            {messages.map(msg => (
              <div key={msg.id} className={clsx('flex gap-2.5', msg.role === 'user' ? 'flex-row-reverse' : 'flex-row')}>
                {/* Avatar */}
                <div
                  className={clsx(
                    'w-7 h-7 rounded-lg flex items-center justify-center shrink-0 mt-0.5',
                    msg.role === 'user'      ? 'bg-indigo-600/30 text-indigo-300'
                    : msg.role === 'error'   ? 'bg-red-600/20 text-red-400'
                    : 'bg-slate-800 text-slate-400',
                  )}
                >
                  {msg.role === 'user'    ? <User size={12} />
                  : msg.role === 'error'  ? <AlertCircle size={12} />
                  : <Bot size={12} />}
                </div>

                {/* Bubble */}
                <div
                  className={clsx(
                    'max-w-[82%] rounded-xl px-3 py-2.5 text-xs leading-relaxed font-mono',
                    msg.role === 'user'
                      ? 'bg-indigo-600/20 border border-indigo-500/30 text-indigo-100 text-right'
                      : msg.role === 'error'
                      ? 'bg-red-500/10 border border-red-500/25 text-red-300'
                      : 'bg-slate-800/60 border border-slate-700/40 text-slate-200',
                  )}
                >
                  <RichMessage text={msg.text} onCopy={() => copyToClipboard(msg.text)} />
                  <div className="flex items-center gap-2 mt-1.5 opacity-40">
                    <span>
                      {msg.timestamp.toLocaleTimeString('en-IN', {
                        hour: '2-digit',
                        minute: '2-digit',
                        second: '2-digit',
                        hour12: false,
                      })}
                    </span>
                    {msg.model && <span>· {msg.model}</span>}
                    {msg.tokens ? <span>· {msg.tokens} tokens</span> : null}
                  </div>
                </div>
              </div>
            ))}

            {/* Loading indicator */}
            {isLoading && (
              <div className="flex gap-2.5">
                <div className="w-7 h-7 rounded-lg flex items-center justify-center bg-slate-800 text-slate-400 shrink-0">
                  <Bot size={12} />
                </div>
                <div className="bg-slate-800/60 border border-slate-700/40 rounded-xl px-3 py-2.5">
                  <div className="flex items-center gap-1.5">
                    {[0, 1, 2].map(i => (
                      <span
                        key={i}
                        className="w-1.5 h-1.5 bg-indigo-400 rounded-full animate-blink"
                        style={{ animationDelay: `${i * 200}ms` }}
                      />
                    ))}
                    <span className="text-slate-500 text-xs font-mono ml-1">Thinking…</span>
                  </div>
                </div>
              </div>
            )}

            <div ref={bottomRef} />
          </div>

          {/* ── Suggested prompts ── */}
          {messages.length <= 1 && (
            <div className="px-4 pb-2 flex flex-wrap gap-1.5 shrink-0">
              {SUGGESTED_QUERIES.slice(0, 2).map(q => (
                <button
                  key={q}
                  onClick={() => sendMessage(q)}
                  className="text-xs px-2.5 py-1.5 rounded-lg bg-slate-800/70 border border-slate-700/40 text-slate-400 hover:text-slate-200 hover:bg-slate-700/60 transition-colors font-mono text-left leading-snug"
                >
                  {q}
                </button>
              ))}
            </div>
          )}

          {/* ── Input bar ── */}
          <div className="px-3 pb-3 pt-2 border-t border-slate-800/80 shrink-0">
            <div className="flex items-center gap-2 bg-slate-800/50 border border-slate-700/40 rounded-xl px-3 py-2.5 focus-within:border-indigo-500/50 focus-within:ring-1 focus-within:ring-indigo-500/20 transition-all">
              <span className="text-indigo-500 font-mono text-sm shrink-0">›</span>
              <input
                ref={inputRef}
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask about your grid…"
                className="flex-1 bg-transparent text-slate-200 text-xs font-mono placeholder:text-slate-600 focus:outline-none"
                disabled={isLoading}
              />
              <button
                onClick={() => sendMessage(input)}
                disabled={!input.trim() || isLoading}
                className="p-1.5 rounded-lg text-indigo-400 hover:text-indigo-300 hover:bg-indigo-500/15 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
              >
                <Send size={13} />
              </button>
            </div>
            <p className="text-slate-700 text-xs mt-1.5 text-center font-mono">
              Powered by Gemini · grid-aware context injected automatically
            </p>
          </div>
        </div>
      )}
    </>
  )
}
