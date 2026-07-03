/**
 * GridPulse AI — LoginModal.tsx
 *
 * A full-screen glassmorphic auth overlay shown when the user is not
 * authenticated.  Supports two modes toggled by a tab bar:
 *
 *   • Login   — OAuth2 form-encoded POST /api/v1/auth/login
 *   • Register — JSON POST /api/v1/auth/register, then auto-switches to Login
 *
 * On success the parent receives the JWT via onAuthSuccess(token) and
 * removes the overlay.
 */

import { useState, useId } from 'react'
import type { FormEvent } from 'react'
import { Zap, LogIn, UserPlus, Eye, EyeOff, Loader2, ShieldCheck, AlertCircle } from 'lucide-react'
import { clsx } from 'clsx'
import { postLogin, postRegister, setToken } from '../lib/api'

// ── Types ─────────────────────────────────────────────────────────────────────

interface LoginModalProps {
  /** Called with the raw JWT string after a successful login. */
  onAuthSuccess: (token: string) => void
}

type TabMode = 'login' | 'register'

// ── Sub-components ────────────────────────────────────────────────────────────

function InputField({
  id,
  label,
  type,
  value,
  onChange,
  placeholder,
  autoComplete,
  disabled,
}: {
  id: string
  label: string
  type: string
  value: string
  onChange: (v: string) => void
  placeholder?: string
  autoComplete?: string
  disabled?: boolean
}) {
  const [show, setShow] = useState(false)
  const isPassword = type === 'password'
  const resolvedType = isPassword ? (show ? 'text' : 'password') : type

  return (
    <div className="space-y-1.5">
      <label htmlFor={id} className="block text-xs font-medium text-slate-400 tracking-wide">
        {label}
      </label>
      <div className="relative">
        <input
          id={id}
          type={resolvedType}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          autoComplete={autoComplete}
          disabled={disabled}
          className={clsx(
            'w-full bg-slate-800/70 border border-slate-700/60 rounded-xl px-4 py-3',
            'text-slate-100 placeholder:text-slate-600 text-sm font-mono',
            'focus:outline-none focus:border-indigo-500/60 focus:ring-1 focus:ring-indigo-500/30',
            'transition-all duration-200',
            isPassword && 'pr-11',
            disabled && 'opacity-50 cursor-not-allowed',
          )}
        />
        {isPassword && (
          <button
            type="button"
            onClick={() => setShow((s) => !s)}
            tabIndex={-1}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300 transition-colors"
            aria-label={show ? 'Hide password' : 'Show password'}
          >
            {show ? <EyeOff size={15} /> : <Eye size={15} />}
          </button>
        )}
      </div>
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────────────────

export function LoginModal({ onAuthSuccess }: LoginModalProps) {
  const uid = useId()

  // ── Tab state ────────────────────────────────────────────────────────────────
  const [mode, setMode] = useState<TabMode>('login')

  // ── Form fields ──────────────────────────────────────────────────────────────
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')

  // ── UI states ─────────────────────────────────────────────────────────────────
  const [isLoading, setIsLoading]     = useState(false)
  const [errorMsg, setErrorMsg]       = useState<string | null>(null)
  const [successMsg, setSuccessMsg]   = useState<string | null>(null)

  // ── Helpers ───────────────────────────────────────────────────────────────────
  const clearMessages = () => { setErrorMsg(null); setSuccessMsg(null) }

  const switchMode = (next: TabMode) => {
    setMode(next)
    setUsername('')
    setPassword('')
    setConfirmPassword('')
    clearMessages()
  }

  // ── Login handler ─────────────────────────────────────────────────────────────
  const handleLogin = async (e: FormEvent) => {
    e.preventDefault()
    clearMessages()

    if (!username.trim() || !password) {
      setErrorMsg('Please enter both username and password.')
      return
    }

    setIsLoading(true)
    try {
      const res = await postLogin(username.trim(), password)
      if (!res) {
        setErrorMsg('Invalid username or password. Please try again.')
        return
      }
      setToken(res.access_token)
      onAuthSuccess(res.access_token)
    } finally {
      setIsLoading(false)
    }
  }

  // ── Register handler ──────────────────────────────────────────────────────────
  const handleRegister = async (e: FormEvent) => {
    e.preventDefault()
    clearMessages()

    if (!username.trim()) {
      setErrorMsg('Username is required.')
      return
    }
    if (password.length < 8) {
      setErrorMsg('Password must be at least 8 characters.')
      return
    }
    if (password !== confirmPassword) {
      setErrorMsg('Passwords do not match.')
      return
    }

    setIsLoading(true)
    try {
      const res = await postRegister(username.trim(), password)
      if (!res) {
        setErrorMsg('Registration failed. The username may already be taken.')
        return
      }
      setSuccessMsg(`Account '${res.username}' created! Switching to login…`)
      setTimeout(() => switchMode('login'), 1400)
    } finally {
      setIsLoading(false)
    }
  }

  // ── Render ────────────────────────────────────────────────────────────────────
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      style={{
        background:
          'radial-gradient(ellipse 80% 80% at 50% -20%, rgba(99,102,241,0.18), transparent), #020817',
      }}
      role="dialog"
      aria-modal="true"
      aria-labelledby={`${uid}-title`}
    >
      {/* ── Decorative ambient blobs ───────────────────────────────────────────── */}
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-indigo-600/6 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-cyan-600/5 rounded-full blur-3xl pointer-events-none" />

      {/* ── Modal card ────────────────────────────────────────────────────────── */}
      <div
        className="relative w-full max-w-sm mx-4 animate-fade-in"
        style={{ animation: 'fadeInScale 0.25s ease both' }}
      >
        <div
          className="rounded-2xl border border-slate-700/50 bg-slate-900/80 backdrop-blur-2xl p-8 shadow-2xl"
          style={{ boxShadow: '0 0 0 1px rgba(99,102,241,0.08), 0 24px 64px rgba(0,0,0,0.6)' }}
        >
          {/* ── Brand header ──────────────────────────────────────────────────── */}
          <div className="flex flex-col items-center gap-3 mb-8">
            <div className="p-3 rounded-2xl bg-indigo-600/15 border border-indigo-500/25">
              <Zap size={22} className="text-indigo-400" />
            </div>
            <div className="text-center">
              <h1 id={`${uid}-title`} className="text-lg font-bold text-slate-100 tracking-tight">
                GridPulse <span className="text-indigo-400">AI</span>
              </h1>
              <p className="text-xs text-slate-500 mt-0.5 font-mono">
                Operator Authentication Portal
              </p>
            </div>
          </div>

          {/* ── Mode tabs ─────────────────────────────────────────────────────── */}
          <div className="flex rounded-xl bg-slate-800/60 p-1 mb-6 gap-1">
            {([
              { key: 'login' as TabMode,    label: 'Login',    Icon: LogIn    },
              { key: 'register' as TabMode, label: 'Register', Icon: UserPlus },
            ] as const).map(({ key, label, Icon }) => (
              <button
                key={key}
                id={`${uid}-tab-${key}`}
                onClick={() => switchMode(key)}
                className={clsx(
                  'flex-1 flex items-center justify-center gap-1.5 py-2 rounded-lg text-xs font-medium transition-all duration-200',
                  mode === key
                    ? 'bg-indigo-600/80 text-white shadow-md'
                    : 'text-slate-500 hover:text-slate-300',
                )}
                aria-selected={mode === key}
                role="tab"
              >
                <Icon size={12} />
                {label}
              </button>
            ))}
          </div>

          {/* ── Form ──────────────────────────────────────────────────────────── */}
          <form
            onSubmit={mode === 'login' ? handleLogin : handleRegister}
            className="space-y-4"
            noValidate
          >
            <InputField
              id={`${uid}-username`}
              label="Username"
              type="text"
              value={username}
              onChange={setUsername}
              placeholder="grid_operator_01"
              autoComplete={mode === 'login' ? 'username' : 'username'}
              disabled={isLoading}
            />

            <InputField
              id={`${uid}-password`}
              label="Password"
              type="password"
              value={password}
              onChange={setPassword}
              placeholder="••••••••"
              autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
              disabled={isLoading}
            />

            {mode === 'register' && (
              <InputField
                id={`${uid}-confirm-password`}
                label="Confirm Password"
                type="password"
                value={confirmPassword}
                onChange={setConfirmPassword}
                placeholder="••••••••"
                autoComplete="new-password"
                disabled={isLoading}
              />
            )}

            {/* ── Feedback messages ──────────────────────────────────────────── */}
            {errorMsg && (
              <div className="flex items-start gap-2 px-3 py-2.5 rounded-xl bg-red-500/10 border border-red-500/25 text-red-400 text-xs font-mono">
                <AlertCircle size={13} className="mt-0.5 shrink-0" />
                <span>{errorMsg}</span>
              </div>
            )}
            {successMsg && (
              <div className="flex items-start gap-2 px-3 py-2.5 rounded-xl bg-emerald-500/10 border border-emerald-500/25 text-emerald-400 text-xs font-mono">
                <ShieldCheck size={13} className="mt-0.5 shrink-0" />
                <span>{successMsg}</span>
              </div>
            )}

            {/* ── Submit ────────────────────────────────────────────────────── */}
            <button
              id={`${uid}-submit`}
              type="submit"
              disabled={isLoading}
              className={clsx(
                'w-full flex items-center justify-center gap-2 py-3 rounded-xl text-sm font-semibold',
                'transition-all duration-200',
                isLoading
                  ? 'bg-indigo-700/50 text-indigo-300 cursor-not-allowed'
                  : 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-lg hover:shadow-indigo-500/25',
              )}
            >
              {isLoading ? (
                <>
                  <Loader2 size={15} className="animate-spin" />
                  {mode === 'login' ? 'Signing in…' : 'Creating account…'}
                </>
              ) : mode === 'login' ? (
                <>
                  <LogIn size={15} />
                  Sign In
                </>
              ) : (
                <>
                  <UserPlus size={15} />
                  Create Account
                </>
              )}
            </button>
          </form>

          {/* ── Footer hint ───────────────────────────────────────────────────── */}
          <p className="text-center text-[11px] text-slate-600 font-mono mt-6">
            {mode === 'login' ? (
              <>
                No account?{' '}
                <button
                  onClick={() => switchMode('register')}
                  className="text-indigo-400 hover:text-indigo-300 transition-colors underline underline-offset-2"
                >
                  Register here
                </button>
              </>
            ) : (
              <>
                Already have an account?{' '}
                <button
                  onClick={() => switchMode('login')}
                  className="text-indigo-400 hover:text-indigo-300 transition-colors underline underline-offset-2"
                >
                  Sign in
                </button>
              </>
            )}
          </p>
        </div>

        {/* ── Security note ────────────────────────────────────────────────────── */}
        <p className="text-center text-[10px] text-slate-700 font-mono mt-3 flex items-center justify-center gap-1">
          <ShieldCheck size={10} />
          Protected by JWT · HS256 · 60-min expiry
        </p>
      </div>

      {/* ── Keyframe animation injected inline (no external CSS needed) ────────── */}
      <style>{`
        @keyframes fadeInScale {
          from { opacity: 0; transform: scale(0.96) translateY(8px); }
          to   { opacity: 1; transform: scale(1)    translateY(0);    }
        }
      `}</style>
    </div>
  )
}
