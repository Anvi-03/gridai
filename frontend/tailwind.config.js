/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{ts,tsx,js,jsx}'],
  theme: {
    extend: {
      // ── Custom keyframes for meter node states ──────────────────────────────
      keyframes: {
        'pulse-red': {
          '0%, 100%': { boxShadow: '0 0 0 0 rgba(239, 68, 68, 0)' },
          '50%':       { boxShadow: '0 0 24px 6px rgba(239, 68, 68, 0.45)' },
        },
        'pulse-amber': {
          '0%, 100%': { boxShadow: '0 0 0 0 rgba(245, 158, 11, 0)' },
          '50%':       { boxShadow: '0 0 18px 4px rgba(245, 158, 11, 0.35)' },
        },
        'pulse-blue': {
          '0%, 100%': { boxShadow: '0 0 0 0 rgba(99, 102, 241, 0)' },
          '50%':       { boxShadow: '0 0 14px 3px rgba(99, 102, 241, 0.3)' },
        },
        'blink': {
          '0%, 100%': { opacity: '1' },
          '50%':       { opacity: '0' },
        },
        'slide-in-right': {
          from: { transform: 'translateX(100%)' },
          to:   { transform: 'translateX(0)' },
        },
        'fade-in': {
          from: { opacity: '0', transform: 'translateY(8px)' },
          to:   { opacity: '1', transform: 'translateY(0)' },
        },
        'shimmer': {
          '0%':   { backgroundPosition: '-200% center' },
          '100%': { backgroundPosition: '200% center' },
        },
      },
      animation: {
        'pulse-red':        'pulse-red 1.4s ease-in-out infinite',
        'pulse-amber':      'pulse-amber 2s ease-in-out infinite',
        'pulse-blue':       'pulse-blue 2.5s ease-in-out infinite',
        'blink':            'blink 1s step-end infinite',
        'slide-in-right':   'slide-in-right 0.28s cubic-bezier(0.16,1,0.3,1)',
        'fade-in':          'fade-in 0.3s ease-out',
        'shimmer':          'shimmer 2.5s linear infinite',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', '"Fira Code"', '"Cascadia Code"', 'monospace'],
      },
      colors: {
        // Semantic risk palette
        risk: {
          low:      { DEFAULT: '#10b981', bg: 'rgba(16,185,129,0.12)', border: 'rgba(16,185,129,0.35)' },
          medium:   { DEFAULT: '#f59e0b', bg: 'rgba(245,158,11,0.12)', border: 'rgba(245,158,11,0.35)' },
          high:     { DEFAULT: '#f97316', bg: 'rgba(249,115,22,0.12)', border: 'rgba(249,115,22,0.35)' },
          critical: { DEFAULT: '#ef4444', bg: 'rgba(239,68,68,0.12)',  border: 'rgba(239,68,68,0.35)' },
        },
      },
      backdropBlur: {
        xs: '2px',
      },
    },
  },
  plugins: [],
}
