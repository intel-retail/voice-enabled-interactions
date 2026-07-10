/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        intel: {
          blue: '#0071c5',
          'blue-dark': '#005a9e',
          dark: '#2B2C30',
          gray: '#6A6D75',
        },
        // Kiosk chat theme
        kiosk: {
          bg: '#FFFFFF',
          pane: '#F4F7FB',
          asst: '#EBF2FA',
          border: '#C8D8EA',
          user: '#0068B5',
          accent: '#0068B5',
          textmd: '#4A6070',
          textlo: '#8FA0AE',
        },
        // ── Performance Dashboard color system ───────────────────────────────
        // Hardware accelerator colors
        cpu:  { DEFAULT: '#0071c5', light: '#dbeafe', muted: '#93c5fd', dark: '#1d4ed8' },
        gpu:  { DEFAULT: '#16a34a', light: '#dcfce7', muted: '#86efac', dark: '#15803d' },
        npu:  { DEFAULT: '#9333ea', light: '#f3e8ff', muted: '#c084fc', dark: '#7e22ce' },
        // AI pipeline stage colors
        asr:  { DEFAULT: '#ea580c', light: '#ffedd5', muted: '#fb923c', dark: '#c2410c' },
        ret:  { DEFAULT: '#ca8a04', light: '#fef9c3', muted: '#fde047', dark: '#a16207' },
        llm:  { DEFAULT: '#0891b2', light: '#cffafe', muted: '#67e8f9', dark: '#0e7490' },
        tts:  { DEFAULT: '#db2777', light: '#fce7f3', muted: '#f9a8d4', dark: '#be185d' },
        // Dashboard surface colors
        dash: {
          bg:     '#0f172a',
          card:   '#1e293b',
          border: '#334155',
          label:  '#94a3b8',
          value:  '#f1f5f9',
        },
      },
      fontFamily: {
        display: ['"IntelOne Display"', '"Inter"', 'system-ui', 'sans-serif'],
        text:    ['"IntelOne Text"',    '"Inter"', 'system-ui', 'sans-serif'],
        body:    ['"IntelOne Text"',    '"Inter"', 'system-ui', 'sans-serif'],
        mono:    ['"Roboto Mono"', 'ui-monospace', 'monospace'],
      },
      keyframes: {
        'stage-pulse': {
          '0%, 100%': { opacity: '1', transform: 'scale(1)' },
          '50%':       { opacity: '0.7', transform: 'scale(1.04)' },
        },
        'kpi-glow': {
          '0%, 100%': { boxShadow: '0 0 0 0 rgba(0,113,197,0)' },
          '50%':       { boxShadow: '0 0 12px 2px rgba(0,113,197,0.35)' },
        },
        'bar-rise': {
          from: { transform: 'scaleY(0)' },
          to:   { transform: 'scaleY(1)' },
        },
        'number-tick': {
          from: { opacity: '0', transform: 'translateY(-4px)' },
          to:   { opacity: '1', transform: 'translateY(0)' },
        },
        'pipe-flow': {
          '0%':   { strokeDashoffset: '200' },
          '100%': { strokeDashoffset: '0' },
        },
      },
      animation: {
        'stage-pulse':  'stage-pulse 1.8s ease-in-out infinite',
        'kpi-glow':     'kpi-glow 2.5s ease-in-out infinite',
        'number-tick':  'number-tick 0.25s ease-out',
        'bar-rise':     'bar-rise 0.6s ease-out',
        'spin-slow':    'spin 2s linear infinite',
      },
    },
  },
  plugins: [],
}
