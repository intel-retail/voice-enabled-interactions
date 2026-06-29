import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev server proxies all backend calls so the SPA uses a single origin.
// In production nginx performs the same reverse-proxying (see nginx.conf).
const KIOSK_CORE = process.env.VITE_KIOSK_CORE_URL ?? 'http://localhost:8012'
const RAG        = process.env.VITE_RAG_URL        ?? 'http://localhost:8020'
const TTS        = process.env.VITE_TTS_URL        ?? 'http://localhost:8011'
const ASR        = process.env.VITE_ASR_URL        ?? 'http://localhost:8010'
const METRICS    = process.env.VITE_METRICS_URL    ?? 'http://localhost:9000'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 7860,
    strictPort: true,
    proxy: {
      '/api':     { target: KIOSK_CORE, changeOrigin: true },
      '/rag':     { target: RAG,     changeOrigin: true, rewrite: (p) => p.replace(/^\/rag/, '') },
      '/tts':     { target: TTS,     changeOrigin: true, rewrite: (p) => p.replace(/^\/tts/, '') },
      '/asr':     { target: ASR,     changeOrigin: true, rewrite: (p) => p.replace(/^\/asr/, '') },
      '/metrics-svc': { target: METRICS, changeOrigin: true, rewrite: (p) => p.replace(/^\/metrics-svc/, '') },
    },
  },
})
