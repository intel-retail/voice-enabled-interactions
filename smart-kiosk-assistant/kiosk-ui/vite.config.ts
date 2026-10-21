import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'

// Dev server proxies all backend calls so the SPA uses a single origin.
// In production nginx performs the same reverse-proxying (see nginx.conf).
const KIOSK_CORE = process.env.VITE_KIOSK_CORE_URL ?? 'http://localhost:8012'
const RAG        = process.env.VITE_RAG_URL        ?? 'http://localhost:8020'
const TTS        = process.env.VITE_TTS_URL        ?? 'http://localhost:8011'
const ASR        = process.env.VITE_ASR_URL        ?? 'http://localhost:8010'
const METRICS    = process.env.VITE_METRICS_URL    ?? 'http://localhost:9000'
const QUEUE      = process.env.VITE_QUEUE_URL      ?? 'http://localhost:8090'

// Dev-only port + UI mode, so `KIOSK_UI_MODE=customer PORT=7861 npm run dev`
// mirrors the two-container split (kiosk-ui / kiosk-ui-customer) used in
// docker-compose.yml without needing a second checkout or build.
const PORT = Number(process.env.PORT ?? 7860)
const UI_MODE = process.env.KIOSK_UI_MODE ?? 'operator'

// Serves /config.js from the dev server itself (bypassing the static file in
// /public) so KIOSK_UI_MODE can be flipped per `npm run dev` invocation,
// matching how docker-entrypoint.sh generates it in production.
function uiModeConfigPlugin(): Plugin {
  return {
    name: 'kiosk-ui-mode-config',
    configureServer(server) {
      server.middlewares.use('/config.js', (_req, res) => {
        res.setHeader('Content-Type', 'application/javascript')
        res.end(`window.__KIOSK_UI_MODE__ = ${JSON.stringify(UI_MODE)};\n`)
      })
    },
  }
}

export default defineConfig({
  plugins: [react(), uiModeConfigPlugin()],
  server: {
    host: '0.0.0.0',
    port: PORT,
    strictPort: true,
    proxy: {
      '/api/v1/sessions/stream': { target: KIOSK_CORE, changeOrigin: true, ws: true },
      '/api':     { target: KIOSK_CORE, changeOrigin: true },
      '/rag':     { target: RAG,     changeOrigin: true, rewrite: (p) => p.replace(/^\/rag/, '') },
      '/tts':     { target: TTS,     changeOrigin: true, rewrite: (p) => p.replace(/^\/tts/, '') },
      '/asr':     { target: ASR,     changeOrigin: true, rewrite: (p) => p.replace(/^\/asr/, '') },
      '/metrics-svc': { target: METRICS, changeOrigin: true, rewrite: (p) => p.replace(/^\/metrics-svc/, '') },
      '/queue-svc':   { target: QUEUE,   changeOrigin: true, rewrite: (p) => p.replace(/^\/queue-svc/, '') },
    },
  },
})
