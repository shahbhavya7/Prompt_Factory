import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev server config exists for ONE reason: the dashboard has to be usable from
// a phone, over an ngrok tunnel, without becoming a second deployment target.
//
// The shape that makes that work is single-origin. The dashboard talks to three
// things — itself, the FastAPI on :8000, and voice_agent.py's spectator
// websocket on :8765 — and if the browser addressed those directly it would
// need three tunnels, three CORS origins, and a mixed-content exemption for a
// ws:// socket on an https:// page. Proxying both through Vite means the phone
// sees exactly one origin (the tunnel's), so one tunnel covers everything, CORS
// never enters into it, and the socket inherits the page's TLS as wss://.
export default defineConfig({
  plugins: [react()],
  server: {
    // 0.0.0.0, not localhost: ngrok connects from off-box, and the default
    // loopback bind refuses it.
    host: true,
    // Vite rejects requests whose Host header it does not recognise, which is
    // every request arriving through a tunnel — the ngrok subdomain changes on
    // each restart, so there is no fixed hostname to allow. This is a dev
    // server reachable only via a tunnel the user started themselves.
    allowedHosts: true,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
      // The spectator broadcast. `ws: true` is what makes Vite upgrade the
      // connection rather than treat it as a normal request.
      '/ws': {
        target: 'ws://localhost:8765',
        ws: true,
        rewrite: (path) => path.replace(/^\/ws/, ''),
      },
    },
  },
})
