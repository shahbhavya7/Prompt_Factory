/** Starting and stopping voice_agent.py itself.
 *
 * Deliberately HTTP rather than the spectator websocket, and it has to be:
 * that websocket is opened BY voice_agent.py, so it does not exist while the
 * agent is down — which is precisely when "start the agent" needs to be
 * reachable. This API is the only piece already running at that moment.
 *
 * Note the two different "stop"s. `stopAgent` here ends the PROCESS. The
 * dashboard's End-call button goes over the websocket instead, which ends the
 * CALL and runs the learning loop while leaving the agent up for the next one.
 */

// Same-origin by default, proxied to :8000 by vite.config.js. It used to point
// straight at http://localhost:8000, which is correct only while the browser
// and the API are the same machine — the moment the dashboard is opened on a
// phone through a tunnel, "localhost" is the PHONE. Relative means "wherever
// this page came from", which is right in both cases.
const API_BASE = import.meta.env.VITE_API_BASE || "/api";

async function request(path, options) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) message = body.detail;
    } catch {
      /* not JSON — keep the status line */
    }
    throw new Error(message);
  }
  return res.json();
}

export function voiceStatus() {
  return request("/voice/status");
}

/** Spawn the agent. Resolves once it has survived its first couple of seconds,
 *  so a boot failure (missing credentials, database down) surfaces here rather
 *  than as a websocket that never connects. */
export function startAgent() {
  return request("/voice/start", { method: "POST" });
}

export function stopAgent() {
  return request("/voice/stop", { method: "POST" });
}

/** Mint a LiveKit token AND dispatch the agent into a fresh room, so this
 *  browser can join the call with its own microphone.
 *
 *  Both halves happen server-side in one request on purpose: the worker uses
 *  explicit dispatch, so a token alone would put the caller in an empty room.
 *  See api/main.py's /voice/join. */
export function joinCall() {
  return request("/voice/join", { method: "POST" });
}
