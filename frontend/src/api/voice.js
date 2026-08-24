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

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

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
