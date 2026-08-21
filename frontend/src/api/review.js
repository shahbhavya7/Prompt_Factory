/** The review queue's HTTP client.
 *
 * Deliberately HTTP, not the spectator websocket: the queue is reviewed
 * whenever a person has time, which is usually with no call running and often
 * with voice_agent.py shut down entirely. The websocket only exists while a
 * call is live, so a queue built on it could not be opened at the one moment
 * it is most likely to be needed.
 */

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

async function request(path, options) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    // FastAPI puts the message in `detail`; fall back to the status line for
    // anything that isn't a FastAPI error shape (a proxy, a dead port).
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

export function fetchPending() {
  return request("/review/pending");
}

/** Approve one rule. `edits` carries only what the human changed; an absent
 *  field means "keep what the extractor proposed". `set_intent` must be sent
 *  whenever the intent control was touched, because intent=null is itself a
 *  valid choice (a general rule) and cannot double as "unchanged". */
export function approveRule(id, edits) {
  return request(`/review/${id}/approve`, {
    method: "POST",
    body: JSON.stringify(edits),
  });
}

export function discardRule(id) {
  return request(`/review/${id}/discard`, { method: "POST" });
}
