/** Start / end the call from the dashboard.
 *
 * Three states, one button, because there is exactly one sensible action at any
 * moment and a permanently greyed-out twin is noise:
 *
 *   agent down          -> "Start agent & call"  (HTTP: the API spawns run.sh voice)
 *   agent up, no call   -> "Start call"          (websocket: a fresh session)
 *   call in progress    -> "End call & learn"    (websocket: close + consolidate)
 *
 * Plus a "Stop agent" companion wherever a process is actually running, which
 * INCLUDES the disconnected state — an agent can be alive and still holding
 * port 8765 while its websocket is down, and that is the case most in need of
 * a kill button.
 *
 * The first case is the one that needs the HTTP path: the websocket that
 * carries the other two is opened by voice_agent.py, so it does not exist yet
 * when the agent is down.
 */
export default function CallControls({
  connected,
  callActive,
  starting,
  agentBooting,
  agentStopping,
  processAlive,
  onStartAgent,
  onStopAgent,
  onStart,
  onEnd,
}) {
  if (callActive) {
    return (
      <button className="btn btn-primary" disabled={!connected} onClick={onEnd}>
        End call &amp; learn
      </button>
    );
  }

  // `connected` is the websocket, which is the honest signal for whether the
  // agent is actually up — more reliable than the last /voice/status poll,
  // since an agent started from a terminal is equally connected.
  if (!connected) {
    // A process can still be ALIVE while the websocket is down: an agent that
    // crashed mid-boot, or one whose job ended but whose process is still
    // holding port 8765. That is precisely the "kill the leftover port" case,
    // and it is the state the dashboard shows as `not connected` — so Stop has
    // to be reachable here, not only when the agent is healthy. `processAlive`
    // comes from /voice/status, which counts agents this API did not spawn.
    return (
      <div className="btn-row">
        <button className="btn btn-primary" disabled={agentBooting} onClick={onStartAgent}>
          {agentBooting ? "Starting agent…" : "Start agent & call"}
        </button>
        {processAlive && (
          <button className="btn btn-ghost" disabled={agentStopping} onClick={onStopAgent}>
            {agentStopping ? "Stopping…" : "Kill stray agent"}
          </button>
        )}
      </div>
    );
  }

  // Agent up, no call: offer both another call and shutting the agent down.
  // The secondary button is what frees the websocket port when the session is
  // over — ending a CALL deliberately leaves the agent running for the next one.
  return (
    <div className="btn-row">
      <button
        className="btn btn-primary"
        // Disabled while starting so a double-click cannot ask for two calls.
        // The agent refuses the second anyway (see _start_call_from_dashboard),
        // but the button should not invite the mistake.
        disabled={starting || agentStopping}
        onClick={onStart}
      >
        {starting ? "Starting…" : "Start call"}
      </button>
      <button className="btn btn-ghost" disabled={agentStopping} onClick={onStopAgent}>
        {agentStopping ? "Stopping…" : "Stop agent"}
      </button>
    </div>
  );
}
