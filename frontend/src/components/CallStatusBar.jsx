export default function CallStatusBar({
  connected,
  callActive,
  sessionId,
  campaign,
  starting,
  agentBooting,
  startError,
}) {
  return (
    <div className="call-state-bar">
      <span className={`chip ${connected ? "active" : "ended"}`}>
        {connected ? "watching voice_agent.py" : "not connected"}
      </span>
      {/* Distinguishes the two waits, because they fail differently and take
          very different times: booting the agent is ~10s of engine load and
          Deepgram connections, while starting a call on a live agent is ~2s. */}
      <span className={`chip ${callActive || starting || agentBooting ? "active" : ""}`}>
        {callActive
          ? "call in progress"
          : agentBooting
            ? "starting voice_agent.py…"
            : starting
              ? "starting the call…"
              : "no call in progress"}
      </span>
      {/* Same chip, not a new one — a renewal call needs to read as visibly
          different from any other campaign right here, not in a second slot. */}
      {sessionId && (
        <span className="chip">{campaign ? `${campaign} · ${sessionId}` : sessionId}</span>
      )}
      {/* The agent's own reason for refusing a start. Shown rather than only
          logged: the button is remote, so a refusal that appears nowhere is
          indistinguishable from a button that does nothing. */}
      {startError && !callActive && (
        <span className="chip ended" title={startError}>
          could not start — {startError}
        </span>
      )}
    </div>
  );
}
