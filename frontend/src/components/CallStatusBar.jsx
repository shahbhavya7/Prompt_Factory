export default function CallStatusBar({ connected, callActive, sessionId }) {
  return (
    <div className="call-state-bar">
      <span className={`chip ${connected ? "active" : "ended"}`}>
        {connected ? "watching voice_agent.py" : "not connected"}
      </span>
      <span className={`chip ${callActive ? "active" : ""}`}>
        {callActive ? "call in progress" : "no call in progress"}
      </span>
      {sessionId && <span className="chip">{sessionId}</span>}
    </div>
  );
}
