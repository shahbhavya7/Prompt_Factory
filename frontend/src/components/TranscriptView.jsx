import { useEffect, useRef } from "react";

export default function TranscriptView({ turns }) {
  const logRef = useRef(null);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [turns]);

  if (turns.length === 0) {
    return (
      <div className="chat-empty">
        Waiting for a call — start one with <code>./run.sh voice</code> and speak into your mic.
      </div>
    );
  }

  return (
    <div className="chat-log" ref={logRef}>
      {turns.map((t) => (
        <div key={t.turn_index}>
          <div className="message-row caller">
            <div className="bubble">{t.user_text}</div>
          </div>
          <div className="message-row maya">
            <div className={`bubble ${t.outcome === "error" ? "error" : ""}`}>
              {t.reply_text}
            </div>
            <div className="message-meta">
              <span className={`outcome-chip ${t.outcome}`}>{t.outcome}</span>
              <span style={{ color: "var(--color-ink-faint)" }}>
                {t.latency_ms?.toFixed(0)} ms
              </span>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
