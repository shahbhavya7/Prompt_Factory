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
              {/* Which path produced this reply. A cached reply skipped prompt
                  assembly and the model entirely, which is the difference the
                  latency next to it reflects — so it is labelled rather than
                  left looking like an ordinary generated turn. */}
              <span className={`path-chip ${t.cache_hit ? "path-chip--cache" : "path-chip--full"}`}>
                {t.cache_hit ? "⚡ cache" : "⚙ full pipeline"}
              </span>
              <span className={`outcome-chip ${t.outcome}`}>{t.outcome}</span>
              <span style={{ color: "var(--color-ink-faint)" }}>
                {t.latency_ms?.toFixed(0)} ms
              </span>
              {t.cache_hit && t.cache_similarity != null && (
                <span style={{ color: "var(--color-ink-faint)" }}>
                  match {t.cache_similarity.toFixed(3)}
                </span>
              )}
              {/* This turn was saved for reuse — so the next caller asking the
                  same thing gets it back on the fast path. Shown because the
                  cache filling up is otherwise completely invisible. */}
              {t.cache_stored?.stored && (
                <span className="path-chip path-chip--saved" title={t.cache_stored.id}>
                  💾 saved for reuse
                  {t.cache_stored.pending ? " (pinned)" : ""}
                </span>
              )}
              {/* And why a turn was NOT saved. Without this the cache filling
                  up slowly is indistinguishable from it being broken — which
                  cost a real debugging session once: the transcript showed
                  "full pipeline" on every turn with nothing to say why. The
                  reason is on the chip's tooltip rather than inline so it does
                  not crowd the meta row. */}
              {t.cache_stored && !t.cache_stored.stored && (
                <span
                  className="path-chip path-chip--notsaved"
                  title={`not saved: ${t.cache_stored.reason}`}
                >
                  not saved
                </span>
              )}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
