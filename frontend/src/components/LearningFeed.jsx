function describe(entry) {
  const section = entry.intent && entry.intent !== "none" ? entry.intent : "the general section";
  switch (entry.outcome) {
    // A candidate clearing every automated gate now earns a place in the human
    // review queue, not a place in memory — so this deliberately does NOT say
    // "learned". Nothing is live until a person approves it.
    case "queued-for-approval":
      return {
        icon: "⏳",
        headline: `proposed a new rule for “${section}” — waiting for your approval`,
        proof: entry.detail,
      };
    // Retained for transcripts recorded before the approval queue existed,
    // whose stored learning_results still carry the old outcome name.
    case "inserted":
      return {
        icon: "✓",
        headline: `learned something new — added to the “${section}” section`,
        proof: entry.detail, // "id=learned_408c8991"
      };
    case "duplicate-skipped":
      return {
        icon: "•",
        headline: `already knew this — skipped as a duplicate in “${section}”`,
        proof: entry.detail,
      };
    case "conflict-needs-review":
      return {
        icon: "⚠",
        headline: `conflicts with what we already know about “${section}” — waiting for your review`,
        proof: entry.detail,
      };
    case "ungrounded-rejected":
      return {
        icon: "✗",
        headline: "not actually said on this call — waiting for your review",
        proof: entry.detail,
      };
    default:
      return { icon: "•", headline: entry.outcome, proof: entry.detail };
  }
}

export default function LearningFeed({ ended, done, entries, onReview }) {
  if (!ended) return null;

  const waiting = entries.filter((e) =>
    ["queued-for-approval", "conflict-needs-review", "ungrounded-rejected"].includes(e.outcome)
  ).length;

  return (
    <div className="learning-panel">
      <h3>After the call — what the agent proposed</h3>
      {entries.length === 0 ? (
        <p style={{ color: "var(--color-ink-soft)", margin: 0 }}>
          {done
            ? "Reviewed the call — nothing new to learn from it."
            : "Reviewing the call for anything new…"}
        </p>
      ) : (
        entries.map((entry, i) => {
          const { icon, headline, proof } = describe(entry);
          return (
            <div className="learning-item" key={i}>
              <span className="outcome">
                {icon} {headline}
                {proof ? ` — ${proof}` : ""}
              </span>
              <span className="text">{entry.text}</span>
            </div>
          );
        })
      )}

      {waiting > 0 && (
        <button className="btn btn-primary learning-panel__cta" onClick={onReview}>
          Review {waiting} proposed {waiting === 1 ? "rule" : "rules"} →
        </button>
      )}
    </div>
  );
}
