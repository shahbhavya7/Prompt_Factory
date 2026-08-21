function describe(entry) {
  const section = entry.intent && entry.intent !== "none" ? entry.intent : "the general section";
  switch (entry.outcome) {
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
        headline: `conflicts with what we already know about “${section}” — held for a person to check`,
        proof: entry.detail,
      };
    case "ungrounded-rejected":
      return {
        icon: "✗",
        headline: "discarded — not actually said on this call",
        proof: entry.detail,
      };
    default:
      return { icon: "•", headline: entry.outcome, proof: entry.detail };
  }
}

export default function LearningFeed({ ended, done, entries }) {
  if (!ended) return null;

  return (
    <div className="learning-panel">
      <h3>After the call — what memory learned</h3>
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
    </div>
  );
}
