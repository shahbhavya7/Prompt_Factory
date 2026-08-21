import { useState } from "react";

/** One queued rule awaiting a human decision.
 *
 * Reads top-to-bottom as the reviewer's actual question: what happened on the
 * call, what the AI wants to remember from it, and what it would be matched
 * by — then approve / edit / discard.
 *
 * The cue is shown by default rather than hidden behind the edit toggle
 * because it is what actually gets embedded: a rule with a bad cue is stored
 * and then never retrieved, which is invisible unless the reviewer is looking
 * at the cue when they decide.
 */

const REASON_META = {
  pending: {
    label: "Awaiting approval",
    tone: "neutral",
    blurb: "Cleared the automatic checks. Nothing has been stored yet.",
  },
  conflict: {
    label: "Conflicts with an existing rule",
    tone: "warn",
    blurb:
      "This contradicts a rule already in memory. Approving it will not replace that rule — both would then be live, competing for the same turns.",
  },
  ungrounded: {
    label: "Not attested in the transcript",
    tone: "bad",
    blurb:
      "The line this was based on was not found in the call. Likely invented by the extractor — read it carefully before approving.",
  },
};

export default function ReviewCard({ item, intents, onApprove, onDiscard, busy }) {
  const [editing, setEditing] = useState(false);
  const [text, setText] = useState(item.text);
  const [cue, setCue] = useState(item.cue);
  const [priority, setPriority] = useState(item.priority || "normal");
  // "" is the general pool (intent NULL); "__new__" reveals the free-text box.
  const [intent, setIntent] = useState(item.intent || "");
  const [newIntent, setNewIntent] = useState("");
  const [intentTouched, setIntentTouched] = useState(false);

  // An unrecognised reason is a legacy row from before this queue existed
  // (the table predates it and kept its rows). Show the reason verbatim rather
  // than mislabelling it as "awaiting approval" — it still needs a decision,
  // but the blurb should not claim to know why it is here.
  const meta =
    REASON_META[item.reason] ||
    {
      label: item.reason,
      tone: "warn",
      blurb:
        "Queued by an earlier version of the learning loop. Read it on its merits — the original reason is no longer recorded.",
    };
  const dirty =
    text !== item.text ||
    cue !== item.cue ||
    priority !== (item.priority || "normal") ||
    intentTouched;

  const resolvedIntent = intent === "__new__" ? newIntent.trim().toLowerCase() : intent;
  const customIntentMissing = intent === "__new__" && !newIntent.trim();

  function handleApprove() {
    onApprove(item.id, {
      text,
      cue,
      priority,
      // Sent whenever the control was touched at all, so "the human chose the
      // general pool" (null) is distinguishable from "left as proposed".
      set_intent: intentTouched,
      intent: resolvedIntent || null,
    });
  }

  function reset() {
    setText(item.text);
    setCue(item.cue);
    setPriority(item.priority || "normal");
    setIntent(item.intent || "");
    setNewIntent("");
    setIntentTouched(false);
    setEditing(false);
  }

  return (
    <article className={`review-card review-card--${meta.tone}`}>
      <header className="review-card__head">
        <span className={`review-chip review-chip--${meta.tone}`}>{meta.label}</span>
        {item.intent ? (
          <span className="review-chip review-chip--quiet">intent · {item.intent}</span>
        ) : (
          <span className="review-chip review-chip--quiet">general pool</span>
        )}
        {item.learned_kind && (
          <span className="review-chip review-chip--quiet">{item.learned_kind}</span>
        )}
        {item.created_at && (
          <time className="review-card__time">
            {new Date(item.created_at).toLocaleString()}
          </time>
        )}
      </header>

      <p className="review-card__blurb">{meta.blurb}</p>

      {(item.trigger_message || item.trigger_reply) && (
        <div className="review-trigger">
          <span className="review-trigger__label">What happened on the call</span>
          {item.trigger_message && (
            <p className="review-trigger__line">
              <span className="review-trigger__who">Caller</span>
              {item.trigger_message}
            </p>
          )}
          {item.trigger_reply && (
            <p className="review-trigger__line">
              <span className="review-trigger__who review-trigger__who--maya">Maya</span>
              {item.trigger_reply}
            </p>
          )}
        </div>
      )}

      <div className="review-field">
        <label className="review-field__label">
          The rule
          <span className="review-field__hint">what Maya should do — shown to the model</span>
        </label>
        {editing ? (
          <textarea
            className="review-input"
            rows={4}
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
        ) : (
          <p className="review-field__value">{text}</p>
        )}
      </div>

      <div className="review-field">
        <label className="review-field__label">
          Matched by
          <span className="review-field__hint">
            this is what gets embedded — a bad cue means the rule is never found
          </span>
        </label>
        {editing ? (
          <textarea
            className="review-input review-input--mono"
            rows={3}
            value={cue}
            onChange={(e) => setCue(e.target.value)}
          />
        ) : (
          <p className="review-field__value review-field__value--mono">{cue || "(none)"}</p>
        )}
      </div>

      {editing && (
        <div className="review-row">
          <div className="review-field review-field--inline">
            <label className="review-field__label" htmlFor={`intent-${item.id}`}>
              Section
            </label>
            <select
              id={`intent-${item.id}`}
              className="review-select"
              value={intent}
              onChange={(e) => {
                setIntent(e.target.value);
                setIntentTouched(true);
              }}
            >
              <option value="">general pool (no intent)</option>
              {intents.map((label) => (
                <option key={label} value={label}>
                  {label}
                </option>
              ))}
              <option value="__new__">+ new section…</option>
            </select>
            {intent === "__new__" && (
              <input
                className="review-input review-input--slim"
                placeholder="new_intent_name"
                value={newIntent}
                onChange={(e) => setNewIntent(e.target.value)}
              />
            )}
          </div>

          <div className="review-field review-field--inline">
            <label className="review-field__label" htmlFor={`prio-${item.id}`}>
              Priority
            </label>
            <select
              id={`prio-${item.id}`}
              className="review-select"
              value={priority}
              onChange={(e) => setPriority(e.target.value)}
            >
              {["critical", "high", "normal", "low"].map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </div>
        </div>
      )}

      {intent === "__new__" && newIntent.trim() && (
        <p className="review-warn">
          A brand-new section needs example caller phrasings in
          <code> kb.INTENT_EXEMPLARS </code>
          before the router can classify anyone into it — until then this rule is
          stored but never retrieved.
        </p>
      )}

      {item.existing_chunk_id && (
        <p className="review-warn">
          Conflicts with <code>{item.existing_chunk_id}</code>, which stays live and
          unchanged whatever you decide here.
        </p>
      )}

      <footer className="review-actions">
        <button
          className="btn btn-primary"
          onClick={handleApprove}
          disabled={busy || !text.trim() || customIntentMissing}
        >
          {dirty ? "Approve with edits" : "Approve"}
        </button>
        {editing ? (
          <button className="btn btn-quiet" onClick={reset} disabled={busy}>
            Cancel edits
          </button>
        ) : (
          <button className="btn btn-quiet" onClick={() => setEditing(true)} disabled={busy}>
            Edit
          </button>
        )}
        <button
          className="btn btn-danger"
          onClick={() => onDiscard(item.id)}
          disabled={busy}
        >
          Discard
        </button>
      </footer>
    </article>
  );
}
