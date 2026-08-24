const OUTCOME_TEXT = {
  grounded: "✓ checked the reply — it stuck to that section",
  ungrounded: "⚠ the reply drifted from that section",
  spliced: "⚠ the reply borrowed from the wrong section",
  "no-rule": "— nothing in memory matched closely enough",
  unscored: "— not scored",
  error: "✗ something went wrong generating the reply",
};

/** One card per turn — the live version of "did it fetch from memory,
 * and what did it actually find." Walks the same 4 steps as the mermaid
 * diagram in docs/VOICE_PIPELINE.md: what was said, searched memory,
 * found one section, built a small prompt from it, checked the reply
 * stuck to it. `retrieval` arrives before the reply exists; `turn`
 * (same turn_index) fills in steps 3–4 once the reply is validated. */
export default function MemoryLookupCard({ retrieval, turn }) {
  const hasMatch = !!retrieval.governing_rule_id;
  const cached = !!turn?.cache_hit;

  return (
    <div
      className={`learning-panel${cached ? " learning-panel--cached" : ""}`}
      style={{ marginBottom: "var(--space-2)" }}
    >
      {cached && (
        <div className="cache-banner">
          <strong>⚡ served from cache</strong> — this exact question had been
          answered and verified before, so the saved reply was reused: no prompt
          built, no model call
          {turn.cache_similarity != null
            ? ` (match ${turn.cache_similarity.toFixed(3)})`
            : ""}
          .
        </div>
      )}
      <div className="detail-row">
        <span className="k">1 · caller said</span>
        <span className="v" style={{ fontFamily: "inherit", fontWeight: 400 }}>
          “{retrieval.user_text}”
        </span>
      </div>
      <div className="detail-row">
        <span className="k">2 · searched memory for</span>
        <span className="v">
          {retrieval.intent ? `intent “${retrieval.intent}”` : "the general pool"}
          {" "}(cos {retrieval.intent_cosine?.toFixed(3)})
        </span>
      </div>
      <div className="detail-row">
        <span className="k">3 · found</span>
        <span className="v">
          {hasMatch
            ? `${retrieval.governing_rule_id} — ${retrieval.governing_rule_title}`
            : "no matching section"}
        </span>
      </div>
      {hasMatch && retrieval.governing_rule_snippet && (
        <div className="detail-row">
          <span className="k">that section says</span>
          <span className="v" style={{ fontFamily: "inherit", fontWeight: 400 }}>
            {retrieval.governing_rule_snippet}
          </span>
        </div>
      )}
      <div className="detail-row">
        <span className="k">4 · prompt built</span>
        <span className="v">
          {cached ? (
            <em>none — the saved reply was reused as-is</em>
          ) : (
            <>
              {retrieval.assembled_tokens?.toLocaleString()} tok, not{" "}
              {retrieval.monolith_tokens?.toLocaleString()}
            </>
          )}
        </span>
      </div>

      {turn ? (
        <>
          <div className="detail-row">
            <span className="k">5 · checked</span>
            <span className="v">
              {cached ? (
                <em>
                  already verified when it was first answered — not re-checked
                </em>
              ) : (
                <>
                  {OUTCOME_TEXT[turn.outcome] || turn.outcome} (cos{" "}
                  {turn.grounding_cosine?.toFixed(3)})
                </>
              )}
            </span>
          </div>
          <div className="detail-row">
            <span className="k">6 · Maya replied</span>
            <span className="v" style={{ fontFamily: "inherit", fontWeight: 400 }}>
              “{turn.reply_text}”
            </span>
          </div>
          {/* Whether this turn joined the cache. Worth its own row: a saved
              turn is why a LATER call is fast, and a refusal is the only place
              the exclusion rules become visible. A cache hit is skipped — it
              is already stored by definition. */}
          {!cached && turn.cache_stored && (
            <div className="detail-row">
              <span className="k">7 · saved for reuse</span>
              <span className="v">
                {turn.cache_stored.stored ? (
                  <>
                    ✓ yes — the next caller asking this skips the model
                    {turn.cache_stored.id ? ` (${turn.cache_stored.id})` : ""}
                  </>
                ) : (
                  <em>no — {turn.cache_stored.reason}</em>
                )}
              </span>
            </div>
          )}
          {turn.prompt_sent && (
            <details className="detail-prompt">
              <summary>
                {cached
                  ? "what was matched in the cache (no prompt was sent)"
                  : `full prompt sent to the model this turn (${retrieval.assembled_tokens?.toLocaleString()} tok)`}
              </summary>
              <pre>{turn.prompt_sent}</pre>
            </details>
          )}
        </>
      ) : (
        <div className="status-line">generating the reply…</div>
      )}
    </div>
  );
}
