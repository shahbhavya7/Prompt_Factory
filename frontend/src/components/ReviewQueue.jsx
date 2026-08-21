import { useCallback, useEffect, useState } from "react";
import { approveRule, discardRule, fetchPending } from "../api/review";
import ReviewCard from "./ReviewCard";

/** The review inbox: every rule the agent proposed and no human has ruled on.
 *
 * Rows are removed optimistically on a decision so the queue visibly drains as
 * the reviewer works, rather than snapping around on each refetch. A failure
 * puts the row back and shows why.
 */
export default function ReviewQueue({ onCountChange, refreshToken }) {
  const [items, setItems] = useState([]);
  const [intents, setIntents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [busyId, setBusyId] = useState(null);
  const [flash, setFlash] = useState(null);

  const load = useCallback(async () => {
    try {
      setError(null);
      const data = await fetchPending();
      setItems(data.pending || []);
      setIntents(data.intents || []);
      onCountChange?.(data.count ?? (data.pending || []).length);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [onCountChange]);

  useEffect(() => {
    load();
  }, [load, refreshToken]);

  async function act(id, fn, describe) {
    setBusyId(id);
    setError(null);
    const snapshot = items;
    // Optimistic: drop the row now so a long queue drains as you work.
    setItems((rows) => rows.filter((r) => r.id !== id));
    try {
      const result = await fn();
      setFlash(describe(result));
      onCountChange?.(Math.max(0, snapshot.length - 1));
      // Refetch so a newly-approved custom intent shows up as a choice on the
      // remaining cards, and any rule queued by a call meanwhile appears.
      load();
    } catch (err) {
      setItems(snapshot);
      setError(err.message);
    } finally {
      setBusyId(null);
    }
  }

  const handleApprove = (id, edits) =>
    act(
      id,
      () => approveRule(id, edits),
      (r) =>
        r.warning
          ? `Stored as ${r.chunk_id} — but ${r.warning}`
          : `Approved — stored as ${r.chunk_id} in ${
              r.intent ? `the “${r.intent}” section` : "the general pool"
            }. Live from the next call on.`
    );

  const handleDiscard = (id) =>
    act(id, () => discardRule(id), () => "Discarded — it will not be stored.");

  if (loading) {
    return <p className="status-line">Loading the review queue…</p>;
  }

  return (
    <section className="review-queue">
      <header className="review-queue__head">
        <div>
          <h2 className="app-title review-queue__title">Rules awaiting your approval</h2>
          <p className="app-subtitle">
            Nothing the agent proposes reaches memory until you approve it here.
          </p>
        </div>
        <button className="btn btn-quiet" onClick={load} disabled={busyId !== null}>
          Refresh
        </button>
      </header>

      {flash && (
        <div className="review-flash" role="status">
          {flash}
          <button className="review-flash__close" onClick={() => setFlash(null)} aria-label="Dismiss">
            ×
          </button>
        </div>
      )}

      {error && (
        <div className="review-flash review-flash--bad" role="alert">
          {error}
          {error.startsWith("Failed to fetch") || error.includes("NetworkError") ? (
            <span className="review-flash__hint">
              {" "}
              — is the API running? <code>./run.sh demo</code>
            </span>
          ) : null}
        </div>
      )}

      {items.length === 0 ? (
        <div className="review-empty">
          <p className="review-empty__title">Nothing waiting.</p>
          <p className="review-empty__body">
            When a call ends, anything the agent wants to remember from it shows up
            here for you to approve.
          </p>
        </div>
      ) : (
        <div className="review-list">
          {items.map((item) => (
            <ReviewCard
              key={item.id}
              item={item}
              intents={intents}
              busy={busyId === item.id}
              onApprove={handleApprove}
              onDiscard={handleDiscard}
            />
          ))}
        </div>
      )}
    </section>
  );
}
